# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only MegrezMoe model."""

import logging
import re
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    parallel_state,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather,
    attn_tp_reduce_scatter,
    dp_gather_partial,
    dp_scatter,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_local_attention_dp_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, EPMoE
from sglang.srt.layers.moe.ep_moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaAttention
from sglang.srt.utils import (
    BumpAllocator,
    DeepEPMode,
    add_prefix,
    is_cuda,
    is_hip,
)

_is_hip = is_hip()
_is_cuda = is_cuda()


logger = logging.getLogger(__name__)


class AttnForwardMethod(IntEnum):
    # Use multi-head attention
    MHA = auto()

    # Use absorbed multi-latent attention
    MLA = auto()

    # Use multi-head attention, but with KV cache chunked.
    # This method can avoid OOM when prefix lengths are long.
    MHA_CHUNKED_KV = auto()


class MegrezMoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. " "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x, forward_mode: Optional[ForwardMode] = None):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MoEGate(nn.Module):
    def __init__(
        self,
        config,
        prefix: str = "",
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((config.n_routed_experts, config.hidden_size)))
        if config.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(torch.empty((config.n_routed_experts)))
        else:
            self.e_score_correction_bias = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight, None)
        return logits


class MegrezMoeMoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        init_experts: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.n_share_experts_fusion = 0

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. " "Only silu is supported for now.")

        self.gate = MoEGate(config=config, prefix=add_prefix("gate", prefix))

        MoEImpl = (
            DeepEPMoE
            if global_server_args_dict["enable_deepep_moe"]
            else (EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE)
        )

        if init_experts:
            self.experts = MoEImpl(
                num_experts=config.n_routed_experts + self.n_share_experts_fusion,
                top_k=config.num_experts_per_tok + min(self.n_share_experts_fusion, 1),
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                renormalize=config.norm_topk_prob,
                quant_config=quant_config,
                use_grouped_topk=True,
                num_expert_group=config.n_group,
                topk_group=config.topk_group,
                correction_bias=self.gate.e_score_correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                prefix=add_prefix("experts", prefix),
                **(
                    dict(deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]])
                    if global_server_args_dict["enable_deepep_moe"]
                    else {}
                ),
            )
        else:
            self.experts = None

        if config.n_shared_experts is not None and self.n_share_experts_fusion == 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            # disable tp for shared experts when enable deepep moe
            if not global_server_args_dict["enable_deepep_moe"]:
                self.shared_experts = MegrezMoeMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=add_prefix("shared_experts", prefix),
                )
            else:
                self.shared_experts = MegrezMoeMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=add_prefix("shared_experts", prefix),
                    tp_rank=0,
                    tp_size=1,
                )

        if global_server_args_dict["enable_deepep_moe"]:
            # TODO: we will support tp < ep in the future
            self.ep_size = get_tensor_model_parallel_world_size()
            self.num_experts = config.n_routed_experts
            self.top_k = config.num_experts_per_tok
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data if self.gate.e_score_correction_bias is not None else None
            )

            self.deepep_dispatcher = DeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=config.n_routed_experts,
                num_local_experts=config.n_routed_experts // self.tp_size,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]],
                async_finish=True,  # TODO
                return_recv_hook=True,
            )

    def set_experts(self, experts):
        self.experts = experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        pre_gate_hidden_states: torch.Tensor = None,
        forward_mode: Optional[ForwardMode] = None,
    ) -> torch.Tensor:
        if not global_server_args_dict["enable_deepep_moe"]:
            return self.forward_normal(hidden_states, pre_gate_hidden_states)
        else:
            return self.forward_deepep(hidden_states, forward_mode, pre_gate_hidden_states)

    def forward_normal(self, hidden_states: torch.Tensor, pre_gate_hidden_states: torch.Tensor = None) -> torch.Tensor:
        shared_output = self._forward_shared_experts(hidden_states)
        # router_logits: (num_tokens, n_experts)
        if pre_gate_hidden_states is not None:
            router_logits = self.gate(pre_gate_hidden_states)
        else:
            router_logits = self.gate(hidden_states)
        final_hidden_states = (
            self.experts(hidden_states=hidden_states, router_logits=router_logits) * self.routed_scaling_factor
        )
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    def forward_deepep(
        self,
        hidden_states: torch.Tensor,
        forward_mode: ForwardMode,
        pre_gate_hidden_states: torch.Tensor = None,
    ) -> torch.Tensor:
        shared_output = None

        if forward_mode is not None and not forward_mode.is_idle() and hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            if pre_gate_hidden_states is not None:
                router_logits = self.gate(pre_gate_hidden_states)
            else:
                router_logits = self.gate(hidden_states)
            shared_output = self._forward_shared_experts(hidden_states)
            topk_weights, topk_idx = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=True,
                renormalize=self.renormalize,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                correction_bias=self.correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        else:
            topk_idx = torch.full((0, self.top_k), -1, dtype=torch.int, device=hidden_states.device)
            topk_weights = torch.empty((0, self.top_k), dtype=torch.float32, device=hidden_states.device)
        if self.ep_size > 1:
            # TODO(ch-wan): allow users to set num_max_dispatch_tokens_per_rank value
            (
                hidden_states,
                topk_idx,
                topk_weights,
                reorder_topk_ids,
                num_recv_tokens_per_expert,
                seg_indptr,
                masked_m,
                expected_m,
            ) = self.deepep_dispatcher.dispatch(
                hidden_states,
                topk_idx,
                topk_weights,
                forward_mode=forward_mode,
            )
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            reorder_topk_ids=reorder_topk_ids,
            seg_indptr=seg_indptr,
            masked_m=masked_m,
            expected_m=expected_m,
            num_recv_tokens_per_expert=num_recv_tokens_per_expert,
            forward_mode=forward_mode,
        )
        if self.ep_size > 1:
            final_hidden_states = self.deepep_dispatcher.combine(
                final_hidden_states,
                topk_idx,
                topk_weights,
                forward_mode,
            )
        final_hidden_states *= self.routed_scaling_factor

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states

    def _forward_shared_experts(self, hidden_states):
        if self.n_share_experts_fusion == 0:
            return self.shared_experts(hidden_states)
        else:
            return None


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class _FFNInputMode(Enum):
    # The MLP sublayer requires 1/tp_size tokens as input
    SCATTERED = auto()
    # The MLP sublayer requires all tokens as input
    FULL = auto()


@dataclass
class _DecoderLayerInfo:
    is_sparse: bool
    ffn_input_mode: _FFNInputMode


class MegrezMoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = config.original_max_position_embeddings
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)
        bias_o_proj = attention_bias
        # support internlm/internlm3-8b with qkv_bias
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        # By default, Llama uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. parasail-ai/GritLM-7B-vllm)
        self.enable_dp_attention = global_server_args_dict["enable_dp_attention"]
        self.layer_id = layer_id
        self.pre_gate = config.pre_gate
        self.local_dp_size = get_local_attention_dp_size()
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=True,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            bias=attention_bias,
        )

        self.info = self._compute_info(config, layer_id=layer_id, is_nextn=is_nextn)
        previous_layer_info = self._compute_info(config, layer_id=layer_id - 1, is_nextn=False)

        if self.info.is_sparse:
            self.mlp = MegrezMoeMoE(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            if self._enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = MegrezMoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_is_scattered = layer_id > 0 and previous_layer_info.ffn_input_mode == _FFNInputMode.SCATTERED
        self.is_last_layer = self.layer_id == config.num_hidden_layers - 1

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @staticmethod
    def _enable_moe_dense_fully_dp():
        return global_server_args_dict["moe_dense_tp_size"] == 1

    @staticmethod
    def _compute_info(config: PretrainedConfig, layer_id: int, is_nextn: bool):
        is_sparse = is_nextn or (
            config.n_routed_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        )
        ffn_input_mode = (
            _FFNInputMode.SCATTERED
            if (global_server_args_dict["enable_deepep_moe"] and is_sparse)
            or (MegrezMoeDecoderLayer._enable_moe_dense_fully_dp() and not is_sparse)
            else _FFNInputMode.FULL
        )

        return _DecoderLayerInfo(is_sparse=is_sparse, ffn_input_mode=ffn_input_mode)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        if self.info.ffn_input_mode == _FFNInputMode.SCATTERED:
            return self.forward_ffn_with_scattered_input(
                positions, hidden_states, forward_batch, residual, zero_allocator
            )
        elif self.info.ffn_input_mode == _FFNInputMode.FULL:
            return self.forward_ffn_with_full_input(positions, hidden_states, forward_batch, residual, zero_allocator)
        else:
            raise NotImplementedError

    def forward_ffn_with_full_input(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        if self.pre_gate and self.layer_id >= self.config.first_k_dense_replace:
            hidden_states = torch.split(hidden_states, hidden_states.shape[0] // 2, dim=0)
            pre_gate_hidden_states = hidden_states[0]
            hidden_states = hidden_states[1]
        else:
            pre_gate_hidden_states = None

        if hidden_states.shape[0] == 0:
            residual = hidden_states
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

            assert not (
                self.attn_tp_size != 1 and self.input_is_scattered
            ), "moe_layer_freq > 1 is not supported when attn_tp_size > 1"

            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # Gather
        if get_tensor_model_parallel_world_size() > 1:
            # all gather and all reduce
            if self.local_dp_size != 1:
                if self.attn_tp_rank == 0:
                    hidden_states += residual
                hidden_states, local_hidden_states = (
                    forward_batch.gathered_buffer,
                    hidden_states,
                )
                dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
                dp_scatter(residual, hidden_states, forward_batch)
                hidden_states = self.post_attention_layernorm(hidden_states)
            else:
                # hidden_states = tensor_model_parallel_all_reduce(hidden_states)
                hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        post_attention_layernorm_hidden_states = hidden_states.clone()
        # Fully Connected
        if isinstance(self.mlp, MegrezMoeMoE):
            hidden_states = self.mlp(hidden_states, pre_gate_hidden_states=pre_gate_hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        # TODO(ch-wan): use reduce-scatter in MLP to avoid this scatter
        # Scatter
        if self.local_dp_size != 1:
            # important: forward batch.gathered_buffer is used both after scatter and after gather.
            # be careful about this!
            hidden_states, global_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            dp_scatter(hidden_states, global_hidden_states, forward_batch)
        pre_gate_hidden_states = post_attention_layernorm_hidden_states
        if self.pre_gate and self.layer_id < self.config.num_hidden_layers - 1:
            hidden_states = torch.cat([pre_gate_hidden_states, hidden_states], dim=0)

        return hidden_states, residual

    def forward_ffn_with_scattered_input(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:

        if self.pre_gate and self.layer_id >= self.config.first_k_dense_replace:
            hidden_states = torch.split(hidden_states, hidden_states.shape[0] // 2, dim=0)
            pre_gate_hidden_states = hidden_states[0]
            hidden_states = hidden_states[1]
        else:
            pre_gate_hidden_states = None

        if hidden_states.shape[0] == 0:
            residual = hidden_states
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.attn_tp_size != 1 and self.input_is_scattered:
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            attn_tp_all_gather(list(hidden_states.tensor_split(self.attn_tp_size)), local_hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        if self.attn_tp_size != 1:
            if self.input_is_scattered:
                tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                hidden_states = tensor_list[self.attn_tp_rank]
                attn_tp_reduce_scatter(hidden_states, tensor_list)
                if hidden_states.shape[0] != 0:
                    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            else:
                if self.attn_tp_rank == 0:
                    hidden_states += residual
                tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                hidden_states = tensor_list[self.attn_tp_rank]
                attn_tp_reduce_scatter(hidden_states, tensor_list)
                residual = hidden_states
                if hidden_states.shape[0] != 0:
                    hidden_states = self.post_attention_layernorm(hidden_states)
        else:
            if hidden_states.shape[0] != 0:
                hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        post_attention_layernorm_hidden_states = hidden_states.clone()
        
        if not (self._enable_moe_dense_fully_dp() and (not self.info.is_sparse) and hidden_states.shape[0] == 0):
            hidden_states = self.mlp(
                hidden_states,
                pre_gate_hidden_states=forward_batch.pre_gate_hidden_states,
                forward_mode=forward_batch.forward_mode,
            )

        if self.is_last_layer and self.attn_tp_size != 1:
            hidden_states += residual
            residual = None
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            attn_tp_all_gather(list(hidden_states.tensor_split(self.attn_tp_size)), local_hidden_states)
        
        pre_gate_hidden_states = post_attention_layernorm_hidden_states
        if self.pre_gate and self.layer_id < self.config.num_hidden_layers - 1:
            hidden_states = torch.cat([pre_gate_hidden_states, hidden_states], dim=0)
        return hidden_states, residual


class MegrezMoeModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        self.alt_stream = torch.cuda.Stream()
        self.layers = nn.ModuleList(
            [
                MegrezMoeDecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_stream=self.alt_stream,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        for i in range(config.num_hidden_layers):
            layer = self.layers[i]
            shared_layer_idx = (
                i - config.first_k_dense_replace
            ) // config.experts_shared_frequency * config.experts_shared_frequency + config.first_k_dense_replace
            if i >= config.first_k_dense_replace and shared_layer_idx != i:
                layer.mlp.set_experts(self.layers[shared_layer_idx].mlp.experts)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.dp_size = get_local_attention_dp_size()

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        zero_allocator = BumpAllocator(
            # TODO for two-batch-overlap, we need a larger buffer size
            buffer_size=len(self.layers) * 2,
            dtype=torch.float32,
            device=(input_embeds.device if input_embeds is not None else input_ids.device),
        )

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        for i in range(len(self.layers)):
            # expert_distribution_recorder.set_current_layer(i)
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual, zero_allocator)
        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MegrezMoeForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_n_share_experts_fusion()
        self.model = MegrezMoeModel(config, quant_config, prefix=add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"],
        )
        self.logits_processor = LogitsProcessor(config)
        self.dp_size = get_local_attention_dp_size()

    def determine_n_share_experts_fusion(self, architecture: str = "MegrezMoeForCausalLM"):

        self.n_share_experts_fusion = 0
        if self.n_share_experts_fusion > 0:
            if not _is_cuda or self.config.architectures[0] != architecture or self.config.n_routed_experts != 256:
                self.n_share_experts_fusion = 0
                global_server_args_dict["n_share_experts_fusion"] = 0
        elif self.n_share_experts_fusion == 0:
            if (
                _is_cuda
                and torch.cuda.get_device_capability("cuda") >= (9, 0)
                and self.config.architectures[0] == architecture
                and self.config.n_routed_experts == 256
                and (not global_server_args_dict["enable_deepep_moe"])
            ):
                self.n_share_experts_fusion = self.tp_size
                global_server_args_dict["n_share_experts_fusion"] = self.tp_size

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    def post_load_weights(self, is_nextn=False):
        pass

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = 0 if self.config.num_hidden_layers == 1 else self.config.num_hidden_layers
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        if self.n_share_experts_fusion > 0:
            weights_list = list(weights)
            weights_dict = dict(weights_list)
            if self.quant_config is None or self.quant_config.get_name() == "w8a8_int8":
                suffix_list = [
                    "down_proj.weight",
                    "down_proj.weight_scale",
                    "gate_proj.weight",
                    "gate_proj.weight_scale",
                    "up_proj.weight",
                    "up_proj.weight_scale",
                ]
            else:
                suffix_list = [
                    "down_proj.weight",
                    "down_proj.weight_scale_inv",
                    "gate_proj.weight",
                    "gate_proj.weight_scale_inv",
                    "up_proj.weight",
                    "up_proj.weight_scale_inv",
                ]
            names_to_remove = []

            moe_layers = (
                range(
                    self.config.first_k_dense_replace,
                    self.config.num_hidden_layers,
                    self.config.moe_layer_freq,
                )
                if not is_nextn
                else [nextn_layer_id]
            )

            for moe_layer in tqdm(
                moe_layers,
                desc=f"Cloning {self.n_share_experts_fusion} " "replicas of the shared expert into MoE",
            ):
                for suffix in suffix_list:
                    shared_expert_weight_name = f"model.layers.{moe_layer}.mlp.shared_experts.{suffix}"
                    for num_repeat in range(self.n_share_experts_fusion):
                        weights_list.append(
                            (
                                f"model.layers.{moe_layer}."
                                f"mlp.experts."
                                f"{self.config.n_routed_experts + num_repeat}"
                                f".{suffix}",
                                weights_dict[shared_expert_weight_name],
                            )
                        )
                    names_to_remove += [shared_expert_weight_name]
            weights = [w for w in weights_list if w[0] not in names_to_remove]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        MoEImpl = (
            DeepEPMoE
            if global_server_args_dict["enable_deepep_moe"]
            else (EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE)
        )
        expert_params_mapping = MoEImpl.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.n_share_experts_fusion,
        )

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (self.config.q_lora_rank is not None)
        cached_a_proj = {} if fuse_qkv_a_proj else None

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if not is_nextn:
                if hasattr(self.config, "num_nextn_predict_layers"):
                    num_nextn_layers = self.config.num_nextn_predict_layers
                    if num_nextn_layers > 0 and name.startswith("model.layers"):
                        name_list = name.split(".")
                        if len(name_list) >= 3 and int(name_list[2]) >= self.config.num_hidden_layers:
                            continue
            else:
                if not name.startswith(nextn_layer_prefix):
                    continue

                # Use shared head and embed weights from target model
                if "shared_head.head" in name or "embed_tokens" in name:
                    continue

                is_decoder = True
                # For nextn specific weights
                for weight_name in nextn_spec_weight_names:
                    if weight_name in name:
                        name = name.replace(nextn_layer_prefix, "model")
                        is_decoder = False
                        break
                # For decoder layer weights
                if is_decoder:
                    name = name.replace(nextn_layer_prefix, "model.decoder")

            if "rotary_emb.inv_freq" in name:
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                pattern = r"mlp\.\d+\.experts"
                if ("mlp.experts." in name or re.search(pattern, name)) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if fuse_qkv_a_proj and ("q_a_proj" in name or "kv_a_proj_with_mqa" in name):
                        cached_a_proj[name] = loaded_weight
                        q_a_proj_name = name if "q_a_proj" in name else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                        kv_a_proj_name = (
                            name if "kv_a_proj_with_mqa" in name else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                        )

                        # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                        if q_a_proj_name in cached_a_proj and kv_a_proj_name in cached_a_proj:

                            q_a_proj_weight = cached_a_proj[q_a_proj_name]
                            kv_a_proj_weight = cached_a_proj[kv_a_proj_name]
                            fused_weight = torch.cat([q_a_proj_weight, kv_a_proj_weight], dim=0)

                            param_name = name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                            param = params_dict[param_name]

                            weight_loader = getattr(param, "weight_loader", default_weight_loader)
                            weight_loader(param, fused_weight)
                            cached_a_proj.pop(q_a_proj_name)
                            cached_a_proj.pop(kv_a_proj_name)
                    else:
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
        self.post_load_weights(is_nextn=is_nextn)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_spec_layer_idx_from_weight_name(config: PretrainedConfig, weight_name: str) -> Optional[int]:
    if hasattr(config, "num_nextn_predict_layers") and (config.num_nextn_predict_layers > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx+i}."):
                return layer_idx + i
    return None


EntryClass = [MegrezMoeForCausalLM]

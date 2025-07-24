from vllm import LLM
from vllm import SamplingParams, ModelRegistry
from megrez_moe import MegrezMoeForCausalLM
from transformers import AutoTokenizer
import sys

ModelRegistry.register_model(
    "MegrezMoeForCausalLM",
    MegrezMoeForCausalLM,
)


def get_vllm_model(model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.0,
        ignore_eos=False,
        max_tokens=4096,
    )

    return llm, sampling_params, tokenizer


def demo_generate(llm, sampling_params):
    outputs = llm.generate(
        prompts=["世界上最高的山峰是"],
        sampling_params=sampling_params,
    )
    for output in outputs:
        print(output.outputs[0].text)


def demo_chat(llm, sampling_params, tokenizer):
    messages = [
        {"role": "user", "content": "世界上最高的山峰是哪座？"},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    outputs = llm.generate(
        prompts=[prompt],
        sampling_params=sampling_params,
    )
    for output in outputs:
        print(output.outputs[0].text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer_vllm_offline.py <model_path>")
        exit(1)
    model_path = sys.argv[1]

    llm, sampling_params, tokenizer = get_vllm_model(model_path)
    demo_generate(llm, sampling_params)
    demo_chat(llm, sampling_params, tokenizer)

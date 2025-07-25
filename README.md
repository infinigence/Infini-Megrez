<div align="center">
  <img src="./assets/megrez-logo.png" alt="Megrez Logo" width="400" />

  <br>

  <a href="https://huggingface.co/Infinigence/Megrez2-3x7B-A3B-Preview">
    <b>🤗 Hugging Face</b>
  </a> &nbsp;|&nbsp;
  <a href="https://www.modelscope.cn/models/InfiniAI/Megrez2-3x7B-A3B-Preview">
    <b>🤖 Model Scope</b>
  </a> &nbsp;|&nbsp;
  <a href="./docs/tech_report.pdf">
    <b>📄 Tech Report</b>
  </a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/spaces/Infinigence/Megrez2-3x7B-A3B-Preview">
    <b>💻 Demo</b>
  </a> &nbsp;|&nbsp;
  <a href="./assets/wechat-official.jpg">
    <b>💬 WeChat Official</b>
  </a> &nbsp;

  <br>

  <strong>中文 | [English](./README_EN.md)</strong>

</div>

# 更新日志

- [2025.07.24] 发布 [Megrez2-3x7B-A3B-Preview](https://github.com/infinigence/Infini-Megrez/tree/main) 专为终端设备设计的大模型，兼顾MoE的精度杠杆与Dense的总参数量友好。

- [2024.12.16] 发布 [Megrez-3B-Omni](https://huggingface.co/Infinigence/Megrez-3B-Omni) 基于Megrez-3B-Instruct 扩展，同时具备图片、文本、音频三种模态数据的理解分析能力。

- [2024.12.16] 发布 [Megrez-3B-Instruct](https://github.com/infinigence/Infini-Megrez/tree/Megrez-3B) 模型能够比肩 Yi-1.5-6B-Chat、 Qwen2-7B-Instruct、 GLM-4-9B-Chat 和 Baichuan2-13B-Chat 等多个6B-13B参数量的模型。

# 模型下载

<div align="center">

| HuggingFace | ModelScope | Wisemodel |
|:---:|:---:|:---:|
| [Megrez2-3x7B-A3B-Preview](https://huggingface.co/Infinigence/Megrez2-3x7B-A3B-Preview) | [Megrez2-3x7B-A3B-Preview](https://www.modelscope.cn/models/InfiniAI/Megrez2-3x7B-A3B-Preview) | [Megrez2-3x7B-A3B-Preview](https://wisemodel.cn/models/Infinigence/Megrez2-3x7B-A3B-Preview) |
| [Megrez-3B-Omni](https://huggingface.co/Infinigence/Megrez-3B-Omni) | [Megrez-3B-Omni](https://www.modelscope.cn/models/InfiniAI/Megrez-3B-Omni) | [Megrez-3B-Omni](https://www.wisemodel.cn/models/Infinigence/Megrez-3B-Omni) |
| [Megrez-3B-Instruct](https://huggingface.co/Infinigence/Megrez-3B-Instruct) | [Megrez-3B-Instruct](https://www.modelscope.cn/models/InfiniAI/Megrez-3b-Instruct) | [Megrez-3B-Instruct](https://www.wisemodel.cn/models/Infinigence/Megrez-3B-Instruct) |

</div>


# Megrez2-3x7B-A3B-Preview

## 模型简介

Megrez2-3x7B-A3B-Preview 是专为终端设备设计的大模型，兼顾MoE的精度杠杆与Dense的总参数量友好。本次发布的为Megrez 2.0预览版本，训练数据量5T Tokens，未来我们计划完成更大规模的数据训练，并提高模型的推理和Agent能力，正式版本预计今年年内发布。

## 基础信息

<div align="center">

| | |
|:---:|:---:|
| **Architecture** | Mixture-of-Experts (MoE) |
| **Total Parameters** | 3x7B |
| **Activated Parameters** | 3B |
| **Experts Shared Frequency**| 3 |
| **Number of Layers** (Dense layer included) | 31 |
| **Number of Dense Layers** | 1 |
| **Attention Hidden Dimension** | 2048 |
| **MoE Hidden Dimension** (per Expert) | 1408 |
| **Number of Attention Heads** | 16 |
| **Number of Experts** | 64 |
| **Selected Experts per Token** | 6 |
| **Number of Shared Experts** | 4 |
| **Vocabulary Size** | 128,880 |
| **Context Length** | 32K |
| **Base Frequency of RoPE** | 1,000,000 |
| **Attention Mechanism** | GQA |
| **Activation Function** | SwiGLU |
</div>

## 性能测试

我们使用开源评测工具 [OpenCompass](https://github.com/open-compass/opencompass) 对 Megrez2-3x7B-A3B-Preview 进行了评测，部分评测结果如下表所示。

<div align="center">
<table>
<thead>
<tr>
<th align="center">Benchmark</th>
<th align="center">Metric</th>
<th align="center"><sup>Megrez2-3x7B<br>-A3B-Preview</sup></th>
<th align="center"><sup>Qwen2.5-3B</sup></th>
<th align="center"><sup>Qwen2.5-7B</sup></th>
<th align="center"><sup>Qwen3-4B</sup></th>
<th align="center"><sup>Qwen3-8B</sup></th>
<th align="center"><sup>Phi-4-mini</sup></th>
<th align="center"><sup>Gemma-3-4B</sup></th>
<th align="center"><sup>GPT-4o-mini <br><sup>2024-07-18</sup></sup></th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">Activate Params (B)</td>
<td align="center"></td>
<td align="center">3.0</td>
<td align="center">3.1</td>
<td align="center">7.6</td>
<td align="center">4.0</td>
<td align="center">8.2</td>
<td align="center">3.8</td>
<td align="center">4.3</td>
<td align="center">-</td>
</tr>
<tr>
<td align="center">Stored Params (B)</td>
<td align="center"></td>
<td align="center">7.5</td>
<td align="center">3.1</td>
<td align="center">7.6</td>
<td align="center">4.0</td>
<td align="center">8.2</td>
<td align="center">3.8</td>
<td align="center">4.3</td>
<td align="center">-</td>
</tr>
<tr>
<td align="center" colspan=9><strong>General Tasks</strong></td>
</tr>
<tr>
<td align="center">C-EVAL</td>
<td align="center">EM</td>
<td align="center"><strong>91.7</strong></td>
<td align="center">68.2</td>
<td align="center">76.2</td>
<td align="center">72.2</td>
<td align="center">77.9</td>
<td align="center">40.0</td>
<td align="center">-</td>
<td align="center">66.3</td>
</tr>
<tr>
<td align="center">MMLU-Pro</td>
<td align="center">EM</td>
<td align="center"><strong>67.6</strong></td>
<td align="center">43.7</td>
<td align="center">56.3</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">52.8</td>
<td align="center">43.6</td>
<td align="center">-</td>
</tr>
<td align="center" colspan=9><strong>Instruction Tasks</strong></td>
<tr>
<td align="center">IF-Eval</td>
<td align="center">Prompt Strict</td>
<td align="center">80.2</td>
<td align="center">58.2</td>
<td align="center">71.2</td>
<td align="center">81.2</td>
<td align="center">83.0</td>
<td align="center">68.6</td>
<td align="center"><strong>90.2</strong></td>
<td align="center">80.4</td>
</tr>
<td align="center" colspan=9><strong>Math & STEM Tasks</strong></td>
<tr>
<td align="center">MATH-500</td>
<td align="center">EM</td>
<td align="center">81.6</td>
<td align="center">65.9</td>
<td align="center">75.5</td>
<td align="center">84.8</td>
<td align="center"><strong>87.4</strong></td>
<td align="center">64.0</td>
<td align="center">75.6</td>
<td align="center">78.2</td>
</tr>
<tr>
<td align="center">GSM8K</td>
<td align="center">EM</td>
<td align="center">83.6</td>
<td align="center">86.7</td>
<td align="center">91.6</td>
<td align="center">-</td>
<td align="center"><strong>93.2</strong></td>
<td align="center">88.6</td>
<td align="center">89.2</td>
<td align="center">-</td>
</tr>
<td align="center" colspan=9><strong>Coding Tasks</strong></td>
<tr>
<td align="center">HumanEval</td>
<td align="center">Pass@1</td>
<td align="center">74.4</td>
<td align="center">74.4</td>
<td align="center">84.8</td>
<td align="center">-</td>
<td align="center"><strong>85.9</strong></td>
<td align="center">74.4</td>
<td align="center">71.3</td>
<td align="center">87.2</td>
</tr>
<tr>
<td align="center">MBPP</td>
<td align="center">Pass@1</td>
<td align="center"><strong>88.0</strong></td>
<td align="center">72.7</td>
<td align="center">79.2</td>
<td align="center">-</td>
<td align="center">77.0</td>
<td align="center">65.3</td>
<td align="center">63.2</td>
<td align="center">-</td>
</tr>
</tbody>
</table>
</div>

## 如何运行

### Transformers

推荐使用最新版本的 `transformers` 或者 `transformers>=4.52.4` 的版本。
以下是一个非常简单的代码片段示例，展示如何运行 Megrez2-3x7B-A3B-Preview 模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

path = "Infinigence/Megrez2-3x7B-A3B-Preview"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

messages = [
    {"role": "user", "content": "世界上最高的山峰是哪座？"},
]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

model_outputs = model.generate(
    model_inputs,
    do_sample=True,
    max_new_tokens=1024
)

output_token_ids = [
    model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
]

responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
print(responses)

# 世界上最高的山峰是珠穆朗玛峰（Mount Everest），位于喜马拉雅山脉的中尼边境。珠穆朗玛峰的海拔高度为8,848.86米（29,031.7英尺），这一数据是由中国和尼泊尔在2020年共同宣布的最新测量结果。珠穆朗玛峰不仅是登山爱好者的圣地，也是地理和科学研究的重要对象。
```

### ModelScope

`ModelScope` 采用了与 `Transformers` 类似（但不完全一致）的编程接口。对于基础使用，仅需将上面代码第一行做如下修改：

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
```

### llama.cpp

即将到来...

## 如何部署

### vLLM

推荐 `vllm>=0.9.2` 的版本

#### vLLM 离线
```shell
cd demo/vllm
export MODEL_PATH="Infinigence/Megrez2-3x7B-A3B-Preview"
python3 infer_vllm_offline.py $MODEL_PATH
```
#### vLLM 在线
在终端中启动vLLM服务，命令如下
```shell
cd demo/vllm
export MODEL_PATH="Infinigence/Megrez2-3x7B-A3B-Preview"
python3 serve_llm_online.py serve $MODEL_PATH --gpu-memory-utilization 0.9 --served-model-name megrez-moe --trust_remote_code
```

现在，可以通过curl发送请求
```shell
curl --location 'http://localhost:8000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-123456' \
--data '{
    "model": "megrez-moe",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "世界上最高的山峰是哪座？"
                }
            ]
        }
    ]
}'
```

### SGLang

推荐 `sglang>=0.4.9.post2` 的版本
```shell
cd demo/sglang
export MODEL_PATH="Infinigence/Megrez2-3x7B-A3B-Preview" 
python3 infer_sglang_offline.py $MODEL_PATH
```


## 最佳实践

为了获得最佳性能，建议以下设置：

1. 采样参数：推荐使用 Temperature=0.7 和 TopP=0.9 。

2. 标准化输出格式：在基准测试时，我们建议使用提示来标准化模型输出，比如：
    * 数学问题：在提示中包含“请逐步推理，并将最终答案放在\boxed{}中。”
    * 选择题：在提示中添加以下 JSON 结构以标准化响应：“请在 answer 字段中仅以选择字母的形式显示您的选择，例如 "answer": "C" 。”

# Megrez-3B-Omni

详细信息请查看 [Infini-Megrez-Omni 仓库](https://github.com/infinigence/Infini-Megrez-Omni)

# Megrez-3B-Instruct

详细信息请查看 [Megrez-3B 分支](https://github.com/infinigence/Infini-Megrez/tree/Megrez-3B)

# 许可声明

我们所有的开源模型均采用Apache 2.0协议授权。

# 引用信息

如果您觉得我们的代码和模型有用，请引用以下信息。

```bibtex
@misc{li2025megrez2technicalreport,
      title={Megrez2 Technical Report}, 
      author={Boxun Li and Yadong Li and Zhiyuan Li and Congyi Liu and Weilin Liu and Guowei Niu and Zheyue Tan and Haiyang Xu and Zhuyu Yao and Tao Yuan and Dong Zhou and Yueqing Zhuang and Bo Zhao and Guohao Dai and Yu Wang},
      year={2025},
      eprint={2507.17728},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.17728}, 
}
```

# 联系我们

如果您有任何问题，请随时提交GitHub issue或联系[微信群组](./assets/wechat-group.jpg)。
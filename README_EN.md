<div align="center">
  <img src="./assets/megrez-logo.png" alt="Megrez Logo" width="400" />

  <br>

  <a href="https://huggingface.co/Infinigence/Megrez2-3x7B-A3B">
    <b>🤗 Hugging Face</b>
  </a> &nbsp;|&nbsp;
  <a href="https://www.modelscope.cn/models/InfiniAI/Megrez2-3x7B-A3B">
    <b>🤖 Model Scope</b>
  </a> &nbsp;|&nbsp;
  <a href="./docs/tech_report.pdf">
    <b>📄 Tech Report</b>
  </a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/spaces/Infinigence/Megrez2-3x7B-A3B">
    <b>💻 Demo</b>
  </a> &nbsp;|&nbsp;
  <a href="./assets/wechat-official.jpg">
    <b>💬 WeChat Official</b>
  </a> &nbsp;

  <br>

  <strong>[中文](./README.md) | English</strong>

</div>

# Changelog

- [2025.09.15] Release [Megrez2-3x7B-A3B](https://github.com/infinigence/Infini-Megrez/tree/main) Official version and model structure is consistent with the preview version, the total amount of training data increased from 5T to 8T, and the performances on benchmark are more balanced.

- [2025.07.24] Released [Megrez2-3x7B-A3B-Preview](https://github.com/infinigence/Infini-Megrez/tree/main) Preview vision, device native large language model combines MoE architecture's accuracy and Dense models' compactness.

- [2024.12.16] Released [Megrez-3B-Omni](https://huggingface.co/Infinigence/Megrez-3B-Omni) This model is an extension of the Megrez-3B-Instruct model and supports analysis of image, text, and audio modalities.

- [2024.12.16] Released [Megrez-3B-Instruct](https://github.com/infinigence/Infini-Megrez/tree/Megrez-3B) This model is comparable to several models with 6B-13B parameters like Yi-1.5-6B-Chat, Qwen2-7B-Instruct, GLM-4-9B-Chat and Baichuan2-13B-Chat.

# Model Downloads

<div align="center">

| HuggingFace | ModelScope | Wisemodel |
|:---:|:---:|:---:|
| [Megrez2-3x7B-A3B](https://huggingface.co/Infinigence/Megrez2-3x7B-A3B) | [Megrez2-3x7B-A3B](https://www.modelscope.cn/models/InfiniAI/Megrez2-3x7B-A3B) | [Megrez2-3x7B-A3B](https://wisemodel.cn/models/Infinigence/Megrez2-3x7B-A3B) |
| [Megrez2-3x7B-A3B-Preview](https://huggingface.co/Infinigence/Megrez2-3x7B-A3B-Preview) | [Megrez2-3x7B-A3B-Preview](https://www.modelscope.cn/models/InfiniAI/Megrez2-3x7B-A3B-Preview) | [Megrez2-3x7B-A3B-Preview](https://wisemodel.cn/models/Infinigence/Megrez2-3x7B-A3B-Preview) |
| [Megrez-3B-Omni](https://huggingface.co/Infinigence/Megrez-3B-Omni) | [Megrez-3B-Omni](https://www.modelscope.cn/models/InfiniAI/Megrez-3B-Omni) | [Megrez-3B-Omni](https://www.wisemodel.cn/models/Infinigence/Megrez-3B-Omni) |
| [Megrez-3B-Instruct](https://huggingface.co/Infinigence/Megrez-3B-Instruct) | [Megrez-3B-Instruct](https://www.modelscope.cn/models/InfiniAI/Megrez-3b-Instruct) | [Megrez-3B-Instruct](https://www.wisemodel.cn/models/Infinigence/Megrez-3B-Instruct) |

</div>


# Megrez2-3x7B-A3B

## Introduction

Megrez2-3x7B-A3B is a device native large language model. Megrez2 takes advantages of both the accuracy of Mixture-of-Experts (MoE) architecture and the compact size of Dense models. This official model was trained on 8T Tokens of data. In the future, we plan to improve the model's reasoning and agent capabilities.

## Model Card

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

## Performance

We evaluated Megrez2-3x7B-A3B using the open-source evaluation tool [OpenCompass](https://github.com/open-compass/opencompass) on several important benchmarks. Some of the evaluation results are shown in the table below.

<div align="center">
<table>
<thead>
<tr>
<th align="center">Benchmark</th>
<th align="center">Metric</th>
<th align="center"><sup>Megrez2-3x7B<br>-A3B</sup></th>
<th align="center"><sup>Megrez2-3x7B<br>-A3B-Preview</sup></th>
<th align="center"><sup>SmallThinker-21B<br>-A3B-Instruct</sup></th>
<th align="center"><sup>Qwen3-30B-A3B</sup></th>
<th align="center"><sup>Qwen3-8B</sup></th>
<th align="center"><sup>Qwen3-4B<br>-Instruct-2507</sup></th>
<th align="center"><sup>Phi4-14B<br>(nothink)</sup></th>
<th align="center"><sup>Gemma3-12B</sup></th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">Activate Params (B)</td>
<td align="center"></td>
<td align="center">3.0</td>
<td align="center">3.0</td>
<td align="center">3.0</td>
<td align="center">3.3</td>
<td align="center">8.2</td>
<td align="center">4.0</td>
<td align="center">14.7</td>
<td align="center">12.2</td>
</tr>
<tr>
<td align="center">Stored Params (B)</td>
<td align="center"></td>
<td align="center">7.5</td>
<td align="center">7.5</td>
<td align="center">21.5</td>
<td align="center">30.5</td>
<td align="center">8.2</td>
<td align="center">4.0</td>
<td align="center">14.7</td>
<td align="center">12.2</td>
</tr>
<tr>
<td align="center">MMLU</td>
<td align="center">EM</td>
<td align="center">85.4</td>
<td align="center"><strong>87.5</strong></td>
<td align="center">84.4</td>
<td align="center">85.1</td>
<td align="center">81.8</td>
<td align="center">-</td>
<td align="center">84.6</td>
<td align="center">78.5</td>
</tr>
<tr>
<td align="center">GPQA</td>
<td align="center">EM</td>
<td align="center"><strong>58.8</strong></td>
<td align="center">28.8</td>
<td align="center">55.0</td>
<td align="center">44.4</td>
<td align="center">38.9</td>
<td align="center">62</td>
<td align="center">55.5</td>
<td align="center">34.9</td>
</tr>
<tr>
<td align="center">IFEval</td>
<td align="center">Inst<br>loose</td>
<td align="center"><strong>87.7</strong></td>
<td align="center">80.2</td>
<td align="center">85.8</td>
<td align="center">84.3</td>
<td align="center">83.9</td>
<td align="center">83.4</td>
<td align="center">63.2</td>
<td align="center">74.7</td>
</tr>
<tr>
<td align="center">MATH-500</td>
<td align="center">EM</td>
<td align="center"><strong>87.2</strong></td>
<td align="center">81.6</td>
<td align="center">82.4</td>
<td align="center">84.4</td>
<td align="center">81.6</td>
<td align="center">-</td>
<td align="center">80.2</td>
<td align="center">82.4</td>
</tr>
</tbody>
</table>
</div>

## How to Run

### Transformers

The latest version of `transformers` is recommended or `transformers>=4.52.4` is required.
The following contains a code snippet illustrating how to use the model generate content based on given inputs.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

path = "Infinigence/Megrez2-3x7B-A3B"
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

`ModelScope` adopts Python API similar to (though not entirely identical to) `Transformers`. For basic usage, simply modify the first line of the above code as follows:

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
```

### llama.cpp

llama.cpp enables LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware. Now supported, please refer to the [support-megrez branch](https://github.com/infinigence/llama.cpp/tree/support-megrez) for details.


## How to Deploy 

### vLLM

Version `vllm>=0.10.1` is required. In the current version, a patch replacement for relevant vllm files is necessary. Going forward, we will submit a pull request to merge this modification into vllm's official version as soon as possible.

1. find your vllm installation path
```python
import vllm
print(vllm.__file__)
```

2. replace vllm related files
```shell
cp ./demo/vllm/patch/layer.py <vllm_install_path>/model_executor/layers/fused_moe/
```

#### vLLM offline
```shell
cd demo/vllm
export MODEL_PATH="Infinigence/Megrez2-3x7B-A3B"
python3 infer_vllm_offline.py $MODEL_PATH
```
#### vLLM online
To start the vLLM service in the terminal, the command is as follows:
```shell
cd demo/vllm
export MODEL_PATH="Infinigence/Megrez2-3x7B-A3B"
python3 serve_llm_online.py serve $MODEL_PATH --gpu-memory-utilization 0.9 --served-model-name megrez-moe --trust_remote_code
```

Now, you can send requests via curl
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

`sglang>=0.4.9.post2` is recommended
```shell
cd demo/sglang
export MODEL_PATH="Infinigence/Megrez2-3x7B-A3B" 
python3 infer_sglang_offline.py $MODEL_PATH
```


## Best Practice

To achieve optimal performance, we recommend the following settings:

1. Sampling Parameters: we suggest using Temperature=0.7 and TopP=0.9 .
   
2. Standardize Output Format: We recommend using prompts to standardize model outputs when benchmarking.
    * Math Problems: Include "Please reason step by step, and put your final answer within \boxed{}." in the prompt.
    * Multiple-Choice Questions: Add the following JSON structure to the prompt to standardize responses: "Please show your choice in the answer field with only the choice letter, e.g., "answer": "C"."


# Megrez-3B-Omni

For detailed information, please refer to the [Infini-Megrez-Omni](https://github.com/infinigence/Infini-Megrez-Omni)

# Megrez-3B-Instruct

For detailed information, please refer to the [Megrez-3B branch](https://github.com/infinigence/Infini-Megrez/tree/Megrez-3B)

# License Agreement

All our open-weight models are licensed under Apache 2.0. 

# Citation

If you find our work helpful, feel free to give us a cite.

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

# Contact

If you have any questions, please feel free to submit a GitHub issue or contact [WeChat groups](./assets/wechat-group.jpg).
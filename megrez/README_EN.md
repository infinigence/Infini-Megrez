# Megrez-3B-Instruct

<p align="center">
    <img src="../assets/megrez_logo.png" width="400"/>
<p>
<p align="center">
        🔗 <a href="https://github.com/infinigence/Infini-Megrez">GitHub</a>&nbsp&nbsp | &nbsp&nbsp
        🏠 <a href="https://cloud.infini-ai.com/genstudio/model/mo-c73owqiotql7lozr">Infini-AI mass</a>&nbsp&nbsp | &nbsp&nbsp
        📖 <a href="../assets/wechat-official.jpg">WeChat Official</a>&nbsp&nbsp | &nbsp&nbsp
        💬 <a href="../assets/wechat-group.jpg">WeChat Groups</a>&nbsp&nbsp   
</p>
<h4 align="center">
    <p>
        <a href="https://github.com/infinigence/Infini-Megrez/blob/main/megrez/README.md">中文</a> | <b>English</b>
    <p>
</h4>

## Introduction
Megrez-3B-Instruct is a large language model trained by [Infinigence AI](https://cloud.infini-ai.com/platform/ai). Megrez-3B aims to provide a fast inference, compact, and powerful edge-side intelligent solution through software-hardware co-design. Megrez-3B has the following advantages:
1. High Accuracy: Megrez-3B successfully compresses the capabilities of the previous 14 billion model into a 3 billion size, and achieves excellent performance on mainstream benchmarks.
2. High Speed: A smaller model does not necessarily bring faster speed. Megrez-3B ensures a high degree of compatibility with mainstream hardware through software-hardware co-design, leading an inference speedup up to 300% compared to previous models of the same accuracy.
3. Easy to Use: In the beginning, we had a debate about model design: should we design a unique but efficient model structure, or use a classic structure for ease of use? We chose the latter and adopt the most primitive LLaMA structure, which allows developers to deploy the model on various platforms without any modifications and minimize the complexity of future development.
4. Rich Applications: We have provided a fullstack WebSearch solution. Our model is functionally trained on web search tasks, enabling it to automatically determine the timing of search invocations and provide better summarization results. The complete deployment code is released on [github](https://github.com/infinigence/InfiniWebSearch).

## Model Card
* Model name: Megrez-3B-Instruct
* Architecture: Llama-2 with GQA  
* Context length: 32K tokens  
* Params (Total): 2.92B  
* Params (Backbone only, w/o Emb or Softmax): 2.29B  
* Vocab Size: 122880  
* Training data: 3T tokens  
* Supported languages: Chinese & English  

## Performance
We evaluated Megrez-3B-Instruct using the open-source evaluation tool [OpenCompass](https://github.com/open-compass/opencompass) on several important benchmarks. Some of the evaluation results are shown in the table below.

The scatter plot of speed, accuracy and model size is roughly as follows. The point size represents the number of model parameters.
![MMLU](../assets/mmlu_en.png)
![MTBench](../assets/chat_en.png)

For more metric data, please refer to 🤗 [Megrez-3B-Instruct](https://huggingface.co/Infinigence/Megrez-3B-Instruct/blob/main/README_EN.md)
### Infernce Speed
|                | Decode Speed(tokens/s) |
|:--------------:|:-----------------:|
| Megrez-3B-Instruct    |      2329.4        |
| Qwen2.5-1.5B          |      3318.8        |
| Qwen2-1.5B            |      3299.5        |
| Qwen2.5-3B-Instruct   |      2248.3        |
| MiniCPM-2B            |      1930.8        |
| Qwen1.5-4B            |      1837.9        |
| Phi-3.5-mini-instruct |      1559.1        |
| Yi-1.5-6B             |      1542.7        |
| Qwen1.5-7B            |      1282.3        |
| Qwen2.5-7B            |      1283.4        |
| Qwen2-7B              |      1279.4        |
| Meta-Llama-3.1-8B     |      1255.9        |
| GLM-4-9B-chat         |      1076.1        |
| MiniCPM3-4B           |      901.1         |
| Baichuan2-13B-Base    |      756.7         |
| Qwen1.5-14B           |      735.6         |

- For details on the configuration for measuring speed, please refer to <a href="https://huggingface.co/Infinigence/Megrez-3B-Instruct/blob/main/README_SPEED.md">README_SPEED.md</a>

## WebSearch
We have provided a fullstack WebSearch solution which has the following advantages:
1. Automatically determine the timing of search invocations: Switch between search and conversation automatically without tendency.
2. In-Context understanding: Generate reasonable search queries or process search results based on multi-turn conversations.
3. Structured output: Each conclusion is attributed to its source for easy verification.
4. One model with two usages: Enable the WebSearch ability by changing system prompt. Or you can use it as a classic LLM.

Our model is functionally trained on web search tasks. Users can build their own Kimi or Perplexity based on this feature, which overcomes the hallucination issues and gets update knowledge.
![WebSearchDemo](../assets/websearch_demo.gif)

## Quick Start
### Online Experience
[MaaS](https://cloud.infini-ai.com/genstudio/model/mo-c73owqiotql7lozr)(recommend) 

### Inference Parameters
- For chat, text generation, and other tasks that benefit from diversity, we recommend to use the inference parameter temperature=0.7.
- For mathematical and reasoning tasks, we recommend to use the inference parameter temperature=0.2 for better determinacy.

### 🤗 Huggingface Inference
``` python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

path = "Infinigence/Megrez-3B-Instruct"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

messages = [
    {"role": "user", "content": "How to make braised chicken in brown sauce?"},
]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

model_outputs = model.generate(
    model_inputs,
    do_sample=True,
    max_new_tokens=1024,
    top_p=0.9,
    temperature=0.2
)

output_token_ids = [
    model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
]

responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
print(responses)
```

### 🤖 ModelScope Inference
```python
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM

model_path = "Infinigence/Megrez-3B-Instruct"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_romote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

messages = [{"role": "user", "content": "How to make braised chicken in brown sauce?"}]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)
model_outputs = model.generate(
    model_inputs,
    do_sample = True,
    max_new_tokens=2048,
    top_p=0.9,
    temperature=0.2
)

output_token_ids = [
    model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
]
responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
print(responses)
```


### 💻 vLLM Inference
- Installation
```bash
# Install vLLM with CUDA 12.1.
pip install vllm
```
- Example code
```python
python inference/inference_vllm.py --model_path <hf_repo_path> --prompt_path prompts/prompt_demo.txt
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_name = "Infinigence/Megrez-3B-Instruct"
prompt = [{"role": "user", "content": "How to make braised chicken in brown sauce?"}]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

llm = LLM(
    model=model_name,
    trust_remote_code=True,
    tensor_parallel_size=1
)
sampling_params = SamplingParams(top_p=0.9, temperature=0.2, max_tokens=1024, repetition_penalty=1.02)

outputs = llm.generate(prompts=input_text, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
```

## License and Statement
- License: Our models are released under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).
- Hallucination: LLMs inherently suffer from hallucination issues. Users are advised not to fully trust the content generated by the model. If accurate outputs are required, we recommend utilizing our WebSearch framework as detailed in [InfiniWebSearch](https://github.com/paxionfull/InfiniWebSearch).
- Mathematics & Reasoning: Small LLMs tend to produce more incorrect calculations or flawed reasoning chains in tasks like mathematics and reasoning. Notably, the softmax distribution of Small LLMs is less sharp compared to LLMs, making them more sensitive to inconsistent reasoning results, especially under higher temperature settings. We recommend lowering the temperature or verifying through multiple inference attempts in deterministic tasks such as mathematics and logical reasoning.
- System Prompt: Like other models, we recommend to use the default system prompt in the configuration file's chat_template for a stable and balanced experience. If users have specific domain needs, we recommend making appropriate fine-tuning based on this model as needed.
- Values & Safety: We have made great effort to ensure the compliance of the training data. However, unforeseen issues may still arise. Users should take full responsibility for evaluating and verifying it on their own when using content generated by our model.

# launch the offline engine
from transformers import AutoTokenizer

import sglang as sgl
from megrez_moe import MegrezMoeForCausalLM
from sglang.srt.models.registry import ModelRegistry

ModelRegistry.models["MegrezMoeForCausalLM"] = MegrezMoeForCausalLM
import sys


def get_sglang_model(model_path):
    llm = sgl.Engine(model_path=model_path, trust_remote_code=True, tp_size=1)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "ignore_eos": False,
        "max_new_tokens": 4096,
    }
    return llm, sampling_params, tokenizer


def demo_generate(llm, sampling_params):
    prompt = "世界上最高的山峰是"
    output_dict = llm.generate(prompt, sampling_params, return_logprob=True)
    print(output_dict["text"])


def demo_chat(llm, sampling_params, tokenizer):
    messages = [
        {"role": "user", "content": "世界上最高的山峰是哪座？"},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    output_dict = llm.generate(prompt, sampling_params, return_logprob=True)
    print(output_dict["text"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer_sglang_offline.py <model_path>")
        exit(1)
    model_path = sys.argv[1]
    llm, sampling_params, tokenizer = get_sglang_model(model_path)
    demo_generate(llm, sampling_params)
    demo_chat(llm, sampling_params, tokenizer)

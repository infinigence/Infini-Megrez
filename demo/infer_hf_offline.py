import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def get_vllm_model(model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to("cuda")
    generation_config = GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.0,
        ignore_eos=False,
        max_new_tokens=4096,
    )

    return llm, generation_config, tokenizer


def demo_generate(llm, generation_config, tokenizer):
    prompt = "世界上最高的山峰是"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(llm.device)
    outputs = llm.generate(
        input_ids=input_ids,
        generation_config=generation_config,
    )
    for output in outputs:
        print(tokenizer.decode(output))


def demo_chat(llm, generation_config, tokenizer):
    messages = [
        {"role": "user", "content": "世界上最高的山峰是哪座？"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(llm.device)
    outputs = llm.generate(
        input_ids=input_ids,
        generation_config=generation_config,
    )
    for output in outputs:
        print(tokenizer.decode(output))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer_vllm_offline.py <model_path>")
        exit(1)
    model_path = sys.argv[1]

    llm, generation_config, tokenizer = get_vllm_model(model_path)
    demo_generate(llm, generation_config, tokenizer)
    demo_chat(llm, generation_config, tokenizer)

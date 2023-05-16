import os
from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer

def get_llama_model_tokenizer_config(model_name_or_path = "decapoda-research/llama-7b-hf", device="cuda", lora_or_peft_config=True):
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    if lora_or_peft_config:
        from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict, PeftConfig
        if isinstance(lora_or_peft_config, PeftConfig):
            peft_config = lora_or_peft_config
        else:
            lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
            target_modules = lora_trainable.split(",")
            lora_rank=8
            modules_to_save="embed_tokens,lm_head"
            lora_dropout=0.1
            lora_alpha=10.0
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=lora_rank, lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                modules_to_save=modules_to_save)
        model = get_peft_model(model, peft_config)
    if "cuda" in device:
        model.half().cuda(device)
    elif "cpu" == device:
        model.to(device)
    # decapoda-research/llama-7b-hf版本代码里面没有说哪个是pad token，手动注入一个用的eostoken来padding
    # 下载的facebook原版再转hf是有pad_token_id的
    # https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E6%89%8B%E5%8A%A8%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6%E4%B8%8E%E8%BD%AC%E6%8D%A2
    if model.config.pad_token_id <0:
        model.config.update({"pad_token_id": model.config.eos_token_id})
    if "chinese" not in model_name_or_path:
        tokenizer.pad_token_id = model.config.pad_token_id
    else:
        model.config.update({"eos_token_id": tokenizer.eos_token_id})
        model.config.update({"pad_token_id": tokenizer.pad_token_id})
    return model, tokenizer, config

def generate_llama_format_input_str(query = "你好", with_prompt=True):
     # The prompt template below is taken from llama.cpp
    # and is slightly different from the one used in training.
    # But we find it gives better results
    prompt_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
    )
    input_str = query
    if with_prompt:
        input_str = prompt_input.format_map({'instruction': query})
    return input_str

def format_qa_f_llama(i, query, response="", robo_name="llama"):
    str_qa = ""
    if i ==0 :
        str_qa = ("Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n")
    str_qa += f"### Instruction:\n\n{query}\n\n### Response:\n\n{response}"
    return str_qa

if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = "4,5"
    model, tokenizer, config = get_llama_model_tokenizer_config()

    generation_config = dict(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.3,
        max_new_tokens=400
        )
    instruction = "what is your name? my name is"
    text = generate_llama_format_input_str(query=instruction, with_prompt=False)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    """
    >>> tokenizer = LlamaTokenizer.from_pretrained(model_path)
    The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
    The tokenizer class you load from this checkpoint is 'LLaMATokenizer'. 
    The class this function is called from is 'LlamaTokenizer'.
    """
    # 可能因为tokenizer不是完美匹配，导致第一个token id是0，token输出字符串是<unk>,实际的字符串应该是bos正确的
    # 也看这些特殊字符都不显示所以显示<unk>
    input_ids = inputs.input_ids[:, 0:]
    generation_output = model.generate(
        input_ids=input_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **generation_config
    )
    gen_texts = tokenizer.batch_decode(generation_output[:, inputs.input_ids.shape[1]:])
    print(gen_texts[0])
    print("over")

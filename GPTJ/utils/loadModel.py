from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from peft import LoraConfig, TaskType, get_peft_model
from trl import PPOTrainer,PPOConfig, AutoModelForCausalLMWithValueHead,AutoModelForSeq2SeqLMWithValueHead, create_reference_model, set_seed
import torch
def loadLoraModel(model_path, tokenizer_path):
    new_special_tokens = ['[LCM]', '[gcd]', '[mul]', '[mod]', '[add]', '[div]', '[sub]', '[lcm]', '[find]', '[mult]', '[floor]', '[round]', '[power]', '[modulo]', '[return]', '[divide]', '[reverse]', '[convert]', '[subtact]', '[compare]', '[multiply]', '[find_lcm]', '[round_up]', '[raise_to]', '[subtract]', '[find_gcd]', '[round_down]', '[/LCM]', '[/gcd]', '[/mul]', '[/mod]', '[/add]', '[/div]', '[/sub]', '[/lcm]', '[/find]', '[/mult]', '[/floor]', '[/round]', '[/power]', '[/modulo]', '[/return]', '[/divide]', '[/reverse]', '[/convert]', '[/subtact]', '[/compare]', '[/multiply]', '[/find_lcm]', '[/round_up]', '[/raise_to]', '[/subtract]', '[/find_gcd]', '[/round_down]','<endoftext>', '[CALL]']
    nlp_special_tokens = ['LCM', 'gcd', 'mul', 'mod', 'add', 'div', 'sub', 'lcm', 'find', 'mult', 'floor', 'round', 'power', 'modulo', 'return', 'divide', 'reverse', 'convert', 'subtact', 'compare', 'multiply', 'find_lcm', 'round_up', 'raise_to', 'subtract', 'find_gcd', 'round_down','LCM', 'gcd', 'mul', 'mod', 'add', 'div', 'sub', 'lcm', 'find', 'mult', 'floor', 'round', 'power', 'modulo', 'return', 'divide', 'reverse', 'convert', 'subtact', 'compare', 'multiply', 'find_lcm', 'round_up', 'raise_to', 'subtract', 'find_gcd', 'round_down', 'endoftext', 'CALL']

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        )
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token=tokenizer.eos_token

    special_tokens_dict = {'additional_special_tokens': new_special_tokens}
    num_added_toks = tokenizer.add_tokens(new_special_tokens)
    print('We have added', num_added_toks, 'tokens')

    # from customGPTJ import GPTJForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, return_dict=True, low_cpu_mem_usage=True)
    model = get_peft_model(model, peft_config)
    # print(model)
    model.resize_token_embeddings(len(tokenizer))

    for i, special_token in enumerate(new_special_tokens):
        nlp_index = tokenizer.convert_tokens_to_ids([nlp_special_tokens[i]])[0]
        special_token_index = tokenizer.convert_tokens_to_ids([special_token])[0]
        model.transformer.wte.weight.data[special_token_index] = model.transformer.wte.weight.data[nlp_index]
    return model, tokenizer

def loadLoraCheckPoint():
    tokenizer = AutoTokenizer.from_pretrained('path to saved tokenizer after fintuning the model')
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        )
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6b', return_dict=True, low_cpu_mem_usage=True)
    model = get_peft_model(model, peft_config)
    # print(model)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load('path to saved state dict after fintuning the model'))

    return model, tokenizer



def load_PPO_model():
    peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained('EleutherAI/gpt-j-6b',low_cpu_mem_usage=True, peft_config=peft_config)

    tokenizer = AutoTokenizer.from_pretrained('path to saved tokenizer after fintuning the model')
    model.pretrained_model.resize_token_embeddings(len(tokenizer))
    model.pretrained_model.load_state_dict(torch.load('path to saved state dict after fintuning the model'))
    print(model)
    model_ref = create_reference_model(model)
    model.train()

    return model,model_ref,tokenizer




    
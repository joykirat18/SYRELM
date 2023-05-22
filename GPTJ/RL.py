# %%
from trl import PPOTrainer,PPOConfig, AutoModelForCausalLMWithValueHead,AutoModelForSeq2SeqLMWithValueHead, create_reference_model, set_seed
# from ppo_trainer_diffValue import PPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
import torch
import pandas as pd 
import os
from peft import LoraConfig, TaskType, get_peft_model
from utils.pseudoCodeDataloader import getDataset
import argparse
from utils.reward import getReward


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=88888)
parser.add_argument("--model_name", default="gpt2", type=str)
parser.add_argument("--toknizer_name", default="gpt2", type=str)
parser.add_argument("--stateDict_path", default="gpt2", type=str)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--mini_batch_size", default=1, type=int)
parser.add_argument("--num_train_epochs", default=10, type=int)
parser.add_argument("--warmup", default=0.1, type=float)
parser.add_argument("--learning_rate", default=1.41e-6, type=float)
parser.add_argument("--init_kl_coef", default=0.03, type=float)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--input_text_path", default='pseudoCode-Dataset/pseudoCode_csv_full/', type=str)

parser.add_argument("--save", type=str)

args, _ = parser.parse_known_args()

# %%
def load_augmented_model ():

    peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name,low_cpu_mem_usage=True, peft_config=peft_config)
    # model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(args.toknizer_name)
    model.pretrained_model.resize_token_embeddings(len(tokenizer))
    model.pretrained_model.load_state_dict(torch.load(args.stateDict_path))
    print(model)
    model_ref = create_reference_model(model)

    model.to('cuda')
    model.train()

    return model,model_ref,tokenizer


def build_data_pipeline(tokenizer):
    input_text_path = args.input_text_path
    max_seq_length = 256
    from random import shuffle
    from utils.pseudoCodeDataloader import getDataset
    train_data = getDataset(input_text_path + 'train.csv', tokenizer)
    valid_data = getDataset(input_text_path + 'test.csv', tokenizer)

    return train_data, valid_data

# %%
print("Loading model...")
model, model_ref,tokenizer = load_augmented_model()

# %%
def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# %%
import random
seed = random.randint(1, 1000)
print("SEED: ", seed)
set_seed(seed)

# %%
print("Building data pipeline...")
train_data, test_data = build_data_pipeline(tokenizer)


# %%
config = PPOConfig(
    model_name="pseudoCode-gptj6b-rl",
    learning_rate=args.learning_rate,
    init_kl_coef=args.init_kl_coef,
    gamma=args.gamma,
    batch_size=args.batch_size,
    mini_batch_size=args.mini_batch_size,
    optimize_cuda_cache=True,
    seed=args.seed)


# %%
print("Loading ppo model...")
ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer, dataset=train_data)
# lr_scheduler=lr_scheduler
del model
del model_ref
torch.cuda.empty_cache()    

# %%
from transformers import pipeline, set_seed, StoppingCriteriaList, StoppingCriteria
import torch

from utils.generate import generate_pseudoCode

# %%
def getGoldAnswer(target):
    target = target.strip()
    target = target.split('#')
    target = target[-1]
    import re
    # extract number from string
    # print("Target ", target)
    pattern = r'(?<![\w])\d+\.?\d*(?![\w\$])'
    numerical_values = re.findall(pattern, target)

    # target = [re.findall(r'\d+', i) for i in target]
    if(len(numerical_values) == 0):
        return -1
    return float(numerical_values[-1].strip())


import logging
import re
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

import logging
set_global_logging_level(logging.ERROR)


def saveModelBest():
    print("Saving best model")
    os.makedirs(args.save + '/best', exist_ok=True)
    ppo_trainer.model.save_pretrained(args.save + '/best')

def saveModel():
    print("Saving model")
    os.makedirs(args.save, exist_ok=True)
    ppo_trainer.model.save_pretrained(args.save)

num_epochs = 4
from tqdm import tqdm
from utils.generate import getAccuracy
best_acc = getAccuracy(ppo_trainer.model, ppo_trainer.tokenizer,'data/eval_dataset/svamp.json')
best_acc = 0
print("Training PPO model")
baseReward = 0
max_reward = 0
for epoch in tqdm(range(num_epochs)):
    print("Epoch: ", epoch)
    counter = 0
    for _, data_batch in tqdm(enumerate(ppo_trainer.dataloader)):
        try:
            texts = data_batch['text']
            input_ids = data_batch['input_ids']
            attention_masks = data_batch['attention_mask']
            # print(input_ids)
            # print(sentence)
            response_tensors = []
            response_texts = []
            question_texts = []
            gold_answers = []
            query_tensors = []
            query_texts = []
            gold_response_texts = []
            rewards = []
            pointer = 0
            for query in tqdm(texts):

                sentence = query
                prompt = sentence[: sentence.find('<sep>')].strip()
                target = sentence[sentence.find('<sep>')+5:]
                target = target[: target.find('<endoftext>')]

                XtoYInput = prompt
                query_tensor, response_tensor, response_text = generate_pseudoCode(ppo_trainer, tokenizer, XtoYInput, target)
                query_tensors.append(query_tensor.squeeze())
                response_tensors.append(response_tensor.squeeze())
                reward = getReward(response_text, prompt, target, getGoldAnswer(target))
                rewards.append(reward)

                print("reward: ",reward)
                print()

                response_texts.append(response_text)
                query_texts.append(target)
                pointer += 1


            print("rewards: ", rewards)
            reward_sum = 0
            for r in rewards:
                reward_sum += r.item()
            if(max_reward < reward_sum):
                saveModelBest()
                max_reward = reward_sum
            if(counter == 0):
                baseReward = reward_sum
            print("reward sum: ", reward_sum)
            print("Base reward: ", baseReward)
            if(reward_sum > baseReward):
                saveModel()
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, {'query' : [],'response' :[]}, rewards)
            counter += 1
 
        except Exception as e:
            print(e)
            print(sentence)
            print("Error in training")
            print("\n")
            continue
    from utils.generate import getAccuracy
    acc = getAccuracy(ppo_trainer.model, ppo_trainer.tokenizer, 'data/eval_dataset/svamp.json')
    print("accuracySVAMP: ", acc)
    if(best_acc < acc):
        best_acc = acc
        saveModel()




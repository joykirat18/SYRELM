import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import torch
import logging
from tqdm import tqdm
import math
import argparse
import os
from utils.pseudoCodeDataloader import getDataloader
from utils.loadModel import loadLoraModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=88888)
parser.add_argument("--model_name", default="gpt2", type=str)
parser.add_argument("--toknizer_name", default="gpt2", type=str)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--num_train_epochs", default=10, type=int)
parser.add_argument("--warmup", default=0.1, type=float)
parser.add_argument("--learning_rate", default=5e-6, type=float)
parser.add_argument("--input_text_path", default='pseudoCode-Dataset/pseudoCode_csv_full/', type=str)
parser.add_argument("--save", type=str)

args, _ = parser.parse_known_args()

folder = args.input_text_path.split('/')[1]
model_folder = args.model_name.split('/')[-1]
folder = args.save

print("Saving to folder name: ")
print(folder)
train_path = args.input_text_path + 'train.csv'
val_path = args.input_text_path + 'test.csv'


model, tokenizer = loadLoraModel(args.model_name, args.toknizer_name)

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

def create_labels(inputs):
    labels=[]
    for ids,attention_mask in zip(inputs['input_ids'],inputs['attention_mask']):
        label=ids.copy()
        real_len=sum(attention_mask)
        padding_len=len(attention_mask)-sum(attention_mask)
        label[:]=label[:real_len]+[-100]*padding_len
        labels.append(label)
    inputs['labels']=labels

train_dataloader = getDataloader(train_path, tokenizer, args)
valid_dataloader = getDataloader(val_path, tokenizer, args)


num_train_epochs = args.num_train_epochs
training_steps_per_epoch=len(train_dataloader)
total_num_training_steps = int(training_steps_per_epoch*num_train_epochs)
weight_decay=0
learning_rate=args.learning_rate
adam_epsilon=1e-6
warmup_steps=int(total_num_training_steps*args.warmup)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_num_training_steps
)

print("***** Running training *****")
print("  Total_num_training_step = {}".format(total_num_training_steps))
print("  Num Epochs = {}".format(num_train_epochs))
print(f"  Batch_size per device = {args.batch_size}")

def save_model():
    print("saving model")
    model.save_pretrained(folder + '/model')
    tokenizer.save_pretrained(folder + '/tokenizer')
    torch.save(model.state_dict(), folder + '/stateDict.pth')

import os
os.makedirs(folder, exist_ok=True)
save_model()
lowest_loss=10000
model.to('cuda')
for epoch in range(num_train_epochs):
    print(f"Start epoch{epoch+1} of {num_train_epochs}")
    train_loss=0
    epoch_iterator = tqdm(train_dataloader,desc='Iteration')
    model.train()
    model.zero_grad()    
    for _, inputs in enumerate(epoch_iterator):        
        d1,d2,d3=inputs
        d1=d1.to('cuda')
        d2=d2.to('cuda')
        d3=d3.to('cuda')

        optimizer.zero_grad()
        output_XtoY = model(input_ids=d1, attention_mask=d2,labels=d3)
        batch_loss_XtoY=output_XtoY[0]
        batch_loss=batch_loss_XtoY
        batch_loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        train_loss+=batch_loss.item()
        epoch_iterator.set_description('(batch loss=%g)' % batch_loss.item())
        del batch_loss
    print(f'Average train loss per example={train_loss/training_steps_per_epoch} in epoch{epoch+1}')    
    print(f'Starting evaluate after epoch {epoch+1}')
    eval_loss=[]    
    model.eval()    
    for inputs in tqdm(valid_dataloader, desc="eval"):
        d1,d2,d3=inputs
        d1=d1.to('cuda')        
        d2=d2.to('cuda')
        d3=d3.to('cuda')

        with torch.no_grad():
            output_XtoY = model(input_ids=d1, attention_mask=d2,labels=d3)
            batch_loss_XtoY=output_XtoY[0]

            batch_loss=batch_loss_XtoY
        eval_loss+=[batch_loss.cpu().item()]
        del batch_loss
    eval_loss=np.mean(eval_loss)
    perplexity=math.exp(eval_loss)
    print(f'Average valid loss per example={eval_loss} in epoch{epoch+1}')    
    print(f'Perplextiy for valid dataset in epoch{epoch+1} is {perplexity}')
    if(eval_loss < lowest_loss):
        lowest_loss = eval_loss
        save_model()
        from utils.generate import getAccuracy
        acc = getAccuracy(model, tokenizer, 'data/eval_dataset/svamp.json')







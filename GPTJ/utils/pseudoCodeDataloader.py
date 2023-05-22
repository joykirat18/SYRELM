import pandas as pd
import torch
def combinetext(path):

    combine = []
    df = pd.read_csv(path)
    for index, row in df.iterrows():
        prompt = row['prompt']
        pseudoCode = row['summary']
        text = prompt + ' Translate the following into pseudoCode <sep> ' + pseudoCode + ' <endoftext>'
        combine.append(text)

    return combine

def create_labels(inputs):
    labels=[]
    for ids,attention_mask in zip(inputs['input_ids'],inputs['attention_mask']):
        label=ids.copy()
        real_len=sum(attention_mask)
        padding_len=len(attention_mask)-sum(attention_mask)
        label[:]=label[:real_len]+[-100]*padding_len
        labels.append(label)
    inputs['labels']=labels
class PseudoCodeDataset:
    def __init__(self, input):
        self.ids = input['input_ids']
        self.attention_mask = input['attention_mask']
        self.labels=input['labels']
        

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        return [
            torch.tensor(self.ids[item], dtype=torch.long),
            torch.tensor(self.attention_mask[item], dtype=torch.long),
            torch.tensor(self.labels[item], dtype=torch.long),
        ]
from torch.utils.data import Dataset
class TextRLDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=256,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded_text['input_ids'].squeeze()
        attention_mask = encoded_text['attention_mask'].squeeze()
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

def create_labels(inputs):
    labels=[]
    for ids,attention_mask in zip(inputs['input_ids'],inputs['attention_mask']):
        label=ids.copy()
        real_len=sum(attention_mask)
        padding_len=len(attention_mask)-sum(attention_mask)
        label[:]=label[:real_len]+[-100]*padding_len
        labels.append(label)
    inputs['labels']=labels

def getDataloader(path, tokenizer, args):
    text = combinetext(path)
    inputs=tokenizer(text, add_special_tokens=True, padding='max_length',truncation=True,max_length=args.max_seq_length)
    create_labels(inputs)

    dataset=PseudoCodeDataset(inputs)
    data_loader = torch.utils.data.DataLoader(
    dataset,
    shuffle=False,
    batch_size=args.batch_size)

    return data_loader

def getDataset(path, tokenizer):
    text = combinetext(path)
    dataset=TextRLDataset(text, tokenizer)
    return dataset





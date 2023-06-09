from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

data = load_dataset('silver/personal_dialog')
train = data['train']
val = data['validation']
test = data['test']
def process_data(dataset, tokenizer):
    dialogue = None
    profiles = None
    uids = None
    out = []
    j = 0
    for entry in tqdm(dataset):
        if j > 100:
            break
        dialogue = entry['dialog']
        profiles = entry['profile']
        uids = entry['uid']
        tokenized = [tokenizer(sentence, return_tensors="pt") for sentence in dialogue]
        for i in range(len(tokenized) - 1):
            out.append((tokenized[i], tokenized[i + 1]['input_ids']))
        if j < 10:
            print(dialogue[0], tokenized[0])
        j += 1
    return out

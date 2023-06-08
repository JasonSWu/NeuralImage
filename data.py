from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import torch

data = load_dataset('silver/personal_dialog')
train = data['train']
val = data['validation']
test = data['test']
def process_data(dataset, tokenizer):
    dialogue = None
    profile0 = None
    profile1 = None
    uids = None
    out = []
    for entry in dataset:
        dialogue = entry['dialog']
        profile0, profile1 = entry['profile']
        uids = entry['uid']
        tokenized = [tokenizer(sentence, return_tensors="pt") for sentence in dialogue]
        for i in range(len(tokenized) - 1):
            out.append((tokenized[i], tokenized[i + 1]['input_ids']))
    return out

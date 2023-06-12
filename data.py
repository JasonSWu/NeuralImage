from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

def process_data(dataset, tokenizer, n):
    dialogue = None
    profiles = None
    uids = None
    out = []
    j = 0
    for entry in tqdm(dataset):
        if j > n:
            break
        dialogue = entry['dialog']
        profiles = entry['profile']
        uids = entry['uid']
        tokenized = [tokenizer(sentence.replace(" ", ""), return_tensors="pt") for sentence in dialogue]
        for i in range(len(tokenized) - 1):
            out.append((tokenized[i], tokenized[i + 1]['input_ids']))
        j += 1
    return out

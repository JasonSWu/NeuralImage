from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

def process_data(dataset, tokenizer, n, max_len):
    print(dataset['features'])
    dialogue = None
    out = []
    count = 0
    for entry in tqdm(dataset[:n]):
        if count > n:
            break
        if len(entry['uid']) < 3:
            continue
        dialogue = entry['dialog']
        #profiles = entry['profile']
        #uids = entry['uid']
        tokenized = [tokenizer(sentence.replace(" ", ""), return_tensors="pt", padding = 'max_length', max_length = max_len) for sentence in dialogue]
        out.append(tokenized)
    return out
    

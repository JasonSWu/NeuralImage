from datasets import load_dataset, Dataset
from collections import defaultdict
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

def process_data(dataset, tokenizer, n, max_len, bsz):
    dialogue = None
    tmp = dict()
    lens = set()
    out = []
    count = 0
    for entry in tqdm(dataset):
        if count > n:
            break
        len_ = len(entry['uid'])
        if len_ < 3:
            continue
        dialogue = entry['dialog']
        #profiles = entry['profile']
        #uids = entry['uid']
        if len_ not in lens:
            lens.add(len_)
            tmp[len_] = list()
        tmp[len_].append([sent.replace(" ","") for sent in dialogue])
        count += 1
    count = 0
    for len_ in lens:
        dialogues = tmp[len_]
        n_dialogues = len(dialogues)
        while count + bsz <= n_dialogues:
            out.append([tokenizer.batch_encode_plus(batch, padding="max_length", truncation=True, max_length=max_len)
                         for batch in zip(*dialogues[count:count+bsz])])
            count += bsz
    return out
    

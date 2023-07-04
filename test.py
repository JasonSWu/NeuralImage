import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from model import ChatBot, MyDecoder
from model2 import FineTuneTransformer, ManualDecoder
from data import process_data
import os, sys

def pooling_fn(a):
  return torch.mean(a, dim=-2)

device = torch.device("cuda")

config = AutoConfig.from_pretrained("Alethea/GPT2-chitchat")
tokenizer = AutoTokenizer.from_pretrained("Alethea/GPT2-chitchat")
pretrained_model = AutoModelForCausalLM.from_pretrained("Alethea/GPT2-chitchat")
pretrained_model.eval()

hidden_size = config.hidden_size
vocab_size = config.vocab_size

decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
decoder = ManualDecoder(decoder_layer, 3, True, hidden_size, vocab_size, pooling_fn)
optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.1)
a = input()
while a != "q":
    optimizer.load_state_dict(torch.load("./optimizer67"))
    print(optimizer.param_groups[0]['lr'])
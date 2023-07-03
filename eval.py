import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import BertModel, BertTokenizer
from model import ChatBot, MyDecoder
from model2 import ManualDecoder, FineTuneTransformer
from data import process_data
import sys

def pooling_fn(a):
  return torch.mean(a, dim=-2)

def main(decoder_name):
    device = torch.device("cuda")
    config = AutoConfig.from_pretrained("Alethea/GPT2-chitchat")
    tokenizer = AutoTokenizer.from_pretrained("Alethea/GPT2-chitchat")
    pretrained_model = AutoModelForCausalLM.from_pretrained("Alethea/GPT2-chitchat").base_model.to(device)
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    bos = 101
    eos = 102
    max_len = 271 #541 with spaces
    config.pad_token_id = 0

    decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
    decoder = ManualDecoder(decoder_layer, 3, True, hidden_size, vocab_size, pooling_fn)
    decoder.load_state_dict(torch.load(decoder_name))
    decoder = decoder.to(device)
    chatbot = FineTuneTransformer(pretrained_model, decoder, bos, eos, device)
    
    memories = [torch.zeros((1, 1, max_len, hidden_size), device=device)] #want (batch_size, n_mems, seq_len, dim_emb)
    memory_masks = [torch.ones((1, 1, max_len), device=device)] # (n_mems, batch_size, seq_len) or (n_mems, seq_len) to apply to entre batch
    keys = [torch.zeros((1, 1, hidden_size), device=device)]
    input_ = input()
    while input_ != "q":
        tokenized = tokenizer(input_, padding='max_length', max_length=max_len, return_tensors="pt")
        output, encoding = chatbot.forward(tokenized['input_ids'], torch.concat(memories, dim=1), torch.concat(keys, dim=1), 
                                 torch.concat(memory_masks, dim=0), tokenized['attention_mask'], tokenized['token_type_ids'])
        print(output)
        print(tokenizer.decode(output[0]))
        memories.append(torch.unsqueeze(encoding.to(device), dim=0))
        memory_masks.append(torch.unsqueeze(tokenized['attention_mask'].to(device), dim=0))
        keys.append(torch.unsqueeze(pooling_fn(encoding).to(device), dim=0))
        input_ = input()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Input the name of a decoder file")
        exit(0)
    main(sys.argv[1])
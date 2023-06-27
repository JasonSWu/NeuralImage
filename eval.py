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

    decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
    norm_layer = nn.LayerNorm(hidden_size)
    decoder = ManualDecoder(decoder_layer, 3, True, hidden_size, vocab_size, pooling_fn)
    decoder.load_state_dict(torch.load(decoder_name))
    decoder = decoder.to(device)
    chatbot = FineTuneTransformer(pretrained_model, decoder, hidden_size, bos, eos)
    input_ = input()
    while input_ != "q":
        output = chatbot.forward(**tokenizer(input_, return_tensors="pt"))
        print(output)
        print(tokenizer.decode(output[0]))
        input_ = input()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Input the name of a decoder file")
        exit(0)
    main(sys.argv[1])
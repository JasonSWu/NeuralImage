import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, XLMRobertaModel, AutoModelForCausalLM
from model import MyDecoder

device = torch.device("cuda")

config = AutoConfig.from_pretrained("Alethea/GPT2-chitchat")
tokenizer = AutoTokenizer.from_pretrained("Alethea/GPT2-chitchat")
pretrained_model = AutoModelForCausalLM.from_pretrained("Alethea/GPT2-chitchat")
pretrained_model.eval()
for param in pretrained_model.parameters():
    param.requires_grad = False
hidden_size = config.hidden_size
vocab_size = config.vocab_size

decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True, dropout=0.02)
norm_layer = nn.LayerNorm(hidden_size)
decoder = MyDecoder(nn.TransformerDecoder(decoder_layer, num_layers = 4, norm = norm_layer), hidden_size, vocab_size)

def count_parameters(model):
    return sum([p.numel() / 1000000 for p in model.parameters() if p.requires_grad])
print(count_parameters(decoder))
print(count_parameters(pretrained_model))
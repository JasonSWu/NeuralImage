import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, XLMRobertaModel
from transformers import BertModel, BertTokenizer
from model import ChatBot, MyDecoder
from data import process_data
import sys

def main(decoder_name):
    device = torch.device("cuda")
    config = AutoConfig.from_pretrained("bert-base-chinese")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    pretrained_model = BertModel.from_pretrained("bert-base-chinese")
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size

    decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
    norm_layer = nn.LayerNorm(hidden_size)
    decoder = MyDecoder(nn.TransformerDecoder(decoder_layer, num_layers = 4, norm = norm_layer), hidden_size, vocab_size)
    decoder.load_state_dict(torch.load(decoder_name))
    chatbot = ChatBot(pretrained_model, decoder, tokenizer, config.bos_token_id, config.eos_token_id, device)
    input_ = input()
    while input_ != "q":
        print(tokenizer.decode(chatbot.forward(**tokenizer(input_, return_tensors="pt", device=device))[0]))
        input_ = input()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Input the name of a decoder file")
        exit(0)
    main(sys.argv[1])
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
    pretrained_model = BertModel.from_pretrained("bert-base-chinese").to(device)
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    bos = 101
    eos = 102

    decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
    norm_layer = nn.LayerNorm(hidden_size)
    decoder = MyDecoder(nn.TransformerDecoder(decoder_layer, num_layers = 4, norm = norm_layer), hidden_size, vocab_size)
    decoder.load_state_dict(torch.load(decoder_name))
    decoder = decoder.to(device)
    chatbot = ChatBot(pretrained_model, decoder, tokenizer, bos, eos, device)
    chatbot.eval()
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
    print(sys.argv[1])
    main(sys.argv[1])
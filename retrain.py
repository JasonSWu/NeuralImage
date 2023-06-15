import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, XLMRobertaModel
from transformers import BertModel, BertTokenizer
from model import ChatBot, MyDecoder
from data import process_data
import os, sys

def upper_tri_mask(n):
  return torch.triu(torch.ones((n,n)), diagonal=1)

def train(base_llm, decoder, train_dataloader, num_epochs, PAD_IDX, device="cuda"):
  base_llm = base_llm.to(device)
  decoder = decoder.to(device)
  optimizer = torch.optim.SGD(decoder.parameters(), lr=0.01)
  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX) #Ignore padding, dont let it contribute to training
  embed_fn = base_llm.get_input_embeddings()

  for epoch in range(1, num_epochs+1):
    decoder.train()
    total_loss = 0

    for src, tgt in tqdm(train_dataloader):
        #src and tgt should have token IDs, not actual words
        optimizer.zero_grad()
        src, tgt = src.to(device), tgt.to(device)
        encoded_input = base_llm(**src)
        memory = encoded_input.last_hidden_state #Not the memory that we are looking to implement
        #Working with tgt.size() = (batch, seq, embed_size)
        truth = tgt[:, 1:]
        tgt = tgt[:, :-1]
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[1]).bool().to(device)

        embedded_tgt = embed_fn(tgt)
        probabilities = decoder(embedded_tgt, memory, tgt_mask = mask)

        loss = loss_fn(torch.transpose(probabilities, 1, 2), truth) #need (batches, classes, seq). Before transpose, is (bathces, seq, classes)
        loss.backward()
        #torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)

        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_dataloader)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}"))
  return decoder

def main(trained, to_train):
  device = torch.device("cuda")

  config = AutoConfig.from_pretrained("bert-base-chinese")
  tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
  pretrained_model = BertModel.from_pretrained("bert-base-chinese")
  pretrained_model.eval()
  pretrained_model.requires_grad_(False)
  hidden_size = config.hidden_size
  vocab_size = config.vocab_size

  data = load_dataset('silver/personal_dialog')
  train_data = process_data(data['train'], tokenizer, 10000)
  decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
  norm_layer = nn.LayerNorm(hidden_size)
  decoder = MyDecoder(nn.TransformerDecoder(decoder_layer, num_layers = 4, norm = norm_layer), hidden_size, vocab_size)
  decoder.load_state_dict(torch.load(f"./decoder{trained}"))
  decoder = train(pretrained_model, decoder, train_data, to_train, config.pad_token_id, device)
  decoder.eval()
  torch.save(decoder.state_dict(), f"decoder{trained + to_train}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Input the num epochs already trained and the desired number of epochs to train")
        exit(0)
    main(sys.argv[1], int(sys.argv[2]))
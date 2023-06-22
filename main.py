import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from model import ChatBot, MyDecoder
from model2 import FineTuneTransformer, ManualDecoder
from data import process_data
import os

def upper_tri_mask(n):
  return torch.triu(torch.ones((n,n)), diagonal=1)

def train(base_llm, decoder, mem_layer, train_dataloader, num_epochs, PAD_IDX, device="cuda"):
  base_llm = base_llm.to(device)
  decoder = decoder.to(device)
  optimizer = torch.optim.SGD(decoder.parameters(), lr=0.01)
  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX) #Ignore padding, dont let it contribute to training
  embed_fn = base_llm.get_input_embeddings()
  memories = []
  memory_masks = []
  keys = []

  for epoch in range(1, num_epochs+1):
    decoder.train()
    total_loss = 0

    for src, tgt in tqdm(train_dataloader):
        #src and tgt should have token IDs, not actual words
        optimizer.zero_grad()
        src, tgt = src.to(device), tgt.to(device)
        encoded_input = base_llm(**src)
        encoding = encoded_input.last_hidden_state #Not the memory that we are looking to implement
        pooled = encoded_input.pooled_output
        #Working with tgt.size() = (batch, seq, embed_size)
        truth = tgt[:, 1:]
        tgt = tgt[:, :-1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[1]).bool().to(device)

        embedded_tgt = embed_fn(tgt)
        memories = mem_layer(pooled, memories, keys)
        probabilities = decoder(embedded_tgt, encoding, memories, memory_masks, tgt_mask)

        loss = loss_fn(torch.transpose(probabilities, 1, 2), truth) #need (batches, classes, seq). Before transpose, is (bathces, seq, classes)
        loss.backward()
        #torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)

        optimizer.step()
        total_loss += loss.item()

        keys.append(pooled)
        memories.append(encoding)

    train_loss = total_loss / len(train_dataloader)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}"))
  return decoder

device = torch.device("cuda")

config = AutoConfig.from_pretrained("Alethea/GPT2-chitchat")
tokenizer = AutoTokenizer.from_pretrained("Alethea/GPT2-chitchat")
pretrained_model = AutoModelForCausalLM.from_pretrained("Alethea/GPT2-chitchat")
pretrained_model.eval()
pretrained_model.requires_grad_(False)
hidden_size = config.hidden_size
vocab_size = config.vocab_size
bos = 101
eos = 102

data = load_dataset('silver/personal_dialog')
train_data = process_data(data['train'], tokenizer, 100)
decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
norm_layer = nn.LayerNorm(hidden_size)
decoder = ManualDecoder(nn.TransformerDecoder(decoder_layer, num_layers = 4, norm = norm_layer), hidden_size, vocab_size)

decoder = train(pretrained_model, decoder, train_data, 10, config.pad_token_id, device)
torch.save(decoder.state_dict(), "decoder10")
decoder = train(pretrained_model, decoder, train_data, 10, config.pad_token_id, device)
torch.save(decoder.state_dict(), "decoder20")
decoder = train(pretrained_model, decoder, train_data, 10, config.pad_token_id, device)
torch.save(decoder.state_dict(), "decoder30")
decoder = train(pretrained_model, decoder, train_data, 10, config.pad_token_id, device)
torch.save(decoder.state_dict(), "decoder40")
chatbot = FineTuneTransformer(pretrained_model, decoder, tokenizer, 101, 102, device)

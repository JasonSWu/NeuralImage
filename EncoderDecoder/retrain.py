import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from model import ChatBot, MyDecoder
from model2 import FineTuneTransformer, ManualDecoder
from data import process_data
import os, sys

def upper_tri_mask(n):
  return torch.triu(torch.ones((n,n)), diagonal=1)

def pooling_fn(a):
  return torch.mean(a, dim=-2)

def validate(base_llm, decoder, loss_fn, val_dataloader, dim_emb, max_len, bsz, device="cuda"):
  base_llm = base_llm.to(device)
  decoder = decoder.to(device)
  embed_fn = base_llm.get_input_embeddings()
  memories = [torch.zeros((bsz, 1, max_len, dim_emb), device=device)] #want (batch_size, n_mems, seq_len, dim_emb)
  memory_masks = [torch.ones((1, bsz, max_len), device=device)] # (n_mems, batch_size, seq_len) or (n_mems, seq_len) to apply to entre batch
  keys = [torch.zeros((bsz, 1, dim_emb), device=device)] # (batch_size, n_mes, dim_emb)
  decoder.eval()
  total_loss = 0
  total_replies = 0
  for convo in tqdm(val_dataloader):
      n_replies = len(convo) - 1
      total_replies += n_replies
      for i in range(n_replies):
        #src and tgt should have token IDs, not actual words
        src, src_padding_mask = convo[i]['input_ids'].to(device), convo[i]['attention_mask'].to(device)
        tgt, tgt_padding_mask = convo[i + 1]['input_ids'].to(device), convo[i + 1]['attention_mask'][:,:-1].to(device)
        encoded_input = base_llm(input_ids = src, attention_mask = src_padding_mask)
        encoding = encoded_input.last_hidden_state #(seq_length, embed_size)
        pooled = pooling_fn(encoding) #(embed_size)
        #Working with tgt = (batch, seq, embed_size)
        truth = tgt[:, 1:]
        tgt = tgt[:, :-1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[1]).to(device)

        embedded_tgt = embed_fn(tgt)
        probabilities = decoder(embedded_tgt, encoding, src_padding_mask.to(torch.float32),
                                torch.concat(memories, dim=1), torch.concat(keys, dim=1), 
                                torch.concat(memory_masks, dim=0), tgt_mask, tgt_padding_mask.to(torch.float32))

        loss = loss_fn(torch.transpose(probabilities, 1, 2), truth) #need (batches, classes, seq). Before transpose, is (batches, seq, classes)
        
        total_loss += loss.item()

        memories.append(torch.transpose(torch.unsqueeze(encoding, dim=0), 0,1))
        keys.append(torch.transpose(torch.unsqueeze(pooled, dim=0),0,1))
        memory_masks.append(torch.unsqueeze(src_padding_mask, dim=0))
      del memories[1:]
      del memory_masks[1:]
      del keys[1:]
  return total_loss / total_replies

def train(base_llm, decoder, optimizer, loss_fn, train_dataloader, num_epochs, dim_emb, max_len, bsz, val=False, val_dataloader=None, device="cuda"):
  base_llm = base_llm.to(device)
  decoder = decoder.to(device)
  optimizer = optimizer
  embed_fn = base_llm.get_input_embeddings()
  memories = [torch.zeros((bsz, 1, max_len, dim_emb), device=device)] #want (batch_size, n_mems, seq_len, dim_emb)
  memory_masks = [torch.ones((1, bsz, max_len), device=device)] # (n_mems, batch_size, seq_len) or (n_mems, seq_len) to apply to entre batch
  keys = [torch.zeros((bsz, 1, dim_emb), device=device)] # (batch_size, n_mes, dim_emb)
  for epoch in range(1, num_epochs+1):
    decoder.train()
    total_loss = 0
    total_replies = 0
    for convo in tqdm(train_dataloader):
        n_replies = len(convo) - 1
        total_replies += n_replies
        for i in range(n_replies):
          #src and tgt should have token IDs, not actual words
          src, src_padding_mask = convo[i]['input_ids'].to(device), convo[i]['attention_mask'].to(device)
          tgt, tgt_padding_mask = convo[i + 1]['input_ids'].to(device), convo[i + 1]['attention_mask'][:,:-1].to(device)
          optimizer.zero_grad()
          encoded_input = base_llm(input_ids = src, attention_mask = src_padding_mask)
          encoding = encoded_input.last_hidden_state #(seq_length, embed_size)
          pooled = pooling_fn(encoding) #(embed_size)
          #Working with tgt = (batch, seq, embed_size)
          truth = tgt[:, 1:]
          tgt = tgt[:, :-1]
          tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[1]).to(device)

          embedded_tgt = embed_fn(tgt)
          probabilities = decoder(embedded_tgt, encoding, src_padding_mask.to(torch.float32),
                                  torch.concat(memories, dim=1), torch.concat(keys, dim=1), 
                                  torch.concat(memory_masks, dim=0), tgt_mask, tgt_padding_mask.to(torch.float32))

          loss = loss_fn(torch.transpose(probabilities, 1, 2), truth) #need (batches, classes, seq). Before transpose, is (batches, seq, classes)
          loss.backward()
          #torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)

          optimizer.step()
          total_loss += loss.item()

          memories.append(torch.transpose(torch.unsqueeze(encoding, dim=0), 0,1))
          keys.append(torch.transpose(torch.unsqueeze(pooled, dim=0),0,1))
          memory_masks.append(torch.unsqueeze(src_padding_mask, dim=0))
        del memories[1:]
        del memory_masks[1:]
        del keys[1:]

    train_loss = total_loss / total_replies
    if val:
      val_loss = validate(base_llm, decoder, loss_fn, val_dataloader, dim_emb, max_len, bsz, device)
      print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
    else:
      print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}")
  return decoder

def main(trained, to_train, train_size, lr, val):
  device = torch.device("cuda")

  config = AutoConfig.from_pretrained("Alethea/GPT2-chitchat")
  tokenizer = AutoTokenizer.from_pretrained("Alethea/GPT2-chitchat")
  pretrained_model = AutoModelForCausalLM.from_pretrained("Alethea/GPT2-chitchat").base_model
  pretrained_model.eval()
  pretrained_model.requires_grad_(False)
  hidden_size = config.hidden_size
  vocab_size = config.vocab_size
  bos = 101
  eos = 102
  max_len = 271 #541 with spaces
  memory_limit = 166
  config.pad_token_id = 0
  bsz = 16
  val_size = 1000

  data = load_dataset('silver/personal_dialog')
  train_data = process_data(data['train'], tokenizer, train_size, max_len = max_len, bsz = bsz)
  val_data = process_data(data['validation'], tokenizer, val_size, max_len = max_len, bsz = bsz)
  decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
  decoder = ManualDecoder(decoder_layer, 6, True, hidden_size, vocab_size, pooling_fn)
  decoder.load_state_dict(torch.load(f"./decoder{trained}"))
  optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)
  #optimizer.load_state_dict(torch.load(f"./optimizer{trained}"))
  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.pad_token_id) #Ignore padding, dont let it contribute to training
  decoder = train(pretrained_model, decoder, optimizer, loss_fn, train_data, to_train, hidden_size, max_len, bsz, val, val_data, device)
  decoder.eval()
  torch.save(decoder.state_dict(), f"decoder{trained + to_train}")
  torch.save(optimizer.state_dict(), f"optimizer{trained + to_train}")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Input: epochs trained, epochs to train, train size, learning rate, and whether to validate (\"True\" or \"False\")")
        exit(0)
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5] == "True")
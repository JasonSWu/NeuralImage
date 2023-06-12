import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, XLMRobertaModel
from model import ChatBot, MyDecoder
from data import process_data

def upper_tri_mask(n):
    # 0 1 1 1
    # 0 0 1 1
    # 0 0 0 1 
    # 0 0 0 0
    # (batch x sequence)
    return torch.transpose(torch.tril(torch.ones((n,n)), diagonal=-1), 0, 1) #1's represent masking

def train(base_llm, decoder, train_dataloader, num_epochs, PAD_IDX, device="cuda"):
  base_llm = base_llm.to(device)
  decoder = decoder.to(device)
  optimizer = torch.optim.RMSprop(decoder.parameters(), lr=0.1,alpha=0.95)
  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=6) #Ignore padding, dont let it contribute to training
  embed_fn = base_llm.get_input_embeddings()

  for epoch in range(1, num_epochs+1):
    decoder.train()
    total_loss = 0

    for src, tgt in tqdm(train_dataloader):
        #src and tgt should have token IDs, not actual words
        src, tgt = src.to(device), tgt.to(device)
        encoded_input = base_llm(**src)
        memory = encoded_input.last_hidden_state #Not the memory that we are looking to implement
        #Working with tgt.size() = (batch, seq, embed_size)
        print(tgt)
        truth = tgt[:, 1:]
        tgt = tgt[:, :-1]
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[1]).to(device)

        embedded_tgt = embed_fn(tgt)
        probabilities = decoder(embedded_tgt, memory, tgt_mask = mask)
        print(torch.argmax(probabilities, dim=-1))
        optimizer.zero_grad()
        
        loss = loss_fn(torch.transpose(probabilities, 1, 2), truth) #need (batches, classes, seq). Before transpose, is (bathces, seq, classes)
        print(loss.item())
        input()
        loss.backward()
        #torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)

        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_dataloader)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}"))
  return decoder

device = torch.device("cuda")

config = AutoConfig.from_pretrained("xlm-roberta-base")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
pretrained_model = XLMRobertaModel.from_pretrained("xlm-roberta-base", is_decoder=False)
pretrained_model.eval()
pretrained_model.requires_grad_(False)
hidden_size = config.hidden_size
vocab_size = config.vocab_size

data = load_dataset('silver/personal_dialog')
train_data = process_data(data['train'], tokenizer)
decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
norm_layer = nn.LayerNorm(hidden_size)
decoder = MyDecoder(nn.TransformerDecoder(decoder_layer, num_layers = 4, norm = norm_layer), hidden_size, vocab_size)
decoder = train(pretrained_model, decoder, train_data, 1, config.pad_token_id, device)
chatbot = ChatBot(pretrained_model, decoder, tokenizer, config.bos_token_id, config.eos_token_id, device)
example1 = "谢谢你付我的饭钱!"
example2 = "你好"
print(tokenizer.decode(chatbot.forward(**tokenizer(example1, return_tensors="pt"))[0]))
print(tokenizer.decode(chatbot.forward(**tokenizer(example2, return_tensors="pt"))[0]))
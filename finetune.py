import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import BloomForCausalLM, BloomTokenizerFast, BloomConfig
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model import ChatBot, MyDecoder
from model2 import FineTuneTransformer, ManualDecoder
from data import process_data
from bs4 import BeautifulSoup
from typing import Callable, Any, List
import os, sys

def bytes_to_GiB(b):
  return (b / 1073741824)

count = 0
def check_memory():
  global count
  count += 1
  print(f"{count}:", bytes_to_GiB(torch.cuda.memory_allocated()), bytes_to_GiB(torch.cuda.memory_reserved()))

def check_size_in_MB(tensor):
  return (tensor.nelement() * tensor.element_size() / 1048576)

def upper_tri_mask(n):
  return torch.triu(torch.ones((n,n)), diagonal=1)

def pooling_fn(a):
  return torch.mean(a, dim=-2)

def finetune(base_llm, optimizer, loss_fn, train_dataloader, num_epochs, bsz, teacher_force=True, device=torch.device("cuda")):
  optimizer = optimizer
  for epoch in range(1, num_epochs+1):
    total_loss = 0
    torch.cuda.empty_cache()
    for entry in tqdm(train_dataloader):
      #torch.cuda.empty_cache()
      optimizer.zero_grad()
      src, tgt = entry
      tgt = tgt['input_ids']
      #print(check_size_in_MB(src['input_ids']))
      #check_memory()
      #tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
      #print(tokenizer.decode(src['input_ids'][0]), tokenizer.decode(tgt[0]))
      len_tgt = tgt.size()[1]
      probabilities = base_llm(**src).logits
      probabilities = probabilities[:,-len_tgt:]
      loss = loss_fn(torch.transpose(probabilities, 1, 2), tgt) #need (batches, classes, seq). Before transpose, is (batches, seq, classes)
      #check_memory()
      #print(torch.cuda.memory_summary(device=None, abbreviated=False))
      loss.backward()
      #torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
      #check_memory()
      optimizer.step()
      total_loss += loss.item()
      #check_memory()

    train_loss = total_loss / len(train_dataloader)
    print(f"epoch {epoch + 1}: {train_loss}")
  return base_llm

def retrieve_data(process_fn: List[Callable[[str, str], Any]]):
  data = [list() for fn in process_fn]
  indices = list(range(len(data)))
  with open("scraping/poems.html", "rb") as f:
    soup = BeautifulSoup(f, "lxml")
  for entry in soup.find_all("li"):
    title, poem = entry.find_all("p")
    title = title.getText(); poem = poem.getText()
    for i in indices:
      data[i].append(process_fn[i](title, poem))
  return data

def freezer_glm(model, n_dont_freeze):
  #version for ChatGLM-6b
  thawed_layers = []
  chatGLMModel = None
  for name, layer in model.named_children():
    chatGLMModel = layer
  encoder = None
  for name, layer in chatGLMModel.named_children():
    if name == "output_layer":
      layer.requires_grad = True
      thawed_layers += list(layer.parameters())
    elif name == "encoder":
      encoder = layer
    else:
      layer.requires_grad = False
  layers = None
  lowest_i = 27 - n_dont_freeze
  for name, layer in encoder.named_children():
    if name == "final_layernorm":
      layer.requires_grad = True
      thawed_layers += list(layer.parameters())
    elif name == "layers":
      layers = layer
  for name, layer in layers.named_children():
    if int(name) > lowest_i:
      layer.requires_grad = True
      thawed_layers += list(layer.parameters())
    else:
      layer.requires_grad = False
  return thawed_layers

def freezer_bloom(model, n_dont_freeze):
  thawed_layers = []
  bloom_model = None
  for name, layer in model.named_children():
    if name == "transformer":
      bloom_model = layer
    elif name == "lm_head":
      layer.requires_grad = True
      thawed_layers += list(layer.parameters())
  module_list = None
  for name, layer in bloom_model.named_children():
    if name == "h":
      module_list = layer
    else:
      layer.requires_grad = False
  lowest_i = 23 - n_dont_freeze
  for name, layer in module_list.named_children():
    if int(name) > lowest_i:
      layer.requires_grad = True
      thawed_layers += list(layer.parameters())
    else:
      layer.requires_grad = False
  return thawed_layers

def main(num_epochs = 10, lr=0.00002):
    device = torch.device("cuda")
    
    model_name = "IDEA-CCNL/Wenzhong-GPT2-110M"

    #config = BloomConfig.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
    #model = model.quantize(8) only for GLM-6b
    
    thawed_params = freezer_bloom(model, 4)
    model.train()

    max_len = 1024

    def concat(src, tgt):
      return {key: torch.concat((src[key], tgt[key]), dim=1) for key in src.keys()}

    def truncate(tensor_dict, len):
      return {key: value[:,:len].to(device) for key, value in tensor_dict.items()}
    
    raw_prompt = "以下诗句是苏轼，又名苏东坡，题为《{}》：\n"
    def raw_process(title, poem):
      tgt = tokenizer([poem], return_tensors="pt")
      src = tokenizer([raw_prompt.format(title)], return_tensors="pt")
      max_tgt_len = max_len - src['input_ids'].size()[1] + 1
      src = truncate(concat(src, tgt), max_len)
      tgt = truncate(tgt, max_tgt_len)
      return src, tgt
    
    chat_prompt = "[Round 1]\n\n问：{}\n\n答：" #The colons are weird Chinese version of colon
    user_query = "模仿苏东坡（又名苏轼）的风格写一首诗，题为《{}》"
    def chat_process(title, poem):
      tgt = tokenizer([poem], return_tensors="pt")
      src = tokenizer([chat_prompt.format(user_query.format(title))], return_tensors="pt")
      max_tgt_len = max_len - src['input_ids'].size()[1] + 1
      src = truncate(concat(src, tgt), max_len)
      tgt = truncate(tgt, max_tgt_len)
      return src, tgt
    
    #pad_id = 50256
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer1 = torch.optim.SGD(thawed_params, lr=lr)
    optimizer2 = torch.optim.SGD(thawed_params, lr=lr)

    bsz = 8
    
    raw_data, chat_data = retrieve_data([raw_process, chat_process])
    
    finetune(model, optimizer1, loss_fn, raw_data, num_epochs, bsz, device)
    finetune(model, optimizer2, loss_fn, chat_data, num_epochs, bsz, device)
    torch.save(model.state_dict(), "finetuned")

if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]))

'''finetuning scheme:
  raw text:
    teacher forcing
    autoregressive
  chat text:
    teacher forcing
    autoregressive
'''
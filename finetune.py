import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import BloomForCausalLM, BloomTokenizerFast, BloomConfig
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from bs4 import BeautifulSoup
from typing import Callable, Any, List
import os, sys

def bytes_to_GiB(b):
  '''Returns the input converted from bytes to gigabytes.'''
  return (b / 1073741824)

def check_memory():
  '''Checks how much memory has been allocated in the GPU (use torch.cuda.empty_cache() to deal with memory leaks).
  For a more comprehensive breakdown of memory allocation, use torch.cuda.memory_summary(device=, abbreviated=)'''
  if not torch.cuda.is_available():
    print("check_memory: Cuda not available")
  print(bytes_to_GiB(torch.cuda.memory_allocated()), bytes_to_GiB(torch.cuda.memory_reserved()))

def check_size_in_MB(tensor):
  '''Checks the size is megabytes of a tensor'''
  return (tensor.nelement() * tensor.element_size() / 1048576)

def finetune(base_llm, optimizer, loss_fn, train_dataloader, num_epochs, bsz, teacher_force=True, device=torch.device("cuda")):
  for epoch in range(1, num_epochs+1):
    total_loss = 0
    torch.cuda.empty_cache() # Avoid memory leaks which could result in overload
    for entry in tqdm(train_dataloader):
      optimizer.zero_grad() # Clear accumulated gradients
      src, tgt = entry # src and tgt are dictionaries typically containing keys {'input_ids', 'attention_mask'} and more.
      tgt = tgt['input_ids']
      len_tgt = tgt.size()[1]
      probabilities = base_llm(**src).logits # base_llm(**src) outputs final 'logits' and 'last_hidden_state'
      probabilities = probabilities[:,-len_tgt:]
      loss = loss_fn(torch.transpose(probabilities, 1, 2), tgt) # Need (batches, vocab_size, seq_len). Prior to transpose, it is (batches, seq_len, vocab_size)
      loss.backward() # Calculate gradients
      #torch.nn.utils.clip_grad_value_(model.parameters(), 5.0) # Clips any gradients above 5.0 in magnitude
      optimizer.step() # Apply accumulated gradients
      total_loss += loss.item() # Turn loss into integer and add to total

    train_loss = total_loss / len(train_dataloader)
    print(f"epoch {epoch}: {train_loss}")
  return base_llm

def retrieve_data(process_fn: List[Callable[[str, str], Any]]):
  '''Takes in a list of processing functions. Each function must
  take in two strings (the title of the poem and the poem itself) and
  return a dictionary of tensors. Applies each function to each title-poem
  pair in the scraping/poems.html file.'''
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
  '''Freezes all layers except the last linear layer and the top
  n_dont_freeze transformer layers of the ChatGLM2-6b model.
  Returns the unfrozen layers'''
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

def freezer_bloom_or_gpt2(model, n_dont_freeze):
  '''Luckily applies to many generic LLM architectures. Freezes
  all layers except the final linear layer and the last n_dont_freeze
  transformer layers of the bigscience/bloom-560m and IDEA-CCNL/Wenzhong-GPT2-110M.
  Returns the unfrozen layers.'''
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
  lowest_i = len(module_list) - n_dont_freeze
  for name, layer in module_list.named_children():
    if int(name) > lowest_i:
      layer.requires_grad = True
      thawed_layers += list(layer.parameters())
    else:
      layer.requires_grad = False
  return thawed_layers

def main(num_epochs = 30, lr=0.00002, model_file = "None", optimizer1_file = "None", optimizer2_file = "None"):
  device = torch.device("cuda")
  
  model_name = "THUDM/chatglm2-6b"
  model_alias = "glm"

  config = AutoConfig.from_pretrained(model_name, trust_remote_code=True) # trust_remote_code param only necessary for the "Auto.." libraries
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda() # .half() reduce any model from FP32 precision to FP16
  model = model.quantize(8) # only for GLM-6b, reduces to FP8

  if model_file != "None":
    model.load_state_dict(torch.load(model_file))
  
  thawed_params = freezer_glm(model, 5)
  model.train()

  max_len = 1024 # Maximum length of tensors after tokenization (includes prompt)

  def concat(src, tgt):
    '''Returns a dictionary with the same keys, but all values concatenated.'''
    return {key: torch.concat((src[key], tgt[key]), dim=1) for key in src.keys()}

  def truncate(tensor_dict, len):
    '''Truncates all the values of the inputted dictionary.
    Tensors in the dictionary are assumed to be batched.'''
    return {key: value[:,:len].to(device) for key, value in tensor_dict.items()}
  
  raw_prompt = "以下诗句是苏轼，又名苏东坡，题为《{}》：\n" # Prompt for informing model
  def raw_process(title, poem):
    tgt = tokenizer([poem], return_tensors="pt")
    src = tokenizer([raw_prompt.format(title)], return_tensors="pt")
    max_tgt_len = max_len - src['input_ids'].size()[1] + 1 # Amount of tgt left after max_len truncation
    src = truncate(concat(src, tgt), max_len)
    tgt = truncate(tgt, max_tgt_len)
    return src, tgt
  
  chat_prompt = "[Round 1]\n\n问：{}\n\n答：" # The colons are the Chinese version of colon. Different encoding from ":"
  user_query = "模仿苏东坡（又名苏轼）的风格写一首诗，题为《{}》" # Prompt for training the model to respond to user directions
  def chat_process(title, poem):
    tgt = tokenizer([poem], return_tensors="pt")
    src = tokenizer([chat_prompt.format(user_query.format(title))], return_tensors="pt")
    max_tgt_len = max_len - src['input_ids'].size()[1] + 1 # Amount of tgt left after max_len truncation
    src = truncate(concat(src, tgt), max_len)
    tgt = truncate(tgt, max_tgt_len)
    return src, tgt
  
  pad_id = config.pad_token_id
  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

  optimizer1 = torch.optim.SGD(thawed_params, lr=lr)
  optimizer2 = torch.optim.SGD(thawed_params, lr=lr)
  if optimizer1_file != "None":
    optimizer1.load_state_dict(torch.load(optimizer1_file))
  if optimizer2_file != "None":
    optimizer2.load_state_dict(torch.load(optimizer2_file))

  bsz = 8 #not batching right now
  
  raw_data, chat_data = retrieve_data([raw_process, chat_process])
  
  epoch_count = 0 if model_file == "None" else int(model_file[-2:])
  num_epochs += epoch_count
  while num_epochs > epoch_count:
    finetune(model, optimizer1, loss_fn, raw_data, 5, bsz, device)
    finetune(model, optimizer2, loss_fn, chat_data, 5, bsz, device)
    epoch_count += 5
    torch.save(model.state_dict(), f"{model_alias}-{epoch_count}")
    torch.save(optimizer1.state_dict(), f"{model_alias}-optim1-{epoch_count}")
    torch.save(optimizer2.state_dict(), f"{model_alias}-optim2-{epoch_count}")

if __name__ == "__main__":
  '''Input number of epochs to be trained, learning rate, model file ("None" if none),
  optimizer1 file ("None" if none), and optimizer2 file ("None" if none). Number of epochs
  to be trained is actually the number of epochs trained for each prompt. In other words, it's
  actually doubled.'''
  main(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
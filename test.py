import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

config = AutoConfig.from_pretrained("xlm-roberta-base")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
pretrained_model = AutoModelForCausalLM.from_pretrained("xlm-roberta-base")
print(config.__dir__())
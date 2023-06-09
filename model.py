import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class ChatBot(nn.Module):
    def __init__(self, pretrained_model, decoder_network, tokenizer, bos_token_id, eos_token_id, device):
        super(ChatBot, self).__init__()
        self.llm = pretrained_model
        self.decoder = decoder_network
        self.tokenizer = tokenizer
        self.bos = bos_token_id
        self.eos = eos_token_id
        self.device = device
        self.embed = pretrained_model.get_input_embeddings()

    def forward(self, input_ids, token_type_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
            encoded_input = outputs.last_hidden_state
            out_seq = [[self.bos]]
            while out_seq[0][-1] != self.eos:
                tgt = self.embed(torch.tensor(out_seq, dtype=torch.long, device=self.device))
                #print(tgt.size())
                mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[1]).bool().to(self.device)
                #print(mask.size())
                response_logits = self.decoder(tgt, memory = encoded_input, tgt_mask = mask)
                #print(response_logits.size())
                token_id = torch.argmax(response_logits[:, -1, :], dim=-1)
                #print("---------------------------------")
                out_seq[0].append(token_id.item())
            return out_seq
        
class MyDecoder(nn.Module):
    def __init__(self, decoder, hidden_size, vocab_size):
        super(MyDecoder, self).__init__()
        self.decoder = decoder
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, tgt, memory, tgt_mask):
        return self.linear(self.decoder(tgt, memory, tgt_mask=tgt_mask))
import torch
import torch.nn as nn

class ChatBot(nn.Module):
    def __init__(self, pretrained_model, decoder_network, tokenizer, bos_token_id, eos_token_id, device):
        super(ChatBot, self).__init__()
        self.llm = pretrained_model
        self.decoder = decoder_network
        self.tokenizer = tokenizer
        self.bos = bos_token_id
        self.eos = eos_token_id
        self.device = device

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
            encoded_input = outputs.last_hidden_state
            out_seq = [[self.bos]]
            while out_seq[-1] != self.eos:
                tgt = torch.Tensor(out_seq).to(self.device)
                mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[0]).to(self.device)
                response_logits = self.decoder(tgt, memory = encoded_input, tgt_mask = mask)
                token_id = torch.argmax(response_logits[:, -1, :], dim=-1)
                out_seq[0].append(token_id.item())
            return response_logits
        
class MyDecoder(nn.Module):
    def __init__(self, decoder, hidden_size, vocab_size):
        super(MyDecoder, self).__init__()
        self.decoder = decoder
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, tgt, memory, tgt_mask):
        return self.softmax(self.linear(self.decoder(tgt, memory, tgt_mask=tgt_mask)))
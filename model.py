import torch
import torch.nn as nn

class ChatBot(nn.Module):
    def __init__(self, pretrained_model, decoder_network, tokenizer, bos_token_id, eos_token_id):
        super(ChatBot, self).__init__()
        self.llm = pretrained_model
        self.decoder = decoder_network
        self.tokenizer = tokenizer
        self.bos = bos_token_id
        self.eos = eos_token_id

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
            encoded_input = outputs.last_hidden_state
            out_seq = [self.bos]
            while out_seq[-1] != self.eos:
                response_logits = self.response_decoder(tgt = torch.Tensor(out_seq, device="cuda"), memory = encoded_input)
                token_id = torch.argmax(response_logits)
            return response_logits
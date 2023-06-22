import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Memory(nn.Module):
    def __init__(self, dim, n_mems, batched=False):
        super(Memory, self).__init__()
        self.n_mems = n_mems
        self.lin = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(n_mems)
        self.batched = batched
    def forward(self, pooled_output, memories, keys):
        #keys: (n, mems, dim)
        #memories: (n, mems, seq, dim)
        #pooled_output: (n, dim)
        to_dot = self.lin(pooled_output)
        to_dot = torch.unsqueeze(to_dot, dim=-2).repeat([1, self.n_mems, 1] if self.batched else [self.n_mems, 1])
        #to_dot (n, mems, dim)
        weights = torch.softmax(torch.sum(to_dot * keys, dim=-1), dim=-1)
        scaled_memories = torch.einsum('nmsd,nm->nmsd', memories, weights)
        return torch.transpose(scaled_memories, 0, 1) #(mems, n, seq, dim)

class ManualDecoder(nn.Module):
    def __init__(self, layer, N):
        super(ManualDecoder, self).__init__()
        self.layer = layer
        self.layers = clones(layer, N - 1)
        self.norm = nn.LayerNorm(layer.dim_emb)

    def forward(self, x, encoded, memory, memory_masks, tgt_mask, tgt_padding_mask):
        output = self.layer(x, encoded, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        for mem, mem_mask in zip(memory, memory_masks):
            output = self.layer.forward(output, mem, mem_mask)
        for layer in self.layers:
            output = layer.forward(output, encoded, tgt_mask, tgt_padding_mask)
        return self.norm(output)

class FineTuneTransformer(nn.Module):
    def __init__(self, LLM, tokenizer, num_decoder_layers, emb_size, nhead, tgt_vocab_size, embed_fn, max_len, batched=False, dim_feedforward = 512, dropout = 0.1):
        super(FineTuneTransformer, self).__init__()
        self.encoder = LLM
        self.decoder = ManualDecoder(nn.TransformerDecoderLayer(emb_size, nhead, dim_feedforward, dropout), num_decoder_layers)
        self.tokenizer = tokenizer
        self.embed = embed_fn
        self.out = nn.Linear(emb_size, tgt_vocab_size)
        self.memory = Memory(emb_size, max_len, batched)
    
    def get_src_mask(self, src):
        # Essentially no masking
        return torch.zeros((src.shape[0], src.shape[0]),device=device).type(torch.bool)

    def get_tgt_mask(self, tgt):
        # For use in training while teacher forcing. Do not want tokens to cheat and look at the future.
        # Generates triangular mask
        size = tgt.shape[0]
        mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def encode(self, src, src_padding_mask, token_type_ids):
        output = self.encoder(input_ids=src, attention_mask=src_padding_mask, token_type_ids=token_type_ids)
        return output.last_hidden_state, output.pooler_output

    def decode(self, tgt, encoded, memory, memory_mask, src_padding_mask, tgt_padding_mask):
        output = self.decoder(
            self.embed(self.tokenizer(tgt)), encoded, memory, memory_mask, 
            src_padding_mask, self.get_tgt_mask(tgt), tgt_padding_mask, self.training)
        return output
    
    def forward(self, src, tgt, kv_store, keys, memory_masks, src_padding_mask, tgt_padding_mask):
        # Note that this implementation is for teacher forcing.
        encoded, pooled = self.encode(src, src_padding_mask)
        memories = self.memory(pooled, kv_store, keys)
        decoded = self.decode(tgt, encoded, memories, memory_masks, src_padding_mask, tgt_padding_mask)
        output = self.out(decoded)
        return output
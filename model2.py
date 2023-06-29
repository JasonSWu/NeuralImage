import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Memory(nn.Module):
    def __init__(self, dim, batched=False):
        super(Memory, self).__init__()
        self.lin = nn.Linear(dim, 3 * dim)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.lin2 = nn.Linear(3 * dim, dim)
        self.softmax = nn.Softmax()
        self.batched = batched
    def forward(self, pooled_output, memories, keys):
        #keys: (n, mems, dim) or (mems, dim)
        #memories: (n, mems, seq, dim) or (mems, seq, dim)
        #pooled_output: (n, dim) or (dim)
        n_mems = memories.size()[-3]
        to_dot = self.lin2(self.relu(self.lin(pooled_output)))
        to_dot = torch.unsqueeze(to_dot, dim=-2).repeat([1, n_mems, 1] if self.batched else [n_mems, 1])
        #to_dot (n, mems, dim) or (mems, dim)
        weights = torch.softmax(torch.sum(to_dot * keys, dim=-1), dim=-1) * n_mems #dot product along last dimension
        scaled_memories = torch.einsum('...msd,...m->...msd', memories, weights) #scale each memory
        return torch.transpose(scaled_memories, 0, 1) if self.batched else scaled_memories #(mems, n, seq, dim)

class ManualDecoder(nn.Module):
    def __init__(self, layer, N, batched, hidden_size, vocab_size, pooling_fn):
        super(ManualDecoder, self).__init__()
        self.layer = layer
        self.layers = clones(layer, N - 1)
        self.norm = nn.LayerNorm(hidden_size)
        self.memory_layer = Memory(hidden_size, batched=batched)
        self.pooler = pooling_fn
        self.lin = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, encoded, src_padding_mask, memories, mem_keys, memory_padding_masks, tgt_mask, tgt_padding_mask):
        #print(tgt.size(), encoded.size(), memories.size(), mem_keys.size(), memory_padding_masks.size(), tgt_mask.size(), tgt_padding_mask.size())
        #tgt(bsz, max_seq-1, emb); encoded(bsz, max_seq, emb); memories(n_mems, bsz, max_seq, emb);
        #mem_keys(n_mems, bsz, emb); memory_padding_masks(n_mems, bsz, max_seq); tgt_mask(bsz, max_seq-1, max_seq-1);
        #tgt_padding_mask(bsz, max_seq-1);
        
        memories = self.memory_layer(self.pooler(encoded), memories, mem_keys)
        output = self.layer(tgt, encoded, tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask = src_padding_mask)
        for mem, mem_pad_mask in zip(memories, memory_padding_masks):
            output = self.layer.forward(output, mem, tgt_mask, tgt_key_padding_mask = tgt_padding_mask, memory_key_padding_mask = mem_pad_mask)
        for layer in self.layers:
            output = layer.forward(output, encoded, tgt_mask, tgt_key_padding_mask = tgt_padding_mask, memory_key_padding_mask = src_padding_mask)
        return self.lin(self.norm(output))

class FineTuneTransformer(nn.Module):
    def __init__(self, LLM, decoder, BOS_token_id, EOS_token_id, device, dropout = 0.1):
        super(FineTuneTransformer, self).__init__()
        self.encoder = LLM
        self.decoder = decoder
        self.embed = LLM.get_input_embeddings()
        self.BOS = BOS_token_id
        self.EOS = EOS_token_id
        self.device = device
    
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
    
    def encode(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return output.last_hidden_state

    def decode(self, tgt, encoded, memories, mem_keys, memory_mask, src_padding_mask, tgt_padding_mask):
        output = self.decoder(
            self.embed(tgt), encoded, src_padding_mask, memories, mem_keys, memory_mask, 
            self.get_tgt_mask(tgt), tgt_padding_mask)
        return output
    
    def forward(self, input_ids, kv_store, keys, memory_masks, attention_mask, token_type_ids):
        # Note that this implementation is for teacher forcing.
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        with torch.no_grad():
            encoding = self.encode(input_ids, attention_mask, token_type_ids)
            out_seq = [[self.BOS]]
            while out_seq[0][-1] != self.EOS or len(out_seq[0]) > 271:
                tgt = self.embed(torch.tensor(out_seq, dtype=torch.long, device=self.device))
                #print(tgt.size())
                #No tgt_padding_maks. We are making tgt
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[1]).to(device)
                response_logits = self.decoder(tgt, encoding, attention_mask.to(torch.float32), kv_store, keys, memory_masks, tgt_mask.to(torch.float32), None)
                #print(response_logits.size())
                token_id = torch.argmax(response_logits[:, -1, :], dim=-1)
                #print("---------------------------------")
                out_seq[0].append(token_id.item())
            return out_seq, encoding

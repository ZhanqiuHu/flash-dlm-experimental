import torch
from typing import Optional, Dict, Any, Tuple

class BlockCache(torch.nn.Module):
    def __init__(self, batch_size: int, max_length: int, block_size: int, 
        num_heads: int,
        num_key_value_heads: int,
        hidden_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.block_size = block_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype is not None else torch.float16

        # Initialize caches with correct device and dtype
        self.key_cache = torch.zeros(batch_size, max_length, hidden_dim * num_key_value_heads, device=self.device, dtype=self.dtype)
        self.value_cache = torch.zeros(batch_size, max_length, hidden_dim * num_key_value_heads, device=self.device, dtype=self.dtype)
        self.query_cache = torch.zeros(batch_size, max_length, hidden_dim * num_heads, device=self.device, dtype=self.dtype)

        self.clean_cache_idx = 0
        self.next_cache_idx = 0

        # TODO: Initialize position embeddings
        self.cos_pos_emb = torch.zeros(batch_size, max_length, hidden_dim, device=self.device, dtype=self.dtype)
        self.sin_pos_emb = torch.zeros(batch_size, max_length, hidden_dim, device=self.device, dtype=self.dtype)
        

    def update_cache(self, new_q, new_k, new_v, new_cos_pos_emb, new_sin_pos_emb):
        start = self.clean_cache_idx
        end = start + new_q.shape[1]

        # Move inputs to correct device and dtype if needed
        new_k = new_k.to(device=self.device, dtype=self.dtype)
        new_v = new_v.to(device=self.device, dtype=self.dtype)
        new_q = new_q.to(device=self.device, dtype=self.dtype)

        self.key_cache[:, start:end] = new_k
        self.value_cache[:, start:end] = new_v
        self.query_cache[:, start:end] = new_q

        # Update position embeddings
        self.cos_pos_emb[:, start:end] = new_cos_pos_emb
        self.sin_pos_emb[:, start:end] = new_sin_pos_emb
        
        self.next_cache_idx = end

    
    def get_cache(self):
        # up to the next_cache_idx
        # return the q, k, v up to the next_cache_idx
        q = self.query_cache[:, :self.next_cache_idx]
        k = self.key_cache[:, :self.next_cache_idx]
        v = self.value_cache[:, :self.next_cache_idx]
        cos_pos_emb = self.cos_pos_emb[:, :self.next_cache_idx]
        sin_pos_emb = self.sin_pos_emb[:, :self.next_cache_idx]
        return q, k, v, cos_pos_emb, sin_pos_emb
    
    def save_cache(self, clean_idx=None):
        if clean_idx is not None:
            self.clean_cache_idx = clean_idx
        else:
            self.clean_cache_idx = self.next_cache_idx
    

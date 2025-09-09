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
        

    def update_cache(self, new_q, new_k, new_v, new_cos_pos_emb=None, new_sin_pos_emb=None):
        start = self.clean_cache_idx
        end = start + new_q.shape[1]
        # print out new_q.shape[1]
        # print(f"update_cache: new_q.shape: {new_q.shape}")
        # print(f"update_cache: new_k.shape: {new_k.shape}")
        # print(f"update_cache: new_v.shape: {new_v.shape}")
        # # self.key_cache[:, start:end] = new_k
        # print(f"start: {start}, end: {end}")

        # Move inputs to correct device and dtype if needed
        new_k = new_k.to(device=self.device, dtype=self.dtype)
        new_v = new_v.to(device=self.device, dtype=self.dtype)
        new_q = new_q.to(device=self.device, dtype=self.dtype)

        self.key_cache[:, start:end] = new_k
        self.value_cache[:, start:end] = new_v
        self.query_cache[:, start:end] = new_q

        # Update position embeddings (optional)
        if new_cos_pos_emb is not None and new_sin_pos_emb is not None:
            self.cos_pos_emb[:, start:end] = new_cos_pos_emb
            self.sin_pos_emb[:, start:end] = new_sin_pos_emb
        
        self.next_cache_idx = end

        # insert debug breakpoint here
        # import pdb; pdb.set_trace()
    
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

    def __str__(self):
        # in dict format for all attributes and their values
        return str(self.__dict__)
    
    def __repr__(self):
        # in dict format for all attributes and their values
        return str(self.__dict__)



class BlockCacheV2(torch.nn.Module):
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.batch_size           = batch_size
        self.max_length           = max_length
        self.num_heads            = num_heads
        self.num_key_value_heads  = num_key_value_heads
        self.head_dim             = head_dim

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = dtype  or torch.float16

        # keep separate head dims: (batch, heads, time, head_dim)
        self.key_cache   = torch.zeros(batch_size, num_key_value_heads, max_length, head_dim, device=self.device, dtype=self.dtype)
        self.value_cache = torch.zeros(batch_size, num_key_value_heads, max_length, head_dim, device=self.device, dtype=self.dtype)
        self.query_cache = torch.zeros(batch_size, num_heads,         max_length, head_dim, device=self.device, dtype=self.dtype)

        # optional rotary tables at per-position granularity
        self.cos_pos_emb = torch.zeros(batch_size, max_length, head_dim, device=self.device, dtype=self.dtype)
        self.sin_pos_emb = torch.zeros(batch_size, max_length, head_dim, device=self.device, dtype=self.dtype)

        # pointers into cache
        self.clean_cache_idx = 0
        self.next_cache_idx  = 0

    def update_cache(
        self,
        new_q: torch.Tensor,   # (B, num_heads,        L, head_dim)
        new_k: torch.Tensor,   # (B, num_kv_heads,     L, head_dim)
        new_v: torch.Tensor,   # (B, num_kv_heads,     L, head_dim)
        new_cos_pos_emb: Optional[torch.Tensor] = None,  # (B, L, head_dim)
        new_sin_pos_emb: Optional[torch.Tensor] = None,  # (B, L, head_dim)
    ):
        start = self.clean_cache_idx
        L = new_q.shape[2]
        end = start + L
        assert end <= self.max_length, "Cache overflow"

        # move to correct device/dtype
        new_q = new_q.to(self.device, self.dtype)
        new_k = new_k.to(self.device, self.dtype)
        new_v = new_v.to(self.device, self.dtype)

        # store head-split tensors directly
        self.query_cache[:, :, start:end, :] = new_q
        self.key_cache  [:, :, start:end, :] = new_k
        self.value_cache[:, :, start:end, :] = new_v

        # optionally store rotary embeddings
        if new_cos_pos_emb is not None and new_sin_pos_emb is not None:
            self.cos_pos_emb[:, start:end] = new_cos_pos_emb.to(self.device, self.dtype)
            self.sin_pos_emb[:, start:end] = new_sin_pos_emb.to(self.device, self.dtype)

        self.next_cache_idx = end

    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          full_k       (batch, num_kv_heads, time, head_dim),
          full_v       (batch, num_kv_heads, time, head_dim),
          full_q       (batch, num_heads,     time, head_dim),
          cos_pos_emb  (batch, time, head_dim),
          sin_pos_emb  (batch, time, head_dim)
        where time = next_cache_idx.
        """
        k   = self.key_cache  [:, :, :self.next_cache_idx, :]
        v   = self.value_cache[:, :, :self.next_cache_idx, :]
        q   = self.query_cache[:, :, :self.next_cache_idx, :]
        cos = self.cos_pos_emb[:, :self.next_cache_idx]
        sin = self.sin_pos_emb[:, :self.next_cache_idx]
        return q, k, v, cos, sin

    def save_cache(self, clean_idx: Optional[int] = None):
        """
        Advance the clean pointer.  Tokens before clean_idx
        will be overwritten by the next update.
        """
        self.clean_cache_idx = self.next_cache_idx if clean_idx is None else clean_idx

    def __repr__(self):
        return (
            f"BlockCache(batch={self.batch_size}, "
            f"time=[{self.clean_cache_idx}:{self.next_cache_idx}], "
            f"heads_q={self.num_heads}, heads_kv={self.num_key_value_heads}, "
            f"head_dim={self.head_dim})"
        )

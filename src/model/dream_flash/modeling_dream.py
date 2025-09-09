"""
Dream model implementation with diffusion generation
"""

import math
from typing import List, Optional, Tuple, Union
import os
import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from transformers import PretrainedConfig

from src.model.dream_flash.configuration_dream import ODreamConfig
from src.model.dream_flash.generation_utils import DreamGenerationMixin, DreamGenerationConfig
from src.model.dream_flash.block_utils import BlockCache

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "Dream-7B"
_CONFIG_FOR_DOC = "ODreamConfig"


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Dream
class DreamRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DreamRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Dream
class DreamRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[ODreamConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`DreamRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def reset_parameters(self):
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, self.inv_freq.device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
    

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_v2(q_or_k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_or_k_embed = (q_or_k * cos) + (rotate_half(q_or_k) * sin)
    return q_or_k_embed


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Dream
class DreamMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class DreamAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: ODreamConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = False
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = DreamRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_v2(query_states, cos, sin), apply_rotary_pos_emb_v2(key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

class DreamSdpaAttention(DreamAttention):
    """
    Dream attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `DreamAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    def __init__(self, config: ODreamConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.block_cache = None

    def reset_block_cache(self, batch_size: int, max_length: int, block_size: int):
        self.block_cache = BlockCache(
            batch_size=batch_size,
            max_length=max_length,
            block_size=block_size,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_dim=self.head_dim, 
            # Note: self.hidden_size is h x d, not d
            # self.head_dim is d
            device=self.q_proj.weight.device,  # Get device from model parameter
            dtype=self.q_proj.weight.dtype  # Get dtype from model parameter
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,

        qkv_cache: Optional[DynamicCache] = None,
        use_block_diffusion: Optional[bool] = None,
        block_size: Optional[int] = None,
        save_cache: Optional[bool] = None,
        clean_idx: Optional[int] = None,
        use_full_query_attn: Optional[bool] = None,


        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if use_block_diffusion: # for block diffusion, save_cache must be provided
            assert save_cache is not None, "save_cache must be provided"
        
        if use_block_diffusion and block_size is None:
            raise ValueError("block_size must be set if use_block_diffusion is enabled")

        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "DreamModel is using DreamSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                qkv_cache=qkv_cache,
                use_block_diffusion=use_block_diffusion,
                block_size=block_size,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len_in, _ = hidden_states.size()

        
        qkv_cache = self.block_cache


        if qkv_cache is not None:
            assert use_block_diffusion is not None, "use_block_diffusion must be provided"
            assert block_size is not None, "block_size must be provided"
            assert use_full_query_attn is not None, "use_full_query_attn must be provided"

            new_q = self.q_proj(hidden_states) # [bsz, block_size, num_heads * head_dim] at denoising step
            new_k = self.k_proj(hidden_states) # [bsz, block_size, num_key_value_heads * head_dim] at denoising step
            new_v = self.v_proj(hidden_states) # [bsz, block_size, num_key_value_heads * head_dim] at denoising step

            new_cos_pos_emb, new_sin_pos_emb = self.rotary_emb(new_v, position_ids)


            qkv_cache.update_cache(new_q, new_k, new_v, new_cos_pos_emb, new_sin_pos_emb)
            if save_cache:
                qkv_cache.save_cache(clean_idx=clean_idx)
            

            # Note: get_cache returns q, k, v up to next_cache_idx (including prompts and previous blocks)
            query_states, key_states, value_states, cos_pos_emb, sin_pos_emb = qkv_cache.get_cache()
            # key_states: [bsz, clean_tokens + block_size, num_key_value_heads * head_dim] at denoising step
            # value_states: [bsz, clean_tokens + block_size, num_key_value_heads * head_dim] at denoising step

            if not use_full_query_attn:
                query_states = new_q # [bsz, block_size, num_heads * head_dim] at denoising step
            else:
                # use full query attention
                new_cos_pos_emb, new_sin_pos_emb = cos_pos_emb, sin_pos_emb


            # TODO: cache and update position_embeddings to include up to current block
            position_embeddings = (cos_pos_emb, sin_pos_emb)




        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)



        # update q_len to be up to current block
        q_len = query_states.shape[1]
        kv_len = key_states.shape[1]



        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
            assert False, "Should not be here, not supported yet"
        else:
            if use_block_diffusion:
                cos, sin = position_embeddings
            else:
                cos, sin = position_embeddings
                new_cos_pos_emb, new_sin_pos_emb = cos, sin

        
        if use_block_diffusion:
            # if use_full_query_attn
            # Q = [cached_query_states, new_query_states] # [bsz, clean_tokens + block_size, num_heads * head_dim]
            # else Q = [new_query_states] # [bsz, block_size, num_heads * head_dim]
            if not use_full_query_attn:
                query_states = apply_rotary_pos_emb_v2(query_states, new_cos_pos_emb, new_sin_pos_emb)
                key_states = apply_rotary_pos_emb_v2(key_states, cos, sin)
            else:
                query_states = apply_rotary_pos_emb_v2(query_states, cos, sin)
                key_states = apply_rotary_pos_emb_v2(key_states, cos, sin)
        else:
            query_states = apply_rotary_pos_emb_v2(query_states, cos, sin)
            key_states = apply_rotary_pos_emb_v2(key_states, cos, sin)
        

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # Ensure all tensors have the same dtype before attention
        dtype = self.q_proj.weight.dtype
        query_states = query_states.to(dtype=dtype)
        key_states = key_states.to(dtype=dtype)
        value_states = value_states.to(dtype=dtype)


        # Compute attention weights if output_attentions is True
        attn_weights = None
        if output_attentions:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False, # hard coded
            )

        # cut off attn_output to be the last q_len_in tokens
        attn_output = attn_output[:, :, -q_len_in:]

        attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = attn_output.view(bsz, q_len_in, self.hidden_size)

        

        # Ensure output has correct dtype before projection
        attn_output = attn_output.to(dtype=dtype)
        attn_output = self.o_proj(attn_output)
        

        

        # return attn_output, None, past_key_value
        return attn_output, attn_weights, qkv_cache


class DreamDecoderLayer(nn.Module):
    def __init__(self, config: ODreamConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        
        # self.self_attn = Dream_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.self_attn = DreamSdpaAttention(config, layer_idx)

        self.mlp = DreamMLP(config)
        self.input_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.num_heads = config.num_attention_heads

    
    def reset_block_cache(self, batch_size: int, max_length: int, block_size: int):
        # self.block_cache = BlockCache(
        #     batch_size=batch_size,
        #     max_length=max_length,
        #     block_size=block_size,
        #     # num_heads=self.num_heads,
        #     num_heads=self.num_heads,
        #     num_key_value_heads=self.num_key_value_heads,
        #     hidden_dim=self.head_dim, # Note: self.hidden_size is h x d, not d
        #     # self.head_dim is d
        # )
        self.self_attn.reset_block_cache(batch_size, max_length, block_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,

        # past_key_value: Optional[Tuple[torch.Tensor]] = None,
        qkv_cache: Optional[DynamicCache] = None,
        use_block_diffusion: Optional[bool] = None,
        block_size: Optional[int] = None,
        save_cache: Optional[bool] = None,
        clean_idx: Optional[int] = None,
        use_full_query_attn: Optional[bool] = None,

        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        if use_block_diffusion:
            assert save_cache is not None, "save_cache must be provided"

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,

            # past_key_value=past_key_value,
            qkv_cache=qkv_cache, # rename to qkv_cache
            use_block_diffusion=use_block_diffusion,
            use_full_query_attn=use_full_query_attn,
            block_size=block_size,
            save_cache=save_cache,
            clean_idx=clean_idx,
            
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        hidden_states = residual + hidden_states

        

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class DreamPreTrainedModel(PreTrainedModel):
    config_class = ODreamConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DreamDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ):
        _model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        # NOTE(Lin): we need to override the generation config
        # because the generation config loaded in `from_pretrained` 
        # does not include all the attributes of DreamGenerationConfig
        resume_download = kwargs.get("resume_download", None)
        proxies = kwargs.get("proxies", None)
        subfolder = kwargs.get("subfolder", "")
        from_auto_class = kwargs.get("_from_auto", False)
        from_pipeline = kwargs.get("_from_pipeline", None)
        _model.generation_config = DreamGenerationConfig.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
        )
        return _model

class DreamBaseModel(DreamPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DreamDecoderLayer`]

    Args:
        config: ODreamConfig
    """

    def __init__(self, config: ODreamConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DreamDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DreamRotaryEmbedding(config=config)

        self.cache_initialized = False


        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def reset_block_cache(self, bsz: int, L: int, block_size: int):
        for layer in self.layers:
            layer.reset_block_cache(bsz, L, block_size)
        self.cache_initialized = True

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,

        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        qkv_cache: Optional[DynamicCache] = None,
        use_block_diffusion: Optional[bool] = None,
        block_size: Optional[int] = None,
        use_full_query_attn: Optional[bool] = None,
        save_cache: Optional[bool] = None,
        clean_idx: Optional[int] = None,
        max_length: Optional[int] = None,

        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        if use_block_diffusion:
            assert save_cache is not None, "save_cache must be provided"
            assert max_length is not None, "max_length must be provided"

        # Use config values as defaults for block diffusion parameters
        use_block_diffusion = self.config.use_block_diffusion if use_block_diffusion is None else use_block_diffusion
        block_size = self.config.block_size if block_size is None else block_size
        
        if use_block_diffusion and block_size is None:
            raise ValueError("block_size must be set if use_block_diffusion is enabled")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # if use_cache and past_key_values is None:
        if use_cache and qkv_cache is None and not self.cache_initialized:
            # this should not be called once at the beginning of the generation
            bsz = inputs_embeds.shape[0]
            L = full_seq_len = max_length

            # if use_block_diffusion and use_cache
            if use_block_diffusion and use_cache:   
                for layer_idx in range(self.config.num_hidden_layers):
                    self.layers[layer_idx].reset_block_cache(bsz, L, block_size)
                self.cache_initialized = True
        else:
            if not self.cache_initialized:
                # should not be here: does not support passing in qkv_cache for now
                raise ValueError("does not support passing in qkv_cache for now")
                # qkv_cache = qkv_cache

        if cache_position is None:
            # # past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            # clean_tokens = qkv_cache.get_seq_length() if qkv_cache is not None else 0
            block_cache = self.layers[0].self_attn.block_cache
            clean_tokens = block_cache.clean_cache_idx if block_cache is not None else 0
            # # cache_position = torch.arange(
            # #     past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            # # )
            cache_position = torch.arange(
                clean_tokens, clean_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

            # Dont use cache_position for now


        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,

                    # past_key_values,
                    qkv_cache,
                    use_block_diffusion,
                    block_size,
                    save_cache,
                    clean_idx,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,

                    # NEW: block diffusion
                    # past_key_value=past_key_values,
                    qkv_cache=qkv_cache,
                    use_block_diffusion=use_block_diffusion,
                    use_full_query_attn=use_full_query_attn,
                    block_size=block_size,
                    save_cache=save_cache,
                    clean_idx=clean_idx,

                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DreamForCausalLM(DreamGenerationMixin, DreamPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DreamBaseModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        GREEN = "\033[92m"
        BOLD = "\033[1m"
        END = "\033[0m" 
        print(f"\n{GREEN}{BOLD}INFO: Using Dream-Flash implementation{END}\n")
 
        # Initialize weights and apply final processing
        self.post_init()

    def reset_rope_parameters(self):
        self.model.rotary_emb.reset_parameters()
        for layer in self.model.layers:
            layer.self_attn.rotary_emb.reset_parameters()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,

        # NEW: support saving cache
        save_cache: Optional[bool] = None,  
        clean_idx: Optional[int] = None,
        max_length: Optional[int] = None,
        use_block_diffusion: Optional[bool] = None,
        block_size: Optional[int] = None,
        use_full_query_attn: Optional[bool] = None,

        # NEW: attention weights saving
        save_attention_weights: Optional[bool] = None,
        attention_weights_path: Optional[str] = None,
        step_idx: Optional[int] = None,

        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, MaskedLMOutput]:
        if use_block_diffusion:
            assert save_cache is not None, "save_cache must be provided"

        # Set output_attentions to True if we need to save attention weights
        if save_attention_weights:
            output_attentions = True

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            
            # NEW: block diffusion
            # past_key_values=past_key_values,
            use_block_diffusion=use_block_diffusion,
            use_full_query_attn=use_full_query_attn,
            block_size=block_size,
            qkv_cache=None, # rename to qkv_cache
            save_cache=save_cache,
            clean_idx=clean_idx,
            max_length=max_length,
            


            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        # Save attention weights if requested
        if save_attention_weights and attention_weights_path is not None and step_idx is not None:
            if not os.path.exists(attention_weights_path):
                os.makedirs(attention_weights_path)
            
            # Save attention weights for each layer
            for layer_idx, layer_attn in enumerate(outputs.attentions):
                attn_path = os.path.join(attention_weights_path, f"step_{step_idx}_layer_{layer_idx}.pt")
                torch.save(layer_attn.detach().cpu(), attn_path)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
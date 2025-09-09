"""
Configuration class for Dream model
"""

from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from enum import Enum
from typing import Optional, Union
from dataclasses import dataclass, field

class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"

class LayerNormType(StrEnum):
    default = "default"
    low_precision = "low_precision"
    rms = "rms"
    gemma_rms = "gemma_rms"
    amd_compatible = "amd_compatible"

class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    silu = "silu"
    swiglu = "swiglu"

class BlockType(StrEnum):
    sequential = "sequential"
    parallel = "parallel"
    llama = "llama"
    dream = "dream"

class InitFnType(StrEnum):
    mitchell = "mitchell"
    normal = "normal"
    kaiming_normal = "kaiming_normal"
    fan_in = "fan_in"
    full_megatron = "full_megatron"

@dataclass
class DreamConfig(PretrainedConfig):
    model_type = "Dream"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,  # cache not used in diffusion
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        mask_token_id=151666,
        pad_token_id=151643,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id


class ODreamConfig(DreamConfig):
    model_type = "odream"
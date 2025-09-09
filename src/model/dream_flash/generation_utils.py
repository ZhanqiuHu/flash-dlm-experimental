"""
Adopted from HF repository
https://huggingface.co/Dream-org/Dream-v0-Instruct-7B
"""
import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__, AutoTokenizer
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)

# COLOR (for logging and printing)
YELLOW = "\033[93m"
PURPLE = "\033[95m"
RED = "\033[91m"
BLUE = "\033[94m"
GREEN = "\033[92m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"

tokenizer = AutoTokenizer.from_pretrained(
    "Dream-org/Dream-v0-Instruct-7B",
    trust_remote_code=True
)

# TODO: add a tokenizer for dream-7b for debugging

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, 
    top_k=None, margin_confidence=False, neg_entropy=False, return_probs=False):

    # Zhanqiu modified to fix float16 issue: change dtype to float64
    logits = logits.to(torch.float32)

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        # Zhanqiu modified to fix float16 issue: set bigger epsilon
        # epsilon = 1e-4
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    if return_probs:
        return confidence, x0, probs
    
    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 32)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)

        # NEW FEATURE: early stopping
        self.early_stop: bool = kwargs.pop("early_stop", False)
        self.early_stop_consecutive: int = \
            kwargs.pop("early_stop_consecutive", 5)  # Default to 5 consecutive EOS tokens

        # NEW FEATURE: confidence-based adaptive unmasking 
        # (potentially more aggressive unmasking)
        self.confidence_based_adaptive_unmasking: bool = kwargs.pop("confidence_based_adaptive_unmasking", False)
        self.confidence_threshold: float = kwargs.pop("confidence_threshold", 0.9)

        # NEW FEATURE: decay algorithm and parameters
        self.decay_algorithm: str = kwargs.pop("decay_algorithm", "exp_unmasked_ratio")
        self.decay_params: Dict[str, Any] = kwargs.pop("decay_params", {
            "alpha": 1.0,  # base decay rate
            "gamma": 1.0,  # controls how fast alpha decays with unmasking
        })

        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # NEW: ----- block diffusion: size of each block for block‐wise updates -----
        self.use_block_diffusion: bool = kwargs.pop("use_block_diffusion", False)
        self.block_size: Optional[int] = kwargs.pop("block_size", None)
        self.use_full_query_attn: bool = kwargs.pop("use_full_query_attn", False)

        # NEW: sliding window caching parameters
        self.sliding_window_caching: bool = kwargs.pop("sliding_window_caching", False)
        self.sliding_window_size: int = kwargs.pop("sliding_window_size", 128)

        # NEW: attention weights saving parameters
        self.save_attention_weights: bool = kwargs.pop("save_attention_weights", False)
        self.attention_weights_path: Optional[str] = kwargs.pop("attention_weights_path", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)



        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        assert generation_config.max_length is not None, "max_length must be set"

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )


        # Logging: print out warning if use_block_diffusion or early_stop are disabled
        # in red
        RED = "\033[91m"
        RESET = "\033[0m"
        GREEN = "\033[92m"
        if not generation_config.use_block_diffusion:
            print(f"{RED}WARNING: use_block_diffusion is disabled. Inference will be very slow. This is not recommended.{RESET}")
        else:
            print(f"{GREEN}INFO: use_block_diffusion is enabled.{RESET}")
        if not generation_config.early_stop:
            print(f"{RED}WARNING: early_stop is disabled. This is not recommended.{RESET}")
        else:
            print(f"{GREEN}INFO: early_stop is enabled.{RESET}")

        # reset cache if use_block_diffusion is enabled
        if generation_config.use_block_diffusion:
            bsz = input_ids.shape[0]
            L = generation_config.max_length
            block_size = generation_config.block_size
            print(f"Resetting cache: bsz={bsz}, L={L}, block_size={block_size}")
            # Note: this is now moved within _sample_block_diffusion
            # for layer_idx in range(self.config.num_hidden_layers):
            #     self.model.layers[layer_idx].reset_block_cache(bsz, L, block_size)
            self.cache_initialized = True
        else:
            print("Not using block diffusion")
            self.cache_initialized = False


        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # Common initialization
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        max_new_tokens = generation_config.max_new_tokens

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        if generation_config.use_block_diffusion:
            GREEN = "\033[92m"
            RESET = "\033[0m"
            print(f"{GREEN}Using block diffusion sampling{RESET}", flush=True)
            sequences = self._sample_block_diffusion(
                x, attention_mask, tok_idx,
                generation_config,
                generation_tokens_hook_func,
                generation_logits_hook_func,
                histories
            )
            # sequences = self._sample_block_diffusion_v2(
            #     x, attention_mask, tok_idx,
            #     generation_config,
            #     generation_tokens_hook_func,
            #     generation_logits_hook_func,
            #     histories
            # )
        else:
            sequences = self._sample_baseline(
                x, attention_mask, tok_idx,
                generation_config,
                generation_tokens_hook_func,
                generation_logits_hook_func,
                histories
            )

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=sequences,
                history=histories,
            )
        else:
            return sequences

    # Helper function for block diffusion sampling
    def _sample_block_diffusion(
        self, x, attention_mask, tok_idx,
        generation_config,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        histories
        ):

        BLUE = "\033[94m"   
        RESET = "\033[0m"
        print(f"Inside dream-flash block diffusion", flush=True)

        steps, eps, alg, alg_temp, temperature, top_p, top_k, mask_token_id = (
            generation_config.steps,
            generation_config.eps,
            generation_config.alg,
            generation_config.alg_temp,
            generation_config.temperature,
            generation_config.top_p,
            generation_config.top_k,
            generation_config.mask_token_id
        )
        block_size = generation_config.block_size
        use_full_query_attn = generation_config.use_full_query_attn
        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        prompt_len = (x != mask_token_id).sum(dim=-1).max().item()

        # num_blocks = steps // block_size
        # TODO: fix this to -- num_blocks = max new tokens // block_size
        try:
            num_blocks = generation_config.max_new_tokens // block_size
        except:
            import pdb;pdb.set_trace()
        # For now, enforce the max_new_tokens to be divisible by block_size
        assert generation_config.max_new_tokens % block_size == 0, "max_new_tokens must be divisible by block_size"
        num_steps_per_block = steps // num_blocks
        # For now, enforce the steps to be divisible by num_blocks
        assert steps % num_blocks == 0, "steps must be divisible by num_blocks"

        avg_tokens_per_step = generation_config.max_new_tokens / steps

        # NEW FEATURE: early stopping
        # Keep track of the last clean token position and token
        last_clean_token_position = -1 
        last_clean_token = None

        for block_idx in range(num_blocks):
            # Iterating over blocks
            block_start = prompt_len + block_idx * block_size
            block_end = prompt_len + (block_idx + 1) * block_size

            # for step_idx in range(block_size):
            for step_idx in range(num_steps_per_block):

                # i = block_idx * block_size + step_idx
                # fix this to -- i = block_idx * num_steps_per_block + step_idx
                i = block_idx * num_steps_per_block + step_idx # i is the global step index


                x_partial = x[:, block_start:block_end].clone()
                block_mask_index = x_partial == mask_token_id
                # save_cache = step_idx == block_size - 1 # save cache for the last step of the block
                # last step of the block
                save_cache = step_idx == num_steps_per_block - 1

                # Another attempt to recover accuracy
                bsz = x.shape[0]
                L = generation_config.max_length
                end_of_sentence = L 
                
                if i == 0:
                    # At the first global step i, reset the block cache
                    for layer_idx in range(self.config.num_hidden_layers):
                        self.model.layers[layer_idx].reset_block_cache(bsz, L, block_size)

                    # Pass in the entire input x, and the block caching will be used
                    # We get the logits of corresponding block 
                    logits = self(x, None, tok_idx if tok_idx is not None else None,
                                use_block_diffusion=True,
                                use_full_query_attn=use_full_query_attn,
                                max_length=generation_config.max_length,
                                block_size=block_size,
                                save_cache=True,
                                # clean_idx=prompt_len).logits[:, block_start:block_end]
                                clean_idx=block_start-1).logits
                    # Shift logits before cropping to the block
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                    logits = logits[:, block_start:block_end]
                else:
                    # assert length of x_partial_with_room is 2 x block_size
                    
                    # feed in x[:, block_start:]
                    logits = self(x[:, block_start-1:], None, tok_idx[:, block_start:] if tok_idx is not None else None,
                                use_block_diffusion=True,
                                use_full_query_attn=use_full_query_attn,
                                max_length=generation_config.max_length,
                                block_size=block_size,
                                save_cache=save_cache,
                                clean_idx=block_end-1).logits # output length: block start until the end of the sequence

                    # Shift logits before cropping to the block
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)  
                    # but only get the logits of the corresponding block
                    logits = logits[:, 1:block_size+1]
                mask_logits = logits[block_mask_index]
            

                t, s = timesteps[i], timesteps[i + 1]

                if alg == 'origin':
                    # p_transfer = 1 - s / t if step_idx < block_size - 1 else 1
                    p_transfer = 1 - s / t if step_idx < num_steps_per_block - 1 else 1
                    x0 = torch.zeros_like(x_partial[block_mask_index], device=self.device, dtype=torch.long) + mask_token_id
                    transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                    _, x0[transfer_index_t_s] = sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                    x_partial[block_mask_index] = x0.clone()
                    x[:, block_start:block_end] = x_partial.clone()

                    if generation_config.early_stop:
                        if transfer_index_t_s.any():
                            last_pos = torch.nonzero(block_mask_index)[transfer_index_t_s].max().item()
                            last_clean_token_position = max(last_clean_token_position, block_start + last_pos)
                            last_clean_token = x[0, last_clean_token_position].item()

                else:
                    if alg == 'maskgit_plus':
                        confidence, x0 = sample_tokens(
                            mask_logits, 
                            temperature=temperature, 
                            top_p=top_p, 
                            top_k=top_k,
                            )
                    elif alg == 'topk_margin':
                        confidence, x0 = sample_tokens(
                            mask_logits, 
                            temperature=temperature, 
                            top_p=top_p, 
                            top_k=top_k, 
                            margin_confidence=True,                            
                            )
                    elif alg == 'entropy':
                        confidence, x0 = sample_tokens(
                            mask_logits, 
                            temperature, 
                            top_p=top_p, 
                            top_k=top_k, 
                            neg_entropy=True,
                            )
                    elif alg == 'position_weighted':
                        # Sample tokens normally
                        confidence, x0 = sample_tokens(
                            mask_logits,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                        )
                        # Apply position-based weights to confidence
                        # Get positions of masked tokens
                        positions = torch.nonzero(block_mask_index, as_tuple=False).squeeze(-1)
                        # Calculate weights: higher for lower positions (more left)
                        # Normalize positions to [0, 1] range and invert (1 - pos) to give higher weights to lower positions
                        max_pos = positions.max().item() if positions.numel() > 0 else 1
                        pos_weights = 1.0 - (positions.float() / max_pos)
                        # Apply weights to confidence
                        confidence = confidence * pos_weights
                    else:
                        raise RuntimeError(f"Unknown alg: {alg}")

                num_mask_token = block_mask_index.sum()
                total_num_mask_token = num_mask_token + (num_blocks - block_idx - 1) * block_size
                # number_transfer_tokens = int(total_num_mask_token * (1 - s / t)) if step_idx < block_size - 1 else num_mask_token
                number_transfer_tokens = int(total_num_mask_token * (1 - s / t)) \
                    if step_idx < num_steps_per_block - 1 \
                        else num_mask_token # unmask all masked tokens at last step of the block

                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                    else:
                        confidence = confidence / alg_temp
                        confidence = F.softmax(confidence, dim=-1)
                        transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
                    x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + mask_token_id
                    x0_[transfer_index] = x0[transfer_index].clone()
                    x_partial[block_mask_index] = x0_.clone()
                    x[:, block_start:block_end] = x_partial.clone()


                    if generation_config.early_stop:
                        last_pos = torch.nonzero(block_mask_index)[transfer_index].max().item()
                        last_clean_token_position = max(last_clean_token_position, block_start + last_pos)
                        last_clean_token = x[0, last_clean_token_position].item()


                x = generation_tokens_hook_func(i, x, logits)

                if histories is not None:
                    histories.append(x.clone())

                if generation_config.early_stop:
                    if self._check_early_stop(x, last_clean_token_position, last_clean_token, generation_config):
                        print(f"Early stopping at block {block_idx}, step {step_idx}, i: {i}", flush=True)
                        mask_index_remaining = (x == generation_config.mask_token_id)
                        x[mask_index_remaining] = generation_config.eos_token_id
                        return x


        return x

    def _check_early_stop(self, x, last_clean_token_position, last_clean_token, generation_config):
        """
        Check early stopping condition:
        1. The last clean token must be EOS.
        2. The last `early_stop_consecutive` unmasked tokens are all EOS tokens.
        3. All tokens before the last clean token position are unmasked (no MASK tokens).
        """
        eos_token_id = generation_config.eos_token_id
        mask_token_id = generation_config.mask_token_id
        early_stop_consecutive = generation_config.early_stop_consecutive

        # Condition 0: last clean token must be EOS
        if last_clean_token != eos_token_id:
            return False

        # Only support batch_size=1
        x_seq = x[0, :last_clean_token_position + 1]  # Up to last clean token

        # Condition 2: all tokens before last_clean_token_position are unmasked
        if (x_seq == mask_token_id).any():
            return False  # MASK tokens still exist

        # Condition 1: last `early_stop_consecutive` tokens are EOS
        if last_clean_token_position + 1 < early_stop_consecutive:
            return False  # not enough tokens

        last_tokens = x[0, last_clean_token_position - early_stop_consecutive + 1 : last_clean_token_position + 1]
        if (last_tokens == eos_token_id).all():
            return True

        return False


    # Helper function for standard sampling
    def _sample_baseline(
        self, x, attention_mask, tok_idx,
        generation_config,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        histories
    ):
        steps, eps, alg, alg_temp, temperature, top_p, top_k, mask_token_id = (
            generation_config.steps,
            generation_config.eps,
            generation_config.alg,
            generation_config.alg_temp,
            generation_config.temperature,
            generation_config.top_p,
            generation_config.top_k,
            generation_config.mask_token_id
        )

        enable_confidence_based = generation_config.confidence_based_adaptive_unmasking


        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
        avg_tokens_per_step = (x == mask_token_id).sum().item() / steps

        x = generation_tokens_hook_func(None, x, None)

        prompt_len = (x != mask_token_id).sum(dim=-1).max().item()
        last_clean_token_position = prompt_len - 1
        last_clean_token = x[0, last_clean_token_position]


        def expected_mth_largest(min_val, max_val, N, m=2):
            """
            Compute the expected value of the second-largest sample (top 2)
            assuming uniform distribution between min_val and max_val.

            Args:
                min_val (float): Minimum value of the dataset.
                max_val (float): Maximum value of the dataset.
                N (int): Total number of samples.

            Returns:
                float: Expected value of the second-largest sample.
            """
            if m < 1 or m > N:
                raise ValueError("m must be between 1 and N.")
            
            order_stat_rank = N - m + 1  # Convert top-m to order statistic index
            expected_value = min_val + (order_stat_rank / (N + 1)) * (max_val - min_val)
            return expected_value

        i = 0
        if generation_config.confidence_based_adaptive_unmasking:
            num_confidents_dict = {}
        while i < steps:
            mask_index = (x == mask_token_id)
            num_mask_token = mask_index.sum().item()
            if num_mask_token == 0:
                break  # All tokens unmasked

            logits = self(x, attention_mask, tok_idx, save_cache=False).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            logits = generation_logits_hook_func(i, x, logits)
            mask_logits = logits[mask_index]
            t, s = timesteps[i], timesteps[i + 1]

            if generation_config.early_stop:
                positions = torch.nonzero(mask_index[0], as_tuple=False).squeeze(-1)


            if alg == 'maskgit_plus':
                confidence, x0, mask_probs = sample_tokens(
                    mask_logits, 
                    temperature=temperature, 
                    top_p=top_p, 
                    top_k=top_k,
                    return_probs=True,
                    )
            elif alg == 'topk_margin':
                confidence, x0, mask_probs = sample_tokens(
                    mask_logits, 
                    temperature=temperature, 
                    top_p=top_p, 
                    top_k=top_k, 
                    margin_confidence=True,
                    return_probs=True,
                    
                    )
            elif alg == 'entropy':
                confidence, x0, mask_probs = sample_tokens(
                    mask_logits, 
                    temperature, 
                    top_p=top_p, 
                    top_k=top_k, 
                    neg_entropy=True,
                    return_probs=True,

                    )
            elif alg == 'position_weighted':
                # Sample tokens normally
                confidence, x0 = sample_tokens(
                    mask_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                # TODO: fix this
                # Apply position-based weights to confidence
                # Get positions of masked tokens
                positions = torch.nonzero(mask_index, as_tuple=False).squeeze(-1)
                # Calculate weights: higher for lower positions (more left)
                # Normalize positions to [0, 1] range and invert (1 - pos) to give higher weights to lower positions
                max_pos = positions.max().item() if positions.numel() > 0 else 1
                pos_weights = 1.0 - (positions.float() / max_pos)
                # Apply weights to confidence
                confidence = confidence * pos_weights

            else:
                raise RuntimeError(f"Unknown alg: {alg}")

            if torch.isnan(confidence).any():
                print("⚠️ Some confidence values are NaN")
                import pdb; pdb.set_trace()

            if enable_confidence_based:
                # --------- Confidence-based adaptive unmasking ----------
                # Compute positions of masked tokens
                positions = torch.nonzero(mask_index[0], as_tuple=False).squeeze(-1)  # shape: [num_masked_tokens]

                # Optionally modify confidence based on decay_algorithm
                if generation_config.decay_algorithm != "none":
                    if generation_config.decay_algorithm == "exponential":
                        alpha = generation_config.decay_params.get("alpha", 1.0)
                        pos_weights = torch.exp(-alpha * positions / x.shape[1])
                        confidence = confidence * pos_weights
                    elif generation_config.decay_algorithm == "exponential_on_raw_logits":
                        pass
                    elif generation_config.decay_algorithm == "linear":
                        alpha = generation_config.decay_params.get("alpha", 1.0)
                        pos_weights = 1.0 - alpha * positions / x.shape[1]
                        pos_weights = pos_weights.clamp(min=0.0)
                        confidence = confidence * pos_weights
                    elif generation_config.decay_algorithm == "exp_unmasked_ratio":
                        alpha = generation_config.decay_params.get("alpha", 1.0)
                        gamma = generation_config.decay_params.get("gamma", 1.0)
                        gen_len = x.shape[1] - prompt_len
                        relative_positions = positions - prompt_len
                        unmasked_ratio = 1 - (num_mask_token / gen_len)
                        unmasked_ratio = torch.tensor(unmasked_ratio, device=x.device)
                        alpha_t = alpha * torch.exp(-gamma * unmasked_ratio)
                        pos_weights = torch.exp(-alpha_t * relative_positions / gen_len)
                        confidence = confidence * pos_weights
                    elif generation_config.decay_algorithm == "exp_unmasked_ratio_v2":
                        alpha = generation_config.decay_params.get("alpha", 1.0)
                        gamma = generation_config.decay_params.get("gamma", 1.0)
                        full_len = x.shape[1]
                        unmasked_ratio = 1 - (num_mask_token / full_len)
                        unmasked_ratio = torch.tensor(unmasked_ratio, device=x.device)
                        alpha_t = alpha * torch.exp(-gamma * unmasked_ratio)
                        pos_weights = torch.exp(-alpha_t * positions / full_len)
                        confidence = confidence * pos_weights
                    elif generation_config.decay_algorithm == "kl_divergence":
                        print(f"KL divergence: {kl_divergence}", flush=True)
                        pass

                # Select tokens to unmask based on decay algorithm
                if generation_config.decay_algorithm == "closest_confident_group":
                    # Get parameters from decay_params
                    max_scan_tokens = generation_config.decay_params.get("max_scan_tokens", 10)
                    confidence_threshold = generation_config.decay_params.get("confidence_threshold", 0.999)


                    num_to_scan = min(max_scan_tokens, len(confidence))
                    group = []
                    for idx in range(num_to_scan):
                        if confidence[idx] > confidence_threshold:
                            group.append(idx)
                        else:
                            # if len(group) > 0:
                            #     break  # Stop after first group
                            break

                    if len(group) > 0:
                        confident_indices = torch.tensor(group, device=confidence.device)
                    else:
                        _, confident_indices = torch.topk(confidence, k=1)

                    num_confident = confident_indices.numel()
                    number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
                    num_to_unmask = max(num_confident, number_transfer_tokens)

                else:
                    # Default adaptive unmasking
                    confidence_threshold = expected_mth_largest(confidence.min().item(), confidence.max().item(), num_mask_token, m=2)
                    confident_indices = torch.nonzero(confidence > confidence_threshold, as_tuple=False).squeeze(-1)
                    num_confident = confident_indices.numel()
                    number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
                    num_to_unmask = max(num_confident, number_transfer_tokens)

                # Actually perform the unmasking
                if num_to_unmask > 0:
                    if num_confident > number_transfer_tokens:
                        num_confidents_dict[i] = num_confident
                        selected_indices = confident_indices
                    else:
                        _, selected_indices = torch.topk(confidence, num_to_unmask)

                    x0_ = torch.zeros_like(x0, device=self.device) + mask_token_id
                    x0_[selected_indices] = x0[selected_indices]
                    x[mask_index] = x0_

                    if generation_config.early_stop:
                        last_pos = positions[selected_indices].max().item()
                        last_clean_token_position = max(last_clean_token_position, last_pos)
                        last_clean_token = x[0, last_clean_token_position].item()

                    tokens_unmasked = num_to_unmask
                    delta_step = max(1, int(tokens_unmasked // avg_tokens_per_step))
                    i += delta_step



            else:
                # --------- Standard step-based unmasking ----------
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token

                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                    else:
                        confidence = confidence / alg_temp
                        confidence = F.softmax(confidence, dim=-1)
                        transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
                    x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + mask_token_id
                    x0_[transfer_index] = x0[transfer_index].clone()
                    x[mask_index] = x0_

                    if generation_config.early_stop:
                        last_pos = positions[transfer_index].max().item()
                        last_clean_token_position = max(last_clean_token_position, last_pos)
                        last_clean_token = x[0, last_clean_token_position].item()


            x = generation_tokens_hook_func(i, x, logits)
            if histories is not None:
                histories.append(x.clone())

            if generation_config.early_stop:
                if self._check_early_stop(x, last_clean_token_position, last_clean_token, generation_config):
                    print(f"Early stopping at step {i}", flush=True)
                    mask_index_remaining = (x == generation_config.mask_token_id)
                    x[mask_index_remaining] = generation_config.eos_token_id
                    break

            i += 1  # Normal step increment
        if generation_config.confidence_based_adaptive_unmasking:
            print(f"num_confidents_dict: {num_confidents_dict}", flush=True)
        return x

    def _sample_block_diffusion_v2(
        self, x, attention_mask, tok_idx,
        generation_config,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        histories
    ):

        BLUE = "\033[94m"   
        RESET = "\033[0m"
        print(f"Inside dream-flash block diffusion v2", flush=True)

        steps, eps, alg, alg_temp, temperature, top_p, top_k, mask_token_id = (
            generation_config.steps,
            generation_config.eps,
            generation_config.alg,
            generation_config.alg_temp,
            generation_config.temperature,
            generation_config.top_p,
            generation_config.top_k,
            generation_config.mask_token_id
        )
        block_size = generation_config.block_size
        use_full_query_attn = generation_config.use_full_query_attn
        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # add a flag to use sliding window caching
        sliding_window_caching = generation_config.sliding_window_caching
        sliding_window_size = generation_config.sliding_window_size

        prompt_len = (x != mask_token_id).sum(dim=-1).max().item()

        # num_blocks = steps // block_size
        # TODO: fix this to -- num_blocks = max new tokens // block_size
        if not sliding_window_caching:
            num_blocks = generation_config.max_new_tokens // block_size
            # For now, enforce the max_new_tokens to be divisible by block_size
            assert generation_config.max_new_tokens % block_size == 0, "max_new_tokens must be divisible by block_size"
            num_steps_per_block = steps // num_blocks
            # For now, enforce the steps to be divisible by num_blocks
            assert steps % num_blocks == 0, "steps must be divisible by num_blocks"

        avg_tokens_per_step = generation_config.max_new_tokens / steps
        

        # Log if using sliding window caching
        if sliding_window_caching:
            BLUE = "\033[94m"   
            RESET = "\033[0m"
            print(f"{BLUE}Using sliding window caching with size {sliding_window_size}{RESET}", flush=True)


        # NEW FEATURE: early stopping
        # Keep track of the last clean token position and token
        last_clean_token_position = -1 
        last_clean_token = None

        if sliding_window_caching:
            diffusion_step = -1
            bsz = x.shape[0]
            # while there's any masked token in the sequence, keep generating
            while (x == mask_token_id).any():
                diffusion_step += 1
                masked_tok_positions = torch.nonzero(x == mask_token_id, as_tuple=False).squeeze(-1)
                # find the first masked token 
                first_masked_tok_pos = masked_tok_positions[0].item()
                sliding_window_start = max(0, first_masked_tok_pos-sliding_window_size)
                # sliding_window_end = min(L, first_masked_tok_pos+sliding_window_size)
                sliding_window_end = L # Do end of sequence for now

                x_curr_block = x[:, sliding_window_start:sliding_window_end]
                # mask_pos_curr_block = (x_curr_block == mask_token_id).nonzero(as_tuple=True)[0]
                mask_index_curr_block = (x_curr_block == mask_token_id)

                if diffusion_step == 0:
                    # At the first global step 0, reset the block cache
                    for layer_idx in range(self.config.num_hidden_layers):
                        self.model.layers[layer_idx].reset_block_cache(bsz, L, block_size)
                
                    # Model forward
                    logits = self(x, None, tok_idx if tok_idx is not None else None,
                                use_block_diffusion=True,
                                use_full_query_attn=False,
                                max_length=generation_config.max_length,
                                block_size=block_size, # this is actually not really used by the model forward pass
                                save_cache=True,
                                clean_idx=sliding_window_start).logits
                    
                    # all the mask positions in the full sequence of length L
                    # x_curr_block = x[:, sliding_window_start:sliding_window_end]
                    # gen_mask_pos = (x_curr_block == mask_token_id).nonzero(as_tuple=True)[0]
                    
                    # mask_pos w.r.t. to the full sequence of length L
                    # mask_pos = gen_mask_pos + sliding_window_start

                    # only the mask positions in the current block
                    # length: sliding_window_end - sliding_window_start
                    # mask_logits = logits[mask_pos]

                    logits_block = logits[:, sliding_window_start:sliding_window_end]
                    # # shift the logits to the right by 1, length: L
                    # logits_block = torch.cat([logits_block[:, :1], logits_block[:, :-1]], 1)

                else:
                    # for later steps > step 0
                    gen_seq = x[:, sliding_window_start:sliding_window_end]
                    gen_tok_idx = tok_idx[:, sliding_window_start:sliding_window_end] if tok_idx is not None else None

                    logits = self(
                        gen_seq, 
                        gen_tok_idx, 
                        None,
                        use_block_diffusion=True,
                        use_full_query_attn=False,
                        max_length=generation_config.max_length,
                        block_size=block_size, # not really used, can ignore
                        save_cache=True,
                        clean_idx=sliding_window_start).logits

                
                    # gen_mask_pos = (gen_seq[0] == mask_token_id).nonzero(as_tuple=True)[0]
                    
                    # # mask_pos w.r.t. the full sequence of length L
                    # mask_pos = gen_mask_pos + sliding_window_start

                    logits_block = logits
                    # # mask_logits = logits[gen_mask_pos]
                    # logits_block = torch.cat([logits_block[:, :1], logits_block[:, :-1]], 1)
                    # mask_logits = logits[mask_index_curr_block]

                # shift the logits to the right by 1, length: block_size
                logits_block = torch.cat([logits_block[:, :1], logits_block[:, :-1]], dim=1)
                mask_logits = logits_block[mask_index_curr_block]

                t, s = timesteps[i], timesteps[i + 1]

                if alg == 'origin':
                    # p_transfer = 1 - s / t if step_idx < block_size - 1 else 1
                    p_transfer = 1 - s / t if step_idx < num_steps_per_block - 1 else 1
                    # x0 = torch.zeros_like(x_curr_block[mask_pos_curr_block], device=self.device, dtype=torch.long) + mask_token_id
                    x0 = torch.zeros_like(x_curr_block[mask_index_curr_block], device=self.device, dtype=torch.long) + mask_token_id

                    transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                    _, x0[transfer_index_t_s] = sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                    # x_curr_block[mask_pos_curr_block] = x0.clone()
                    x_curr_block[mask_index_curr_block] = x0.clone()
                    x[:, block_start:block_end] = x_curr_block.clone()

                    if generation_config.early_stop:
                        if transfer_index_t_s.any():
                            # last_pos = torch.nonzero(block_mask_index)[transfer_index_t_s].max().item()
                            # last_pos = torch.nonzero(mask_pos_curr_block)[transfer_index_t_s].max().item()
                            last_pos = torch.nonzero(mask_index_curr_block)[transfer_index_t_s].max().item()
                            last_clean_token_position = max(last_clean_token_position, block_start + last_pos)
                            last_clean_token = x[0, last_clean_token_position].item()

                else:
                    if alg == 'maskgit_plus':
                        confidence, x0 = sample_tokens(
                            mask_logits, 
                            temperature=temperature, 
                            top_p=top_p, 
                            top_k=top_k,
                            )
                    elif alg == 'topk_margin':
                        confidence, x0 = sample_tokens(
                            mask_logits, 
                            temperature=temperature, 
                            top_p=top_p, 
                            top_k=top_k, 
                            margin_confidence=True,                            
                            )
                    elif alg == 'entropy':
                        confidence, x0 = sample_tokens(
                            mask_logits, 
                            temperature, 
                            top_p=top_p, 
                            top_k=top_k, 
                            neg_entropy=True,
                            )
                    # TODO: Need to check if this is correct
                    elif alg == 'position_weighted':
                        # Sample tokens normally
                        confidence, x0 = sample_tokens(
                            mask_logits,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                        )
                        # Apply position-based weights to confidence
                        # Get positions of masked tokens
                        # positions = torch.nonzero(mask_pos_curr_block, as_tuple=False).squeeze(-1)
                        positions = torch.nonzero(mask_index_curr_block, as_tuple=False).squeeze(-1)
                        # Calculate weights: higher for lower positions (more left)
                        # Normalize positions to [0, 1] range and invert (1 - pos) to give higher weights to lower positions
                        max_pos = positions.max().item() if positions.numel() > 0 else 1
                        pos_weights = 1.0 - (positions.float() / max_pos)
                        # Apply weights to confidence
                        confidence = confidence * pos_weights
                    else:
                        raise RuntimeError(f"Unknown alg: {alg}")

                # num_mask_token = mask_pos_curr_block.numel()
                # assert mask_pos_curr_block.numel() == mask_pos_curr_block.sum(), "mask_pos_curr_block should only contain 1s"
                num_mask_token = mask_index_curr_block.sum()


                total_num_mask_token = num_mask_token + (num_blocks - block_idx - 1) * block_size
                # number_transfer_tokens = int(total_num_mask_token * (1 - s / t)) if step_idx < block_size - 1 else num_mask_token
                number_transfer_tokens = int(total_num_mask_token * (1 - s / t)) \
                    if step_idx < num_steps_per_block - 1 \
                        else num_mask_token # unmask all masked tokens at last step of the block

                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                    else:
                        confidence = confidence / alg_temp
                        confidence = F.softmax(confidence, dim=-1)
                        transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
                    x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + mask_token_id
                    x0_[transfer_index] = x0[transfer_index].clone()
                    # x_partial[block_mask_index] = x0_.clone()
                    # x_curr_block[mask_pos_curr_block] = x0_.clone()
                    x_curr_block[mask_index_curr_block] = x0_.clone()
                    # x[:, block_start:block_end] = x_partial.clone()
                    x[:, block_start:block_end] = x_curr_block.clone()

                    if generation_config.early_stop:
                        # last_pos = torch.nonzero(block_mask_index)[transfer_index].max().item()
                        # last_pos = torch.nonzero(mask_pos_curr_block)[transfer_index].max().item()
                        last_pos = torch.nonzero(mask_index_curr_block)[transfer_index].max().item()
                        last_clean_token_position = max(last_clean_token_position, block_start + last_pos)
                        last_clean_token = x[0, last_clean_token_position].item()


                x = generation_tokens_hook_func(i, x, logits)

                if histories is not None:
                    histories.append(x.clone())

                # Check for EOS in newly generated tokens and handle early stopping
                if generation_config.early_stop:
                    # Check if any EOS token was generated in this iteration
                    newly_generated = (x0_ != mask_token_id)
                    if newly_generated.any():
                        newly_generated_tokens = x0_[newly_generated]
                        if generation_config.eos_token_id in newly_generated_tokens:
                            # Find the first EOS position in the full sequence
                            eos_positions = torch.nonzero(x[0] == generation_config.eos_token_id).squeeze(-1)
                            if eos_positions.numel() > 0:
                                first_eos_pos = eos_positions[0].item()
                                x[0, first_eos_pos:] = generation_config.eos_token_id
                                return x
                    
                    # If no masked tokens left, early stop
                    if not (x == mask_token_id).any():
                        return x

                if generation_config.early_stop:
                    if self._check_early_stop(x, last_clean_token_position, last_clean_token, generation_config):
                        print(f"Early stopping at block {block_idx}, step {step_idx}, i: {i}", flush=True)
                        mask_index_remaining = (x == generation_config.mask_token_id)
                        x[mask_index_remaining] = generation_config.eos_token_id
                        return x
                
        else:       
            # Use regular block(-cached) diffusion
            for block_idx in range(num_blocks):
                # Iterating over blocks
                block_start = prompt_len + block_idx * block_size
                block_end = prompt_len + (block_idx + 1) * block_size

                # for step_idx in range(block_size):
                for step_idx in range(num_steps_per_block):

                    # i = block_idx * block_size + step_idx
                    # fix this to -- i = block_idx * num_steps_per_block + step_idx
                    i = block_idx * num_steps_per_block + step_idx # i is the global step index


                    # x_partial = x[:, block_start:block_end].clone()
                    x_curr_block = x[:, block_start:block_end]
                    # mask_pos_curr_block = (x_curr_block == mask_token_id).nonzero(as_tuple=True)[0]
                    mask_index_curr_block = (x_curr_block == mask_token_id)

                    # save_cache = step_idx == block_size - 1 # save cache for the last step of the block
                    # last step of the block
                    save_cache = step_idx == num_steps_per_block - 1

                    # Another attempt to recover accuracy
                    bsz = x.shape[0]
                    L = generation_config.max_length
                    end_of_sentence = L 

                    if i == 0:
                        # At the first global step 0, reset the block cache
                        for layer_idx in range(self.config.num_hidden_layers):
                            self.model.layers[layer_idx].reset_block_cache(bsz, L, block_size)

                        # Pass in the entire input x, and the block caching will be used
                        # We get the logits of corresponding block 
                        logits = self(x, None, tok_idx if tok_idx is not None else None,
                                    use_block_diffusion=True,
                                    use_full_query_attn=False,
                                    max_length=generation_config.max_length,
                                    block_size=block_size,
                                    save_cache=True,
                                    # clean_idx=prompt_len).logits[:, block_start:block_end]
                                    clean_idx=block_start).logits
                        
                        # logits size: batch_size, L, vocab_size
                        logits_block = logits[:, block_start:block_end]

                    else:        
                        # At later steps > step 0
                        gen_seq = x[:, block_start:]
                        gen_tok_idx = tok_idx[:, block_start:] if tok_idx is not None else None
                        logits = self(gen_seq, 
                                    gen_tok_idx, 
                                    None,
                                    use_block_diffusion=True,
                                    use_full_query_attn=False,
                                    max_length=generation_config.max_length,
                                    block_size=block_size,
                                    save_cache=save_cache,
                                    clean_idx=block_end).logits # output length: block start until the end of the sequence
                    
                        logits_block = logits[:, :block_size]
        
                        # shift the logits to the right by 1, length: block_size
                        logits_block = torch.cat([logits_block[:, :1], logits_block[:, :-1]], dim=1)
                        mask_logits = logits_block[mask_index_curr_block]

                        t, s = timesteps[i], timesteps[i + 1]

                        if alg == 'origin':
                            # p_transfer = 1 - s / t if step_idx < block_size - 1 else 1
                            p_transfer = 1 - s / t if step_idx < num_steps_per_block - 1 else 1
                            # x0 = torch.zeros_like(x_curr_block[mask_pos_curr_block], device=self.device, dtype=torch.long) + mask_token_id
                            x0 = torch.zeros_like(x_curr_block[mask_index_curr_block], device=self.device, dtype=torch.long) + mask_token_id

                            transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                            _, x0[transfer_index_t_s] = sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                            # x_curr_block[mask_pos_curr_block] = x0.clone()
                            x_curr_block[mask_index_curr_block] = x0.clone()
                            x[:, block_start:block_end] = x_curr_block.clone()

                            if generation_config.early_stop:
                                if transfer_index_t_s.any():
                                    # last_pos = torch.nonzero(block_mask_index)[transfer_index_t_s].max().item()
                                    # last_pos = torch.nonzero(mask_pos_curr_block)[transfer_index_t_s].max().item()
                                    last_pos = torch.nonzero(mask_index_curr_block)[transfer_index_t_s].max().item()
                                    last_clean_token_position = max(last_clean_token_position, block_start + last_pos)
                                    last_clean_token = x[0, last_clean_token_position].item()

                        else:
                            if alg == 'maskgit_plus':
                                confidence, x0 = sample_tokens(
                                    mask_logits, 
                                    temperature=temperature, 
                                    top_p=top_p, 
                                    top_k=top_k,
                                    )
                            elif alg == 'topk_margin':
                                confidence, x0 = sample_tokens(
                                    mask_logits, 
                                    temperature=temperature, 
                                    top_p=top_p, 
                                    top_k=top_k, 
                                    margin_confidence=True,                            
                                    )
                            elif alg == 'entropy':
                                confidence, x0 = sample_tokens(
                                    mask_logits, 
                                    temperature, 
                                    top_p=top_p, 
                                    top_k=top_k, 
                                    neg_entropy=True,
                                    )
                            # TODO: Need to check if this is correct
                            elif alg == 'position_weighted':
                                # Sample tokens normally
                                confidence, x0 = sample_tokens(
                                    mask_logits,
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=top_k,
                                )
                                # Apply position-based weights to confidence
                                # Get positions of masked tokens
                                # positions = torch.nonzero(mask_pos_curr_block, as_tuple=False).squeeze(-1)
                                positions = torch.nonzero(mask_index_curr_block, as_tuple=False).squeeze(-1)
                                # Calculate weights: higher for lower positions (more left)
                                # Normalize positions to [0, 1] range and invert (1 - pos) to give higher weights to lower positions
                                max_pos = positions.max().item() if positions.numel() > 0 else 1
                                pos_weights = 1.0 - (positions.float() / max_pos)
                                # Apply weights to confidence
                                confidence = confidence * pos_weights
                            else:
                                raise RuntimeError(f"Unknown alg: {alg}")

                        # num_mask_token = mask_pos_curr_block.numel()
                        # assert mask_pos_curr_block.numel() == mask_pos_curr_block.sum(), "mask_pos_curr_block should only contain 1s"
                        num_mask_token = mask_index_curr_block.sum()


                        total_num_mask_token = num_mask_token + (num_blocks - block_idx - 1) * block_size
                        # number_transfer_tokens = int(total_num_mask_token * (1 - s / t)) if step_idx < block_size - 1 else num_mask_token
                        number_transfer_tokens = int(total_num_mask_token * (1 - s / t)) \
                            if step_idx < num_steps_per_block - 1 \
                                else num_mask_token # unmask all masked tokens at last step of the block

                        if number_transfer_tokens > 0:
                            if alg_temp is None or alg_temp == 0:
                                _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                            else:
                                confidence = confidence / alg_temp
                                confidence = F.softmax(confidence, dim=-1)
                                transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
                            x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + mask_token_id
                            x0_[transfer_index] = x0[transfer_index].clone()
                            # x_partial[block_mask_index] = x0_.clone()
                            # x_curr_block[mask_pos_curr_block] = x0_.clone()
                            x_curr_block[mask_index_curr_block] = x0_.clone()
                            # x[:, block_start:block_end] = x_partial.clone()
                            x[:, block_start:block_end] = x_curr_block.clone()

                            if generation_config.early_stop:
                                # last_pos = torch.nonzero(block_mask_index)[transfer_index].max().item()
                                # last_pos = torch.nonzero(mask_pos_curr_block)[transfer_index].max().item()
                                last_pos = torch.nonzero(mask_index_curr_block)[transfer_index].max().item()
                                last_clean_token_position = max(last_clean_token_position, block_start + last_pos)
                                last_clean_token = x[0, last_clean_token_position].item()


                        x = generation_tokens_hook_func(i, x, logits)

                        if histories is not None:
                            histories.append(x.clone())

                        if generation_config.early_stop:
                            if self._check_early_stop(x, last_clean_token_position, last_clean_token, generation_config):
                                print(f"Early stopping at block {block_idx}, step {step_idx}, i: {i}", flush=True)
                                mask_index_remaining = (x == generation_config.mask_token_id)
                                x[mask_index_remaining] = generation_config.eos_token_id
                                return x


        return x

    def _compute_block_logits(
        self,
        seq: torch.LongTensor,
        tok_idx: Optional[torch.LongTensor],
        clean_idx: int,
        block_size: int,
        save_cache: bool,
        use_full_query_attn: bool
    ) -> torch.FloatTensor:
        """
        Helper to run a forward pass and extract shifted block logits.
        """
        logits = self(
            seq,
            tok_idx,
            None,
            use_block_diffusion=True,
            use_full_query_attn=use_full_query_attn,
            max_length=self.generation_config.max_length,
            block_size=block_size,
            save_cache=save_cache,
            clean_idx=clean_idx
        ).logits
        if use_full_query_attn:
            block_logits = logits[:, :block_size]
        else:
            block_logits = logits[:, clean_idx:clean_idx + block_size]
        # shift for sampling
        return torch.cat([block_logits[:, :1], block_logits[:, :-1]], dim=1)


    def _sample_block_diffusion_v3(
        self,
        x: torch.LongTensor,
        attention_mask,
        tok_idx,
        generation_config,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        histories
    ) -> torch.LongTensor:
        """
        Sliding-window and blockwise diffusion with EOS-clamp after each iteration.
        """
        # Unpack
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        mask_id = generation_config.mask_token_id
        eos_id = generation_config.eos_token_id

        sliding = generation_config.sliding_window_caching
        window = generation_config.sliding_window_size
        block = generation_config.block_size
        max_new = generation_config.max_new_tokens
        max_len = generation_config.max_length

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
        prompt_len = (x != mask_id).sum(dim=-1).max().item()

        last_clean_pos = -1
        last_clean_tok = None

        if sliding:
            step_idx = 0
            while (x == mask_id).any():
                # window bounds
                first = torch.nonzero(x == mask_id)[0,1].item()
                start = max(0, first - window)
                seq = x[:, start:]
                tok = tok_idx[:, start:] if tok_idx is not None else None

                # get logits
                logits_blk = self._compute_block_logits(
                    seq, tok, start, block,
                    save_cache=(step_idx==0),
                    use_full_query_attn=False
                )
                mask_ix = (seq == mask_id)
                mask_logits = logits_blk[mask_ix]

                # sample
                t, s = timesteps[step_idx], timesteps[step_idx+1]
                if alg == 'origin':
                    p = 1 - s/t
                    x0 = torch.full_like(mask_logits, mask_id)
                    sel = torch.rand(x0.shape, device=x.device) < p
                    _, x0[sel] = sample_tokens(mask_logits[sel], temperature, top_p, top_k)
                elif alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p, top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p, top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p, top_k, neg_entropy=True)
                elif alg == 'position_weighted':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p, top_k)
                    # position weights
                    pos = torch.nonzero(mask_ix, as_tuple=False).squeeze(-1)
                    maxp = pos.max().item() if pos.numel()>0 else 1
                    weights = 1.0 - (pos.float()/maxp)
                    confidence = confidence * weights
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")

                # write back
                seq[mask_ix] = x0
                x[:, start:] = seq
                if histories is not None:
                    histories.append(x.clone())

                # clamp after EOS
                for b in range(x.size(0)):
                    eos_pos = (x[b]==eos_id).nonzero(as_tuple=False)
                    if eos_pos.numel()>0:
                        fe = eos_pos.min().item()
                        x[b, fe:] = eos_id

                # early stop
                if generation_config.early_stop and self._check_early_stop(x, last_clean_pos, last_clean_tok, generation_config):
                    x[x==mask_id] = eos_id
                    return x

                step_idx += 1
            return x

        else:
            # blockwise path
            n_blocks = max_new // block
            steps_pb = steps // n_blocks
            for bi in range(n_blocks):
                start = prompt_len + bi*block
                for si in range(steps_pb):
                    i = bi*steps_pb + si
                    seq = x[:, start:]
                    tok = tok_idx[:, start:] if tok_idx is not None else None

                    logits_blk = self._compute_block_logits(
                        seq, tok, start, block,
                        save_cache=(i==0),
                        use_full_query_attn=False
                    )
                    mask_ix = (seq == mask_id)
                    mask_logits = logits_blk[mask_ix]

                    t, s = timesteps[i], timesteps[i+1]
                    if alg == 'origin':
                        p = 1 - s/t
                        x0 = torch.full_like(mask_logits, mask_id)
                        sel = torch.rand(x0.shape, device=x.device) < p
                        _, x0[sel] = sample_tokens(mask_logits[sel], temperature, top_p, top_k)
                    elif alg == 'maskgit_plus':
                        confidence, x0 = sample_tokens(mask_logits, temperature, top_p, top_k)
                    elif alg == 'topk_margin':
                        confidence, x0 = sample_tokens(mask_logits, temperature, top_p, top_k, margin_confidence=True)
                    elif alg == 'entropy':
                        confidence, x0 = sample_tokens(mask_logits, temperature, top_p, top_k, neg_entropy=True)
                    elif alg == 'position_weighted':
                        confidence, x0 = sample_tokens(mask_logits, temperature, top_p, top_k)
                        pos = torch.nonzero(mask_ix, as_tuple=False).squeeze(-1)
                        maxp = pos.max().item() if pos.numel()>0 else 1
                        weights = 1.0 - (pos.float()/maxp)
                        confidence = confidence * weights
                    else:
                        raise RuntimeError(f"Unknown alg: {alg}")

                    seq[mask_ix] = x0
                    x[:, start:] = seq
                    if histories is not None:
                        histories.append(x.clone())

                    # clamp after EOS
                    for b in range(x.size(0)):
                        eos_pos = (x[b]==eos_id).nonzero(as_tuple=False)
                        if eos_pos.numel()>0:
                            fe = eos_pos.min().item()
                            x[b, fe:] = eos_id

                    if generation_config.early_stop and self._check_early_stop(x, last_clean_pos, last_clean_tok, generation_config):
                        x[x==mask_id] = eos_id
                        return x

            return x


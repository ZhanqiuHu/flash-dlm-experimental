# File: spec_diffusion_utils.py
"""
Speculative-diffusion decoding
(Dream draft model  +  Qwen-2.5 verifier)

All `[DEBUG …]` prints from your earlier version are kept intact.
"""

from __future__ import annotations
import sys, warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Union, List
from enum import Enum, auto

import torch
from torch import Tensor
from torch.nn import functional as F

from transformers import (
    __version__, GenerationConfig
)
from transformers.utils          import ModelOutput, logging
from transformers.generation.utils import _crop_past_key_values   # KV-trimmer
from transformers.cache_utils      import DynamicCache           # class wrapper

logger = logging.get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Dataclasses
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SpecDiffusionOutput(ModelOutput):
    sequences: torch.LongTensor
    history  : Optional[Tuple[torch.FloatTensor]] = None
    verification_stats: Optional[dict] = None  # Track verification metrics


class SpecDiffusionConfig(GenerationConfig):
    """
    Extra knobs (everything else falls back to HF GenerationConfig)
    """
    def __init__(self, **kw):
        # sampling
        self.temperature            = kw.pop("temperature", 0.0)
        self.top_p : Optional[float]= kw.pop("top_p", None)
        self.top_k : Optional[int]  = kw.pop("top_k", None)

        # length
        self.max_length             = kw.pop("max_length")
        self.max_new_tokens         = kw.pop("max_new_tokens", None)

        # special tokens
        self.mask_token_id          = kw.pop("mask_token_id")
        self.eos_token_id           = kw.pop("eos_token_id")

        # early-stop
        self.early_stop             = kw.pop("early_stop", True)
        self.early_stop_consecutive = kw.pop("early_stop_consecutive", 1)

        # stop on dream eos
        self.stop_on_dream_eos      = kw.pop("stop_on_dream_eos", False)  # New flag to stop on Dream model's EOS
        print(f"Stop on dream eos: {self.stop_on_dream_eos}")

        # verifier batching
        self.verify_batch_size      = kw.pop("verify_batch_size", 32)

        # verification strategy
        self.sampling_strategy      = kw.pop("sampling_strategy", "deterministic")
        self.confidence_threshold   = kw.pop("confidence_threshold", 0.1)
        self.acceptance_top_k       = kw.pop("acceptance_top_k", 5)  # Add top-k parameter for acceptance

        # sliding window caching
        self.sliding_window_caching = kw.pop("sliding_window_caching", False)  # Default to False
        self.sliding_window_size    = kw.pop("sliding_window_size", 128)

        # misc
        self.return_dict_in_generate= kw.pop("return_dict_in_generate", False)
        self.output_history         = kw.pop("output_history", False)

        # ignored legacy keys
        kw.pop("steps", None); kw.pop("eps", None)

        self.transformers_version   = kw.pop("transformers_version", __version__)
        if kw:
            logger.warning(f"Ignored unknown config keys: {list(kw.keys())}")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Token-mapping helpers
# ──────────────────────────────────────────────────────────────────────────────
class TokenMapper:
    """Handles token ID conversion between Dream and Qwen tokenizers."""
    def __init__(self, dream_tok, qwen_tok):
        self.dtok = dream_tok
        self.qtok = qwen_tok
        
        # Define equivalent special tokens
        self.equivalent_tokens = {
            # Dream token ID -> Qwen token ID
            151643: 151645,  # <|endoftext|> -> <|im_end|>
            151645: 151643,  # <|im_end|> -> <|endoftext|>
        }
        
        # Check tokenizer compatibility once during initialization
        try:
            # Get base vocabularies (excluding special tokens)
            dream_vocab = set(dream_tok.get_vocab().keys())
            qwen_vocab = set(qwen_tok.get_vocab().keys())
            
            # Get special tokens - handle both single strings and lists
            dream_special = set()
            for token in dream_tok.special_tokens_map.values():
                if isinstance(token, list):
                    dream_special.update(token)
                else:
                    dream_special.add(token)
                    
            qwen_special = set()
            for token in qwen_tok.special_tokens_map.values():
                if isinstance(token, list):
                    qwen_special.update(token)
                else:
                    qwen_special.add(token)
            
            # Check if base vocabularies are identical, ignoring mask token
            dream_diff = dream_vocab - qwen_vocab
            qwen_diff = qwen_vocab - dream_vocab
            
            # Only allow mask token and beginoftext token as differences
            allowed_dream_diff = {'<|mask|>', '<|beginoftext|>'}
            allowed_qwen_diff = set()
            
            # Check if vocabularies are compatible (only differ by allowed tokens)
            vocab_match = dream_diff.issubset(allowed_dream_diff) and qwen_diff.issubset(allowed_qwen_diff)
            
            # Check if they have the same vocab size
            vocab_size_match = dream_tok.vocab_size == qwen_tok.vocab_size
            
            # Only skip conversion if both conditions are met
            self.skip_conversion = vocab_match and vocab_size_match
            
            if self.skip_conversion:
                print("Skipping token conversion - tokenizers are compatible")
            else:
                print("Tokenizers are not compatible - will use conversion")
        except Exception as e:
            print(f"Error checking tokenizer compatibility: {e}")
            self.skip_conversion = False
            print("Will use conversion")
        

    def dream_to_verifier_tensor(self, ids: Tensor, dev: torch.device) -> Tensor:
        """Convert Dream token IDs to Qwen token IDs."""
        if self.skip_conversion:
            return ids.to(dev)
        
        txt = self.dtok.decode(ids.tolist(), skip_special_tokens=False)
        qids = self.qtok(txt, add_special_tokens=False).input_ids
        return torch.tensor(qids, device=dev)

    def dream_to_verifier_id(self, did: int) -> int:
        """Convert single Dream token ID to Qwen token ID."""
        if self.skip_conversion:
            return did
        
        return self.qtok(self.dtok.decode([did], skip_special_tokens=False),
                        add_special_tokens=False).input_ids[0]

    def dream_to_verifier_ids(self, dids: Tensor) -> Tensor:
        """Convert batch of Dream token IDs to Qwen token IDs."""
        if self.skip_conversion:
            return dids
        
        # Convert all IDs at once
        txt = self.dtok.decode(dids.tolist(), skip_special_tokens=False)
        qids = self.qtok(txt, add_special_tokens=False).input_ids
        return torch.tensor(qids, device=dids.device)

    def verifier_to_dream_id(self, qid: int) -> int:
        """Convert single Qwen token ID to Dream token ID."""
        if self.skip_conversion:
            return qid
        
        return self.dtok(self.qtok.decode([qid], skip_special_tokens=False),
                        add_special_tokens=False).input_ids[0]

    def verifier_to_dream_ids(self, qids: Tensor) -> Tensor:
        """Convert batch of Qwen token IDs to Dream token IDs."""
        if self.skip_conversion:
            return qids
        
        # Convert all IDs at once
        txt = self.qtok.decode(qids.tolist(), skip_special_tokens=False)
        dids = self.dtok(txt, add_special_tokens=False).input_ids
        return torch.tensor(dids, device=qids.device)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  AR verifier with KV-cache
# ──────────────────────────────────────────────────────────────────────────────
class SamplingStrategy(Enum):
    DETERMINISTIC = auto()
    DYNAMIC_THRESHOLD = auto()
    AGGRESSIVE = auto()
    TOPK = auto()
    TOPK_RELATIVE = auto()  # New strategy that considers relative probabilities within top-k
    ORIGINAL = auto()

    @classmethod
    def from_string(cls, strategy_str: str) -> "SamplingStrategy":
        strategy_map = {
            "deterministic": cls.DETERMINISTIC,
            "dynamic_threshold": cls.DYNAMIC_THRESHOLD,
            "aggressive": cls.AGGRESSIVE,
            "topk": cls.TOPK,
            "topk_relative": cls.TOPK_RELATIVE,
            "original": cls.ORIGINAL,
        }
        return strategy_map[strategy_str.lower()]

class DeepSeekTokenMapper:
    """
    TokenMapper variant that uses text-level reconstruction to ensure
    semantic alignment between Dream and DeepSeek (non-identical tokenizers).
    """
    def __init__(self, dream_tok, verifier_tok):
        self.dtok = dream_tok
        self.vtok = verifier_tok

        self.skip_conversion = False
        print("Using DeepSeekTokenMapper: forcing text-level mapping due to tokenizer mismatch.")

        self.special_token_id_map = {
            # Qwen ID → DeepSeek ID
            151643: 151643,  # <|endoftext|> <-> <｜end▁of▁sentence｜>
            151644: 151644,  # <|im_start|>   <-> <｜User｜>
            151645: 151645,  # <|im_end|>     <-> <｜Assistant｜>
            151646: 151646,  # <|object_ref_start|> <-> <｜begin▁of▁sentence｜>
            151647: 151647,  # <|object_ref_end|>   <-> <|EOT|>
            151648: 151648,  # <|box_start|> <-> <think>
            151649: 151649,  # <|box_end|>   <-> </think>
        }
        self.inverse_special_token_id_map = {v: k for k, v in self.special_token_id_map.items()}

    def dream_to_verifier_tensor(self, ids: Tensor, dev: torch.device) -> Tensor:
        """
        Convert a tensor of Dream token IDs into a tensor of DeepSeek token IDs
        via intermediate decoded string (to preserve subword boundaries).
        """
        text = self.dtok.decode(ids.tolist(), skip_special_tokens=False)
        enc = self.vtok(text, add_special_tokens=False)
        qids = enc.input_ids
        if len(qids) == 0:
            raise ValueError("Token conversion resulted in empty verifier tokens.")
        return torch.tensor(qids, device=dev)

    # def dream_to_verifier_id(self, did: int) -> int:
    #     """
    #     Convert a single Dream token ID to a verifier token ID
    #     using decoded string re-tokenization.
    #     """
    #     text = self.dtok.decode([did], skip_special_tokens=False)
    #     qids = self.vtok(text, add_special_tokens=False).input_ids
    #     if not qids:
    #         raise ValueError(f"Failed to convert Dream token ID {did} → verifier ID via text: '{text}'")
    #     return qids[0]

    def dream_to_verifier_id(self, did: int) -> int:
        if did in self.inverse_special_token_id_map:
            return self.inverse_special_token_id_map[did]
        text = self.dtok.decode([did], skip_special_tokens=False)
        qids = self.vtok(text, add_special_tokens=False).input_ids
        if not qids:
            raise ValueError(f"Failed to convert Dream ID {did} → Verifier ID via text: '{text}'")
        return qids[0]
    
    def dream_to_verifier_ids(self, dids: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of Dream token IDs into DeepSeek token IDs
        by round‐trip decoding & re‐tokenizing as text.
        """
        # Decode the entire tensor to text
        text = self.dtok.decode(dids.tolist(), skip_special_tokens=False)
        # Re‐encode on the verifier tokenizer
        enc  = self.vtok(text, add_special_tokens=False)
        # Return a torch.Tensor of the new IDs on the correct device
        return torch.tensor(enc.input_ids, device=dids.device)


    def verifier_to_dream_id(self, qid: int) -> int:
        if qid in self.special_token_id_map:
            return self.special_token_id_map[qid]
        text = self.vtok.decode([qid], skip_special_tokens=False)
        dids = self.dtok(text, add_special_tokens=False).input_ids
        if not dids:
            raise ValueError(f"Failed to convert verifier ID {qid} → Dream ID via text: '{text}'")
        return dids[0]

    def verifier_to_dream_ids(self, qids: Tensor) -> Tensor:
        text = self.vtok.decode(qids.tolist(), skip_special_tokens=False)
        dids = self.dtok(text, add_special_tokens=False).input_ids
        if not dids:
            raise ValueError("Empty Dream ID sequence from verifier tokens.")
        return torch.tensor(dids, device=qids.device)

    def equal_semantics(self, dream_ids: List[int], verifier_ids: List[int]) -> bool:
        """
        Robust semantic check: whether two sequences decode to the same string.
        This version compares decoded strings from token *lists*, not single tokens.
        """
        dream_text = self.dtok.decode(dream_ids, skip_special_tokens=False).strip()
        verifier_text = self.vtok.decode(verifier_ids, skip_special_tokens=False).strip()
        return dream_text == verifier_text
    



class ARVerifier:
    """
    Thin wrapper around the AR verifier model (Qwen-2.5) that keeps its
    PastKeyValues cache and supports batched verification.
    """
    def __init__(self, ar_model, dream_tok, qwen_tok):
        self.model = ar_model
        # self.token_mapper = TokenMapper(dream_tok, qwen_tok)

        # Detect verifier model type (e.g., DeepSeek)
        model_name = getattr(ar_model.config, 'name_or_path', '').lower()
        if "deepseek" in model_name:
            print("[ARVerifier] Detected DeepSeek model. Using DeepSeekTokenMapper.")
            self.token_mapper = DeepSeekTokenMapper(dream_tok, qwen_tok)
        else:
            print("[ARVerifier] Using default TokenMapper.")
            self.token_mapper = TokenMapper(dream_tok, qwen_tok)

        
        self.cache_q : List[int] = []
        self.kv : DynamicCache | None = None
        self.dev = next(ar_model.parameters()).device
        self.last_logits : Optional[Tensor] = None
        self.verification_stats = None

    def reset(self):
        """Reset the verifier state for a new inference."""
        self.cache_q = []
        self.kv = None
        self.last_logits = None
        if self.verification_stats is not None:
            self.verification_stats['ar_model_calls'] = 0
            self.verification_stats['rejection_correction_calls'] = 0

    def _push(self, qids: Tensor) -> Tensor:
        """Feed `qids` to the verifier, updating its KV-cache."""
        # print(f"pushing {qids.shape} to verifier")
        out = self.model(qids, use_cache=True, past_key_values=self.kv)
        self.kv = out.past_key_values
        self.cache_q.extend(qids[0].tolist())
        self.last_logits = out.logits[0, -1].view(-1)  # Ensure 1D tensor
        if self.verification_stats is not None:
            self.verification_stats['rejection_correction_calls'] += 1
        return self.last_logits

    def _ensure_prefix(self, dream_prefix: Tensor):
        """Sync verifier's KV-cache with Dream prefix if they diverged."""
        # import pdb; pdb.set_trace()
        # Quick check if we need to do anything
        if len(self.cache_q) >= len(dream_prefix):
            return
            
        # Only convert the new tokens we need
        new_tokens = dream_prefix[len(self.cache_q):]
        if len(new_tokens) > 0:
            # Ensure new_tokens is on the correct device
            new_tokens = new_tokens.to(self.dev)
            q_new_tokens = self.token_mapper.dream_to_verifier_tensor(new_tokens, self.dev)
            logits = self._push(q_new_tokens.unsqueeze(0))
            self.last_logits = logits

        # Original version (commented out for reference):
        # # Use complete prefix for verification
        # tgt = self.token_mapper.dream_to_verifier_tensor(dream_prefix, self.dev)
        # common = 0
        # while (common < len(tgt) and common < len(self.cache_q)
        #        and tgt[common].item() == self.cache_q[common]):
        #     common += 1
        # if common < len(tgt):
        #     logits = self._push(tgt[common:].unsqueeze(0))
        #     self.last_logits = logits

    def _verify_deterministic(self, q_chunk: Tensor, probs_blk: Tensor) -> Tuple[List[int], int]:
        """Vectorized deterministic verification strategy."""
        # Get argmax for each position in batch
        selected_ids = torch.argmax(probs_blk, dim=-1)  # [batch_size]
        
        # Ensure we only compare up to the minimum size
        min_size = min(q_chunk[0].size(0), selected_ids.size(0))
        q_chunk_slice = q_chunk[0, :min_size]
        selected_ids_slice = selected_ids[:min_size]
        
        # Create mask for accepted tokens (where draft matches selected)
        accept_mask = (q_chunk_slice == selected_ids_slice)  # [min_size]
        
        # For accepted tokens, use draft tokens
        chosen_dream = []
        accept_len = 0
        for j, (is_accepted, q_id) in enumerate(zip(accept_mask, q_chunk_slice)):
            if is_accepted:
                chosen_dream.append(self.token_mapper.verifier_to_dream_id(q_id.item()))
                accept_len += 1
            else:
                # If not accepted, use selected token
                chosen_dream.append(self.token_mapper.verifier_to_dream_id(selected_ids_slice[j].item()))
                break
        
        return chosen_dream, accept_len

    def _verify_deterministic_sequential(self, q_chunk: Tensor, probs_blk: Tensor) -> Tuple[List[int], int]:
        """Sequential deterministic verification strategy that processes tokens one at a time."""
        chosen_dream = []
        accept_len = 0
        
        # Process each token sequentially
        for q_id, probs_i in zip(q_chunk[0], probs_blk):
            # Get argmax token for current position
            selected_id = probs_i.argmax().item()
            
            # Check if draft token matches verifier's choice
            if selected_id == q_id.item():
                chosen_dream.append(self.token_mapper.verifier_to_dream_id(q_id.item()))
                accept_len += 1
            else:
                # If not accepted, use selected token and stop
                chosen_dream.append(self.token_mapper.verifier_to_dream_id(selected_id))
                break
        
        return chosen_dream, accept_len

    def _verify_dynamic_threshold(self, q_chunk: Tensor, probs_blk: Tensor, confidence_threshold: float, rng: torch.Generator) -> Tuple[List[int], int]:
        """Original dynamic threshold verification strategy."""
        chosen_dream = []
        accept_len = 0
        for q_id, probs_i in zip(q_chunk[0], probs_blk):
            dream_prob = probs_i[q_id].item()
            
            # Accept if probability is above threshold
            if dream_prob > confidence_threshold:
                chosen_dream.append(self.token_mapper.verifier_to_dream_id(q_id.item()))
                accept_len += 1
            else:
                # Sample from Qwen's distribution excluding Dream's token
                probs_alt = probs_i.clone()
                probs_alt[q_id] = 0  # Zero out draft token
                probs_alt.div_(probs_alt.sum())
                alt_q = torch.multinomial(probs_alt, 1, generator=rng).item()
                chosen_dream.append(self.token_mapper.verifier_to_dream_id(alt_q))
                break
        return chosen_dream, accept_len

    def _verify_aggressive(self, q_chunk: Tensor, probs_blk: Tensor, rng: torch.Generator) -> Tuple[List[int], int]:
        """Original aggressive verification strategy."""
        chosen_dream = []
        accept_len = 0
        for q_id, probs_i in zip(q_chunk[0], probs_blk):
            # Get top-k tokens from verifier's distribution
            top_k = 5
            top_k_probs, top_k_ids = probs_i.topk(top_k)
            
            # Check if draft token is in top-k
            if q_id.item() in top_k_ids:
                chosen_dream.append(self.token_mapper.verifier_to_dream_id(q_id.item()))
                accept_len += 1
                continue
            
            # Check if verifier's probability for draft token is high enough
            draft_prob = probs_i[q_id]
            if draft_prob > 0.05:
                chosen_dream.append(self.token_mapper.verifier_to_dream_id(q_id.item()))
                accept_len += 1
                continue
            
            # Check if draft token is in top-p
            sorted_probs, sorted_ids = probs_i.sort(descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            top_p_mask = cumsum <= 0.95
            top_p_ids = sorted_ids[top_p_mask]
            
            if q_id.item() in top_p_ids:
                chosen_dream.append(self.token_mapper.verifier_to_dream_id(q_id.item()))
                accept_len += 1
                continue
            
            # If not accepted, sample from verifier's distribution
            probs_alt = probs_i.clone()
            probs_alt[q_id] = 0
            probs_alt.div_(probs_alt.sum())
            alt_q = torch.multinomial(probs_alt, 1, generator=rng).item()
            chosen_dream.append(self.token_mapper.verifier_to_dream_id(alt_q))
            break
        return chosen_dream, accept_len

    def _verify_topk(self, q_chunk: Tensor, probs_blk: Tensor, acceptance_top_k: int) -> Tuple[List[int], int]:
        """Original top-k verification strategy."""
        chosen_dream = []
        accept_len = 0
        for q_id, probs_i in zip(q_chunk[0], probs_blk):
            # Get top-k tokens from verifier's distribution
            top_k = acceptance_top_k
            top_k_probs, top_k_ids = probs_i.topk(top_k)
            
            # Check if draft token is in top-k
            if q_id.item() in top_k_ids:
                chosen_dream.append(self.token_mapper.verifier_to_dream_id(q_id.item()))
                accept_len += 1
                continue
            
            # If not accepted, use argmax instead of sampling
            probs_alt = probs_i.clone()
            probs_alt[q_id] = float('-inf')
            alt_q = torch.argmax(probs_alt).item()
            chosen_dream.append(self.token_mapper.verifier_to_dream_id(alt_q))
            break
        return chosen_dream, accept_len

    def _verify_original(self, q_chunk: Tensor, probs_blk: Tensor, rng: torch.Generator) -> Tuple[List[int], int]:
        """Original verification strategy."""
        chosen_dream = []
        accept_len = 0
        for q_id, probs_i in zip(q_chunk[0], probs_blk):
            if torch.rand((), device=self.dev, generator=rng) < probs_i[q_id]:
                chosen_dream.append(self.token_mapper.verifier_to_dream_id(q_id.item()))
                accept_len += 1
            else:
                probs_alt = probs_i.clone()
                probs_alt[q_id] = 0
                probs_alt.div_(probs_alt.sum())
                alt_q = torch.multinomial(probs_alt, 1, generator=rng).item()
                chosen_dream.append(self.token_mapper.verifier_to_dream_id(alt_q))
                break
        return chosen_dream, accept_len

    def _verify_topk_relative(
        self,
        q_chunk: Tensor,
        probs_blk: Tensor,
        top_k: int = 2,
        relative_threshold: float = 0.5,  # Accept if prob is within relative_threshold of top prob
    ) -> Tuple[List[int], int]:
        """Top-k relative verification strategy.
        
        Args:
            q_chunk: Draft tokens [1, batch_size]
            probs_blk: Verifier probabilities [batch_size, vocab_size]
            top_k: Number of top tokens to consider
            relative_threshold: Accept if token's prob is within this ratio of top prob
                               e.g., 0.5 means accept if prob >= 0.5 * top_prob
        """
        chosen_dream = []
        accept_len = 0
        
        # Get top-k probabilities and indices for each position
        topk_probs, topk_ids = probs_blk.topk(top_k, dim=-1)  # [batch_size, top_k]
        
        # Process each token
        for i, (q_id, probs_i, topk_probs_i, topk_ids_i) in enumerate(zip(q_chunk[0], probs_blk, topk_probs, topk_ids)):
            # Get probability of draft token
            draft_prob = probs_i[q_id].item()
            
            # Get top probability
            top_prob = topk_probs_i[0].item()
            
            # Check if draft token is in top-k and its probability is close enough to top
            if q_id in topk_ids_i and draft_prob >= relative_threshold * top_prob:
                chosen_dream.append(self.token_mapper.verifier_to_dream_id(q_id.item()))
                accept_len += 1
            else:
                # If not accepted, use top token
                chosen_dream.append(self.token_mapper.verifier_to_dream_id(topk_ids_i[0].item()))
                break
        
        return chosen_dream, accept_len

    def _verify_topk_relative_vectorized(
        self,
        q_chunk: Tensor,
        probs_blk: Tensor,
        top_k: int = 2,
        relative_threshold: float = 0.5,  # Accept if prob is within relative_threshold of top prob
    ) -> Tuple[List[int], int]:
        """Vectorized version of top-k relative verification strategy.
        
        Args:
            q_chunk: Draft tokens [1, batch_size]
            probs_blk: Verifier probabilities [batch_size, vocab_size]
            top_k: Number of top tokens to consider
            relative_threshold: Accept if token's prob is within this ratio of top prob
                               e.g., 0.5 means accept if prob >= 0.5 * top_prob
        """
        # Get top-k probabilities and indices for each position
        topk_probs, topk_ids = probs_blk.topk(top_k, dim=-1)  # [batch_size, top_k]
        
        # Get probabilities of draft tokens
        draft_probs = torch.gather(probs_blk[:q_chunk[0].size(0)], 1, q_chunk[0].unsqueeze(1))  # [batch_size, 1]
        
        # Create mask for tokens in top-k
        in_topk = (q_chunk[0].unsqueeze(1) == topk_ids[:q_chunk[0].size(0)]).any(dim=1)  # [batch_size]
        
        # Create mask for tokens meeting relative threshold
        meets_threshold = draft_probs.squeeze(1) >= relative_threshold * topk_probs[:q_chunk[0].size(0), 0]  # [batch_size]
        
        # Combine masks
        accept_mask = in_topk & meets_threshold  # [batch_size]
        
        # Find first rejection
        first_reject = (~accept_mask).nonzero()
        if len(first_reject) > 0:
            accept_len = first_reject[0].item()
        else:
            accept_len = len(accept_mask)
        
        # Vectorized token selection
        # Create a tensor of selected tokens where:
        # - For accepted positions: use draft tokens
        # - For rejected positions: use top token
        selected_tokens = torch.where(
            accept_mask.unsqueeze(1),
            q_chunk[0, :accept_mask.size(0)].unsqueeze(1),
            topk_ids[:accept_mask.size(0), 0].unsqueeze(1)
        ).squeeze(1)
        
        # Convert to Dream tokens and take only up to accept_len
        chosen_dream = [
            self.token_mapper.verifier_to_dream_id(t.item())
            for t in selected_tokens[:accept_len + 1]  # +1 to include the rejection token
        ]
        
        return chosen_dream, accept_len

    @torch.no_grad()
    def batch_verify(
        self,
        dream_prefix: Tensor,
        draft_ids   : Tensor,          # Dream-vocab ids for *masked* slots
        batch_size  : int = 32,
        rng: torch.Generator | None = None,
        verification_stats: dict | None = None,  # Add verification_stats parameter
        sampling_strategy: str = "deterministic",  # Use string-based strategy selection
        confidence_threshold: float = 0.1,  # Fixed confidence threshold
        acceptance_top_k: int = 5,  # Add top-k parameter
        top_p: float = 0.95,  # Add top-p parameter
    ) -> List[int]:
        """
        Verify `draft_ids` in contiguous chunks of `batch_size`.
        Stops at the first rejection and returns the Dream-vocab *choices*
        (length == processed tokens).  KV-cache is trimmed so it contains
        only the accepted prefix.
        """
        self._ensure_prefix(dream_prefix[:-1])  # Only ensure prefix up to last token
        dev = self.dev
        rng = _rng_on_device(rng, dev)

        # Update verification stats for this batch
        self.verification_stats = verification_stats

        # Get last token from prefix and ensure it's on the correct device
        last_token = dream_prefix[-1].unsqueeze(0).to(dev)  # Shape: [1]
        
        # Ensure draft_ids is on the correct device
        draft_ids = draft_ids.to(dev)

        chosen_dream : List[int] = []
        i = 0
        while i < len(draft_ids):
            # Debug logging
            logger.debug(f"Processing chunk starting at index {i}, remaining tokens: {len(draft_ids) - i}")
            
            # Prepare chunk with proper overlap
            if i == 0:
                # First chunk: [last_token, draft_ids[0:batch_size-1]]
                chunk_ar_inp = torch.cat([
                    last_token,
                    draft_ids[i:i + batch_size-1]
                ])
            else:
                # Subsequent chunks: [draft_ids[i-1:i + batch_size-1]]
                chunk_ar_inp = draft_ids[i-1:i+batch_size-1]
            
            # Debug logging
            logger.debug(f"Chunk size: {len(chunk_ar_inp)}")

            # Convert to Qwen IDs in batch
            q_chunk_ar_inp = self.token_mapper.dream_to_verifier_ids(chunk_ar_inp).unsqueeze(0)
            
            # print(f"verifying {q_chunk_ar_inp.shape} with {len(self.cache_q)} cache")
            out = self.model(
                q_chunk_ar_inp, # AR input is left-shifted by 1, because it predicts the next token
                use_cache=True, 
                past_key_values=self.kv)
            
            logits_blk = out.logits[0]
            if verification_stats is not None:
                verification_stats['ar_model_calls'] += 1
            
            # Get probabilities directly from logits
            probs_blk = F.softmax(logits_blk, dim=-1)

            # Use the correct draft_ids for comparison
            chunk = draft_ids[i:i + batch_size]
            q_chunk = self.token_mapper.dream_to_verifier_ids(chunk).unsqueeze(0)

            # Call appropriate verification strategy
            strategy = SamplingStrategy.from_string(sampling_strategy)
            if strategy == SamplingStrategy.DETERMINISTIC:
                chosen_dream_chunk, accept_len = self._verify_deterministic(q_chunk, probs_blk)
            elif strategy == SamplingStrategy.DYNAMIC_THRESHOLD:
                chosen_dream_chunk, accept_len = self._verify_dynamic_threshold(q_chunk, probs_blk, confidence_threshold, rng)
            elif strategy == SamplingStrategy.AGGRESSIVE:
                chosen_dream_chunk, accept_len = self._verify_aggressive(q_chunk, probs_blk, rng)
            elif strategy == SamplingStrategy.TOPK:
                chosen_dream_chunk, accept_len = self._verify_topk(q_chunk, probs_blk, acceptance_top_k)
            elif strategy == SamplingStrategy.TOPK_RELATIVE:
                chosen_dream_chunk, accept_len = self._verify_topk_relative(q_chunk, probs_blk, top_k=acceptance_top_k, relative_threshold=top_p)
            elif strategy == SamplingStrategy.ORIGINAL:
                chosen_dream_chunk, accept_len = self._verify_original(q_chunk, probs_blk, rng)

            chosen_dream.extend(chosen_dream_chunk)

            # Debug logging
            logger.debug(f"Accepted {accept_len} tokens in this chunk")

            # ── merge + crop KV-cache to keep only accepted cells ────────
            pkv_cls = out.past_key_values.__class__
            if self.kv is None:
                merged = out.past_key_values
            else:
                merged_layers = [
                    (
                        torch.cat([k0, k1], dim=2),
                        torch.cat([v0, v1], dim=2),
                    )
                    for (k0, v0), (k1, v1) in zip(self.kv, out.past_key_values)
                ]
                merged = pkv_cls(merged_layers)

            # Keep KV cache for all accepted tokens
            new_len = len(self.cache_q) + accept_len + 1
            trimmed = _crop_past_key_values(self.model, merged, new_len)
            self.kv = trimmed if isinstance(trimmed, pkv_cls) else pkv_cls(trimmed)
            
            # Update cache_q to match the KV cache
            self.cache_q = self.cache_q[:new_len]
            self.cache_q.extend(q_chunk_ar_inp[0, :accept_len + 1].tolist())

            # stop at first rejection inside this chunk
            if accept_len < len(q_chunk[0]):
                break
                
            # Update i to move to next chunk
            i += batch_size

        return chosen_dream

# ──────────────────────────────────────────────────────────────────────────────
# Helper – ensure RNG is on correct device
# ──────────────────────────────────────────────────────────────────────────────
def _rng_on_device(rng: torch.Generator | None, dev: torch.device) -> torch.Generator:
    if rng is not None and rng.device != dev:
        new_rng = torch.Generator(device=dev)
        new_rng.manual_seed(rng.initial_seed())
        return new_rng
    return rng or torch.Generator(device=dev)

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Early-stop check
# ──────────────────────────────────────────────────────────────────────────────
def _early_stop_ok(seq: Tensor, eos: int, mask: int, k: int) -> bool:
    if k <= 0: return False
    return (seq[0, -k:] == eos).all() and (seq == mask).sum() == 0

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Main speculative generator  (DEBUG prints kept verbatim)
# ──────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def speculative_diffusion_generate(
    *,
    dream_model,
    ar_model,
    input_ids: Tensor,
    attention_mask: Optional[Tensor],
    config: SpecDiffusionConfig,
    dream_tokenizer,
    ar_tokenizer,
    ar_verifier: ARVerifier = None,
    rng: Optional[torch.Generator] = None,
) -> Union[Tensor, SpecDiffusionOutput]:

    rng      = rng or torch.default_generator
    mask_id  = config.mask_token_id
    eos_id   = config.eos_token_id
    max_len  = config.max_length
    dev      = input_ids.device

    # ── pad prompt to max_len ─────────────────────────────────────────────
    seq = F.pad(input_ids, (0, max_len - input_ids.size(1)),
                value=mask_id).to(dev)
    last_non_mask = input_ids.size(1) - 1  # Initialize to last position of input

    tok_idx = None
    if attention_mask is not None:
        att = F.pad(attention_mask, (0, max_len - attention_mask.size(1)), value=1).to(dev)
        tok_idx = att.long().cumsum(-1) - 1
        tok_idx.masked_fill_(att == 0, 1)
        attention_mask = torch.logical_and(att[:, None, :, None],
                                           att[:, None, None, :])

    # Track verification statistics
    verification_stats = {
        'total_tokens': 0,
        'total_tokens_verified': 0,
        'accepted_tokens': 0,
        'rejected_tokens': 0,
        'acceptance_rate': 0.0,
        'diffusion_steps': 0,
        'total_steps': 0,
        'tokens_verified_per_step': [],
        'ar_model_calls': 0,
        'rejection_correction_calls': 0,
        'unmasked_tokens_per_step': [],  # Track number of unmasked tokens at each step
    }

    history = ([seq.clone()] if (config.output_history
                                 and config.return_dict_in_generate) else None)
    
    # Use existing verifier or create new one
    if ar_verifier is None:
        verifier = ARVerifier(ar_model, dream_tokenizer, ar_tokenizer)
    else:
        verifier = ar_verifier
        verifier.reset()  # Reset state for new inference

    stopped = False
    logger.debug(f"start decode, max_len={max_len}")
    diffusion_step = 0
    while (seq == mask_id).any():
        logger.debug("================================================")
        logger.debug(f"diffusion_step: {diffusion_step}")
        verification_stats['total_steps'] += 1

        seq_txt = dream_tokenizer.decode(
            seq[0, input_ids.size(1):].tolist()).rstrip(dream_tokenizer.mask_token)
        logger.debug(f"seq_txt: {seq_txt}")

        # ── Dream forward (draft) ────────────────────────────────────────
        out    = dream_model(seq, attention_mask, tok_idx, save_cache=False)
        logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], 1)
        mask_pos = (seq[0] == mask_id).nonzero(as_tuple=True)[0]
        drafts   = logits[0, mask_pos, :].argmax(-1)

        logger.debug(f"step {diffusion_step} masks={mask_pos.numel()}")

        # ── verifier in batched chunks ───────────────────────────────────
        mask_pos_ls = mask_pos.tolist()
        drafts = drafts.to(dev)  # Ensure drafts are on the correct device

        dream_prefix = seq[0, :mask_pos_ls[0]]        # clean prefix
        chosen_chunk = verifier.batch_verify(
            dream_prefix, 
            drafts,
            batch_size=config.verify_batch_size,
            rng=rng,
            verification_stats=verification_stats,
            sampling_strategy=config.sampling_strategy,
            confidence_threshold=config.confidence_threshold,
            acceptance_top_k=config.acceptance_top_k,
            top_p=config.top_p
        )

        # Update verification statistics
        chunk_size = len(drafts)
        verification_stats['total_tokens'] += chunk_size
        verification_stats['total_tokens_verified'] += chunk_size
        tokens_verified_this_step = chunk_size

        # Compare drafts and chosen tokens
        for dr, ch in zip(drafts.tolist(), chosen_chunk):
            if dr == ch:
                verification_stats['accepted_tokens'] += 1
            else:
                verification_stats['rejected_tokens'] += 1

        # Update sequence with verified tokens
        reject_idx = None
        tokens_updated = 0
        # debugging, counting the number of iterations in the loop
        # num_iterations = 0
        # print(f"len(mask_pos_ls): {len(mask_pos_ls)}")
        # print(f"len(drafts.tolist()): {len(drafts.tolist())}")
        # print(f"len(chosen_chunk): {len(chosen_chunk)}")
        for off, (p, dr, ch) in enumerate(zip(mask_pos_ls, drafts.tolist(), chosen_chunk)):
            # num_iterations += 1
            seq[0, p] = ch
            tokens_updated += 1
            last_non_mask = p  # Update last non-masked position
            logger.debug(f"    slot {p}: draft={dream_tokenizer.decode([dr])} ({dr})"
                  f" → chosen={dream_tokenizer.decode([ch])} ({ch})")
            
            # if ch == eos_id:
            # Hack: for dream and qwen, both <|endoftext|> and <|im_end|> are EOS
            # check if ch is in the list of eos tokens
            # <|box_end|> is the also eos token for qwen
            eos_tokens = [151643, 151645, 151649]
            if ch in eos_tokens:
                verification_stats['unmasked_tokens_per_step'].append(tokens_updated)
                print(f"Early stop (eos) at step {diffusion_step}")
                seq[seq == mask_id] = eos_id
                stopped = True
                break
            
            if ch != dr and reject_idx is None:
                reject_idx = off

                
                if tokens_updated == 1:
                    print(f"\033[1;33mStep {diffusion_step}\033[0m - \033[1;36mDraft: {dream_tokenizer.decode([dr])} ({dr})\033[0m | \033[1;35mAR: {dream_tokenizer.decode([ch])} ({ch})\033[0m")
                    
                    # if config.stop_on_dream_eos and dr == config.eos_token_id:
                    # hardcode Dream eos to be 151643
                    if config.stop_on_dream_eos and dr == 151643:
                        if config.eos_token_id != 151643:
                            # warning
                            logger.warning(f"config.eos_token_id is not 151643, but {config.eos_token_id}")
                        print(f"Early stop (diffusion eos) at step {diffusion_step}")
                        seq[seq == mask_id] = eos_id
                        stopped = True
                        break
                    break
            
            
            
        # print(f"num_iterations: {num_iterations}")
            
                
        if not stopped:
            verification_stats['unmasked_tokens_per_step'].append(tokens_updated)

        if reject_idx is not None:
            diffusion_step += 1
            logger.debug(f"Rejected: {dream_tokenizer.decode([drafts.tolist()[reject_idx]])}"
                  f" ({drafts.tolist()[reject_idx]}) → "
                  f"{dream_tokenizer.decode([chosen_chunk[reject_idx]])}"
                  f" ({chosen_chunk[reject_idx]})")
            continue  # new Dream forward

        # Record tokens verified in this step
        verification_stats['tokens_verified_per_step'].append(tokens_verified_this_step)

        if history is not None:
            history.append(seq.clone())

        if config.early_stop and _early_stop_ok(
                seq, eos_id, mask_id, config.early_stop_consecutive):
            print(f"Early stop: at step {diffusion_step}")
            seq[seq == mask_id] = eos_id
            break

        if (seq == mask_id).sum() == 0:
            print(f"No masked: Early stop at step {diffusion_step}")
            break
        
        if stopped:
            print(f"Stopped at step {diffusion_step}")
            break

    # Calculate final acceptance rate
    if verification_stats['total_tokens'] > 0:
        verification_stats['acceptance_rate'] = (
            verification_stats['accepted_tokens'] / verification_stats['total_tokens']
        )
    verification_stats['diffusion_steps'] = diffusion_step

    if config.return_dict_in_generate:
        return SpecDiffusionOutput(sequences=seq,
                                   history=tuple(history) if history else None,
                                   verification_stats=verification_stats)
    return seq
@torch.inference_mode()
def speculative_block_diffusion_generate(
    *,
    dream_model,
    ar_model,
    input_ids: Tensor,
    attention_mask: Optional[Tensor],
    config: SpecDiffusionConfig,
    dream_tokenizer,
    ar_tokenizer,
    ar_verifier: ARVerifier = None,
    rng: Optional[torch.Generator] = None,
) -> Union[Tensor, SpecDiffusionOutput]:
    """
    Speculative diffusion generation with block KV caching.
    """
    rng = rng or torch.default_generator
    mask_id = config.mask_token_id
    eos_id = config.eos_token_id
    max_len = config.max_length
    dev = input_ids.device

    # Pad prompt to max_len
    seq = F.pad(input_ids, (0, max_len - input_ids.size(1)),
                value=mask_id).to(dev)
    last_non_mask = input_ids.size(1) - 1

    # Setup attention mask and token indices
    tok_idx = None
    if attention_mask is not None:
        att = F.pad(attention_mask, (0, max_len - attention_mask.size(1)), value=1).to(dev)
        tok_idx = att.long().cumsum(-1) - 1
        tok_idx.masked_fill_(att == 0, 1)
        attention_mask = torch.logical_and(att[:, None, :, None],
                                         att[:, None, None, :])

    # Track verification statistics
    verification_stats = {
        'total_tokens': 0,
        'total_tokens_verified': 0,
        'accepted_tokens': 0,
        'rejected_tokens': 0,
        'acceptance_rate': 0.0,
        'diffusion_steps': 0,
        'total_steps': 0,
        'tokens_verified_per_step': [],
        'ar_model_calls': 0,
        'rejection_correction_calls': 0,
        'unmasked_tokens_per_step': []  # Track number of unmasked tokens at each step
    }

    history = ([seq.clone()] if (config.output_history
                               and config.return_dict_in_generate) else None)
    
    # Use existing verifier or create new one
    if ar_verifier is None:
        verifier = ARVerifier(ar_model, dream_tokenizer, ar_tokenizer)
    else:
        verifier = ar_verifier
        verifier.reset()  # Reset state for new inference

    # Block parameters
    block_size = config.block_size if hasattr(config, 'block_size') else 32
    prompt_len = input_ids.size(1)
    bsz = 1  # batch size
    L = max_len  # sequence length

    stopped = False
    logger.debug(f"start decode, max_len={max_len}")
    diffusion_step = 0

    # Check caching strategies
    aggressive_caching = getattr(config, 'aggressive_caching', False)
    use_block_boundary_caching = getattr(config, 'use_block_boundary_caching', False)
    aggressive_verify = getattr(config, 'aggressive_verify', False)  # Add flag for aggressive verification

    while (seq == mask_id).any():
        logger.debug("================================================")
        logger.debug(f"diffusion_step: {diffusion_step}")
        verification_stats['total_steps'] += 1
        
        # Dream forward (draft) with block caching
        if diffusion_step == 0:
            # First step: reset block cache and save cache up to prompt
            for layer_idx in range(dream_model.config.num_hidden_layers):
                dream_model.model.layers[layer_idx].reset_block_cache(bsz, L, block_size)
            
            out = dream_model(seq, None, tok_idx if tok_idx is not None else None,
                            use_block_diffusion=True,
                            use_full_query_attn=False,
                            max_length=max_len,
                            block_size=block_size,
                            save_cache=True,
                            clean_idx=prompt_len) 
            # Logits has shape [1, max_len, vocab_size]
            logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], 1)
            mask_pos = (seq[0] == mask_id).nonzero(as_tuple=True)[0]
            drafts = logits[0, mask_pos, :].argmax(-1)
        else:
            if config.sliding_window_caching:

                # # Original version
                # # Calculate sliding window start position
                # sliding_window_start = max(0, last_non_mask - config.sliding_window_size)
                # gen_seq = seq[:, sliding_window_start:]
                # gen_tok_idx = tok_idx[:, sliding_window_start:] if tok_idx is not None else None
                
                # out = dream_model(gen_seq, None, gen_tok_idx,
                #                 use_block_diffusion=True,
                #                 use_full_query_attn=False,
                #                 max_length=max_len,
                #                 block_size=block_size,
                #                 save_cache=True,
                #                 clean_idx=sliding_window_start + 1)  # Save cache up to sliding window start
                
                # # Logits has shape [1, remaining_len, vocab_size]
                # logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], 1)
                # # Get mask positions relative to the generation part
                # gen_mask_pos = (gen_seq[0] == mask_id).nonzero(as_tuple=True)[0]
                # drafts = logits[0, gen_mask_pos, :].argmax(-1)
                
                # # Convert gen_mask_pos back to full sequence positions for updating
                # mask_pos = gen_mask_pos + sliding_window_start

                # New version
                # Calculate sliding window start and end positions
                sliding_window_start = max(0, last_non_mask - config.sliding_window_size)
                sliding_window_end = min(max_len, last_non_mask + config.sliding_window_size)

                # Debug: print sliding window with size 
                # print(f"Using sliding window from {sliding_window_start} to {sliding_window_end} (size={sliding_window_end - sliding_window_start})")
                
                # Get the generation part from sliding window
                gen_seq = seq[:, sliding_window_start:sliding_window_end]
                gen_tok_idx = tok_idx[:, sliding_window_start:sliding_window_end] if tok_idx is not None else None
                
                
                out = dream_model(gen_seq, None, gen_tok_idx,
                                use_block_diffusion=True,
                                use_full_query_attn=False,
                                max_length=max_len,
                                block_size=block_size,
                                save_cache=True,
                                clean_idx=sliding_window_start + 1)  # Save cache up to sliding window start

                # Logits has shape [1, 2 x window_size, vocab_size]
                logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], 1)

                # Get mask positions relative to the generation part
                gen_mask_pos = (gen_seq[0] == mask_id).nonzero(as_tuple=True)[0]
                drafts = logits[0, gen_mask_pos, :].argmax(-1)
                
                # Convert gen_mask_pos back to full sequence positions for updating
                mask_pos = gen_mask_pos + sliding_window_start



            elif use_block_boundary_caching:
                # Calculate last clean block boundary
                last_clean_block = ((last_non_mask + 1) // block_size) * block_size
                
                # Start from last clean block
                gen_seq = seq[:, last_clean_block:]  # Only the generation part from last clean block
                gen_tok_idx = tok_idx[:, last_clean_block:] if tok_idx is not None else None
                
                out = dream_model(gen_seq, None, gen_tok_idx,
                                use_block_diffusion=True,
                                use_full_query_attn=False,
                                max_length=max_len,
                                block_size=block_size,
                                save_cache=True,
                                clean_idx=last_clean_block + 1)  # Save cache up to last clean block
                
                # Logits has shape [1, remaining_len, vocab_size]
                logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], 1)
                # Get mask positions relative to the generation part
                gen_mask_pos = (gen_seq[0] == mask_id).nonzero(as_tuple=True)[0]
                drafts = logits[0, gen_mask_pos, :].argmax(-1)
                
                # Convert gen_mask_pos back to full sequence positions for updating
                mask_pos = gen_mask_pos + last_clean_block
            elif aggressive_caching:
                # Start from last clean token
                gen_seq = seq[:, last_non_mask:]  # Only the generation part from last clean token
                gen_tok_idx = tok_idx[:, last_non_mask:] if tok_idx is not None else None
                
                out = dream_model(gen_seq, None, gen_tok_idx,
                                use_block_diffusion=True,
                                use_full_query_attn=False,
                                max_length=max_len,
                                block_size=block_size,
                                save_cache=True,
                                clean_idx=last_non_mask + 1)  # Save cache up to last clean token
                
                # Logits has shape [1, remaining_len, vocab_size]
                logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], 1)
                # Get mask positions relative to the generation part
                gen_mask_pos = (gen_seq[0] == mask_id).nonzero(as_tuple=True)[0]
                drafts = logits[0, gen_mask_pos, :].argmax(-1)
                
                # Convert gen_mask_pos back to full sequence positions for updating
                mask_pos = gen_mask_pos + last_non_mask
            else:
                # Original behavior: start from prompt_len
                gen_seq = seq[:, prompt_len:]  # Only the generation part
                gen_tok_idx = tok_idx[:, prompt_len:] if tok_idx is not None else None
                
                out = dream_model(gen_seq, None, gen_tok_idx,
                                use_block_diffusion=True,
                                use_full_query_attn=False,
                                max_length=max_len,
                                block_size=block_size,
                                save_cache=False,
                                clean_idx=None)
                
                # Logits has shape [1, max_len - prompt_len (i.e., gen_len), vocab_size]
                logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], 1)
                # Get mask positions relative to the generation part
                gen_mask_pos = (gen_seq[0] == mask_id).nonzero(as_tuple=True)[0]
                drafts = logits[0, gen_mask_pos, :].argmax(-1)
                
                # Convert gen_mask_pos back to full sequence positions for updating
                mask_pos = gen_mask_pos + prompt_len

        logger.debug(f"step {diffusion_step} masks={mask_pos.numel()}")

        # Verifier in batched chunks
        mask_pos_ls = mask_pos.tolist()
        drafts = drafts.to(dev)  # Ensure drafts are on the correct device

        dream_prefix = seq[0, :mask_pos_ls[0]]

        
        chosen_chunk = verifier.batch_verify(
            dream_prefix, drafts,
            batch_size=config.verify_batch_size,
            rng=rng,
            verification_stats=verification_stats,
            sampling_strategy=config.sampling_strategy,
            confidence_threshold=config.confidence_threshold,
            acceptance_top_k=config.acceptance_top_k,
            top_p=config.top_p
        )
        

        # Update verification statistics
        chunk_size = len(drafts)
        verification_stats['total_tokens'] += chunk_size
        verification_stats['total_tokens_verified'] += chunk_size
        tokens_verified_this_step = chunk_size

        # Compare drafts and chosen tokens
        for dr, ch in zip(drafts.tolist(), chosen_chunk):
            if dr == ch:
                verification_stats['accepted_tokens'] += 1
            else:
                verification_stats['rejected_tokens'] += 1

        # Update sequence with verified tokens
        reject_idx = None
        tokens_updated = 0
        # debugging, counting the number of iterations in the loop
        # num_iterations = 0
        # print(f"len(mask_pos_ls): {len(mask_pos_ls)}")
        # print(f"len(drafts.tolist()): {len(drafts.tolist())}")
        # print(f"len(chosen_chunk): {len(chosen_chunk)}")
        for off, (p, dr, ch) in enumerate(zip(mask_pos_ls, drafts.tolist(), chosen_chunk)):
            # num_iterations += 1
            seq[0, p] = ch
            tokens_updated += 1
            last_non_mask = p  # Update last non-masked position
            logger.debug(f"    slot {p}: draft={dream_tokenizer.decode([dr])} ({dr})"
                  f" → chosen={dream_tokenizer.decode([ch])} ({ch})")

            # if ch == eos_id:
            # Hack: for dream and qwen, both <|endoftext|> and <|im_end|> are EOS
            # check if ch is in the list of eos tokens
            eos_tokens = [151643, 151645]
            if ch in eos_tokens:
                verification_stats['unmasked_tokens_per_step'].append(tokens_updated)
                print(f"Early stop (eos) at step {diffusion_step}")
                seq[seq == mask_id] = eos_id
                stopped = True
                break

            # Token 151643 and 151645 are EOS for dream and qwen
            if ch != dr and reject_idx is None:
                reject_idx = off

                if tokens_updated == 1:
                    print(f"\033[1;33mStep {diffusion_step}\033[0m - \033[1;36mDraft: {dream_tokenizer.decode([dr])} ({dr})\033[0m | \033[1;35mAR: {dream_tokenizer.decode([ch])} ({ch})\033[0m")

                    if config.stop_on_dream_eos and dr == config.eos_token_id:
                        print(f"Early stop (diffusion eos) at step {diffusion_step}")
                        seq[seq == mask_id] = eos_id
                        stopped = True
                        break

                    break
                

            
        # print(f"num_iterations: {num_iterations}")
                
        if not stopped:
            verification_stats['unmasked_tokens_per_step'].append(tokens_updated)

        if reject_idx is not None:
            diffusion_step += 1
            logger.debug(f"Rejected: {dream_tokenizer.decode([drafts.tolist()[reject_idx]])}"
                  f" ({drafts.tolist()[reject_idx]}) → "
                  f"{dream_tokenizer.decode([chosen_chunk[reject_idx]])}"
                  f" ({chosen_chunk[reject_idx]})")
            continue  # new Dream forward

        # Record tokens verified in this step
        verification_stats['tokens_verified_per_step'].append(tokens_verified_this_step)

        if history is not None:
            history.append(seq.clone())

        if stopped:
            print(f"Stopped at step {diffusion_step}")
            break

        if config.early_stop and _early_stop_ok(
                seq, eos_id, mask_id, config.early_stop_consecutive):
            print(f"Early stop: at step {diffusion_step}")
            seq[seq == mask_id] = eos_id
            break

    # Calculate final acceptance rate
    if verification_stats['total_tokens'] > 0:
        verification_stats['acceptance_rate'] = (
            verification_stats['accepted_tokens'] / verification_stats['total_tokens']
        )
    verification_stats['diffusion_steps'] = diffusion_step

    if config.return_dict_in_generate:
        return SpecDiffusionOutput(sequences=seq,
                                 history=tuple(history) if history else None,
                                 verification_stats=verification_stats)
    return seq


# File: spec_diffusion_utils.py
"""
Speculative-diffusion decoding
(Dream draft model  +  Qwen-2.5 verifier)

All `[DEBUG …]` prints from your earlier version are kept intact.
"""

from __future__ import annotations
import sys, warnings
import time
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
class AssistedDiffusionOutput(ModelOutput):
    sequences: torch.LongTensor
    history  : Optional[Tuple[torch.FloatTensor]] = None
    verification_stats: Optional[dict] = None  # Track verification metrics


class AssistedDiffusionConfig(GenerationConfig):
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

        # verifier batching
        self.verify_batch_size      = kw.pop("verify_batch_size", 32)

        # verification strategy
        self.sampling_strategy      = kw.pop("sampling_strategy", "deterministic")
        self.confidence_threshold   = kw.pop("confidence_threshold", 0.1)
        self.acceptance_top_k       = kw.pop("acceptance_top_k", 5)  # Add top-k parameter for acceptance

        # misc
        self.return_dict_in_generate= kw.pop("return_dict_in_generate", False)
        self.output_history         = kw.pop("output_history", False)

        # sliding window diffusion
        self.use_sliding_window_caching = kw.pop("use_sliding_window_caching", False)
        self.sliding_window_size = kw.pop("sliding_window_size", 512)
        
        # block boundary caching
        self.use_block_boundary_caching = kw.pop("use_block_boundary_caching", False)

        # ignored legacy keys
        kw.pop("steps", None); kw.pop("eps", None)

        self.transformers_version   = kw.pop("transformers_version", __version__)
        if kw:
            logger.warning(f"Ignored unknown config keys: {list(kw.keys())}")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Token-mapping helpers
# ──────────────────────────────────────────────────────────────────────────────
def _dream_txt(dtok, ids: List[int]) -> str:
    return dtok.decode(ids, skip_special_tokens=False)

def _dream_to_qwen_tensor(ids: Tensor, dtok, qtok, dev, skip_conversion: bool) -> Tensor:
    if skip_conversion:
        return ids.to(dev)  # Skip conversion, just move to correct device
    
    # Fall back to conversion if vocabularies differ
    txt  = _dream_txt(dtok, ids.tolist())
    qids = qtok(txt, add_special_tokens=False).input_ids
    return torch.tensor(qids, device=dev)

def _dream_id_to_qwen_id(did: int, dtok, qtok, skip_conversion: bool) -> int:
    if skip_conversion:
        return did  # Skip conversion
    
    # Fall back to conversion if vocabularies differ
    return qtok(dtok.decode([did], skip_special_tokens=False),
                add_special_tokens=False).input_ids[0]

def _qwen_id_to_dream_id(qid: int, dtok, qtok, skip_conversion: bool) -> int:
    if skip_conversion:
        return qid  # Skip conversion
    
    # Fall back to conversion if vocabularies differ
    return dtok(qtok.decode([qid], skip_special_tokens=False),
                add_special_tokens=False).input_ids[0]

# ──────────────────────────────────────────────────────────────────────────────
# 3.  AR verifier with KV-cache
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
        

    def dream_to_qwen_tensor(self, ids: Tensor, dev: torch.device) -> Tensor:
        """Convert Dream token IDs to Qwen token IDs."""
        if self.skip_conversion:
            return ids.to(dev)
        
        txt = self.dtok.decode(ids.tolist(), skip_special_tokens=False)
        qids = self.qtok(txt, add_special_tokens=False).input_ids
        return torch.tensor(qids, device=dev)

    def dream_to_qwen_id(self, did: int) -> int:
        """Convert single Dream token ID to Qwen token ID."""
        if self.skip_conversion:
            # Check if token has an equivalent
            return self.equivalent_tokens.get(did, did)
        
        return self.qtok(self.dtok.decode([did], skip_special_tokens=False),
                        add_special_tokens=False).input_ids[0]

    def dream_to_qwen_ids(self, dids: Tensor) -> Tensor:
        """Convert batch of Dream token IDs to Qwen token IDs."""
        if self.skip_conversion:
            # Convert each ID individually to handle equivalents
            # return torch.tensor([self.dream_to_qwen_id(did.item()) for did in dids], device=dids.device)
            return dids
        
        # Convert all IDs at once
        txt = self.dtok.decode(dids.tolist(), skip_special_tokens=False)
        qids = self.qtok(txt, add_special_tokens=False).input_ids
        return torch.tensor(qids, device=dids.device)

    def qwen_to_dream_id(self, qid: int) -> int:
        """Convert single Qwen token ID to Dream token ID."""
        if self.skip_conversion:
            # Check if token has an equivalent
            return self.equivalent_tokens.get(qid, qid)
        
        return self.dtok(self.qtok.decode([qid], skip_special_tokens=False),
                        add_special_tokens=False).input_ids[0]

    def qwen_to_dream_ids(self, qids: Tensor) -> Tensor:
        """Convert batch of Qwen token IDs to Dream token IDs."""
        if self.skip_conversion:
            # return qids
            # return torch.tensor([self.qwen_to_dream_id(qid.item()) for qid in qids], device=qids.device)
            return qids
        
        # Convert all IDs at once
        txt = self.qtok.decode(qids.tolist(), skip_special_tokens=False)
        dids = self.dtok(txt, add_special_tokens=False).input_ids
        return torch.tensor(dids, device=qids.device)




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

class ARAssistant:
    """
    Thin wrapper around the AR verifier model (Qwen-2.5) that keeps its
    PastKeyValues cache and supports batched verification.
    """
    def __init__(self, ar_model, dream_tok, qwen_tok):
        self.model = ar_model
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
            q_new_tokens = self.token_mapper.dream_to_qwen_tensor(new_tokens, self.dev)
            logits = self._push(q_new_tokens.unsqueeze(0))
            self.last_logits = logits

        # Original version (commented out for reference):
        # # Use complete prefix for verification
        # tgt = self.token_mapper.dream_to_qwen_tensor(dream_prefix, self.dev)
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
                chosen_dream.append(self.token_mapper.qwen_to_dream_id(q_id.item()))
                accept_len += 1
            else:
                # If not accepted, use selected token
                chosen_dream.append(self.token_mapper.qwen_to_dream_id(selected_ids_slice[j].item()))
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
                chosen_dream.append(self.token_mapper.qwen_to_dream_id(q_id.item()))
                accept_len += 1
            else:
                # If not accepted, use selected token and stop
                chosen_dream.append(self.token_mapper.qwen_to_dream_id(selected_id))
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
                chosen_dream.append(self.token_mapper.qwen_to_dream_id(q_id.item()))
                accept_len += 1
            else:
                # Sample from Qwen's distribution excluding Dream's token
                probs_alt = probs_i.clone()
                probs_alt[q_id] = 0  # Zero out draft token
                probs_alt.div_(probs_alt.sum())
                alt_q = torch.multinomial(probs_alt, 1, generator=rng).item()
                chosen_dream.append(self.token_mapper.qwen_to_dream_id(alt_q))
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
                chosen_dream.append(self.token_mapper.qwen_to_dream_id(q_id.item()))
                accept_len += 1
                continue
            
            # Check if verifier's probability for draft token is high enough
            draft_prob = probs_i[q_id]
            if draft_prob > 0.05:
                chosen_dream.append(self.token_mapper.qwen_to_dream_id(q_id.item()))
                accept_len += 1
                continue
            
            # Check if draft token is in top-p
            sorted_probs, sorted_ids = probs_i.sort(descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            top_p_mask = cumsum <= 0.95
            top_p_ids = sorted_ids[top_p_mask]
            
            if q_id.item() in top_p_ids:
                chosen_dream.append(self.token_mapper.qwen_to_dream_id(q_id.item()))
                accept_len += 1
                continue
            
            # If not accepted, sample from verifier's distribution
            probs_alt = probs_i.clone()
            probs_alt[q_id] = 0
            probs_alt.div_(probs_alt.sum())
            alt_q = torch.multinomial(probs_alt, 1, generator=rng).item()
            chosen_dream.append(self.token_mapper.qwen_to_dream_id(alt_q))
            break
        return chosen_dream, accept_len

    def _verify_topk(self, q_chunk: Tensor, probs_blk: Tensor, acceptance_top_k: int) -> Tuple[List[int], int]:
        """Top-k verification strategy."""
        chosen_dream = []
        accept_len = 0
        for q_id, probs_i in zip(q_chunk[0], probs_blk):
            # Get top-k tokens from verifier's distribution
            top_k = acceptance_top_k
            top_k_probs, top_k_ids = probs_i.topk(top_k)
            
            # Check if draft token is in top-k
            if q_id.item() in top_k_ids:
                chosen_dream.append(self.token_mapper.qwen_to_dream_id(q_id.item()))
                accept_len += 1
                continue
            
            # If not accepted, fall back to top-1 (use verifier's argmax)
            verifier_argmax = torch.argmax(probs_i).item()
            chosen_dream.append(self.token_mapper.qwen_to_dream_id(verifier_argmax))
            break
        return chosen_dream, accept_len

    def _verify_original(self, q_chunk: Tensor, probs_blk: Tensor, rng: torch.Generator) -> Tuple[List[int], int]:
        """Original verification strategy."""
        chosen_dream = []
        accept_len = 0
        for q_id, probs_i in zip(q_chunk[0], probs_blk):
            if torch.rand((), device=self.dev, generator=rng) < probs_i[q_id]:
                chosen_dream.append(self.token_mapper.qwen_to_dream_id(q_id.item()))
                accept_len += 1
            else:
                probs_alt = probs_i.clone()
                probs_alt[q_id] = 0
                probs_alt.div_(probs_alt.sum())
                alt_q = torch.multinomial(probs_alt, 1, generator=rng).item()
                chosen_dream.append(self.token_mapper.qwen_to_dream_id(alt_q))
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
                chosen_dream.append(self.token_mapper.qwen_to_dream_id(q_id.item()))
                accept_len += 1
            else:
                # If not accepted, use top token
                chosen_dream.append(self.token_mapper.qwen_to_dream_id(topk_ids_i[0].item()))
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
            self.token_mapper.qwen_to_dream_id(t.item())
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
        total_ar_model_time = 0.0
        total_verification_time = 0.0
        total_accept_len = 0  # Track total accepted tokens across all chunks
        last_out = None  # Store the last output for KV cache management
        last_q_chunk_ar_inp = None  # Store the last input for KV cache management
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
            q_chunk_ar_inp = self.token_mapper.dream_to_qwen_ids(chunk_ar_inp).unsqueeze(0)
            
            # AR model forward pass timing
            ar_start = time.perf_counter()
            out = self.model(
                q_chunk_ar_inp, # AR input is left-shifted by 1, because it predicts the next token
                use_cache=True, 
                past_key_values=self.kv)
            ar_end = time.perf_counter()
            total_ar_model_time += (ar_end - ar_start) * 1000  # Convert to ms
            
            logits_blk = out.logits[0]
            if verification_stats is not None:
                verification_stats['ar_model_calls'] += 1
            
            # Get probabilities directly from logits
            probs_blk = F.softmax(logits_blk, dim=-1)

            # Use the correct draft_ids for comparison
            chunk = draft_ids[i:i + batch_size]
            q_chunk = self.token_mapper.dream_to_qwen_ids(chunk).unsqueeze(0)

            # Verification strategy timing
            verify_start = time.perf_counter()
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
            verify_end = time.perf_counter()
            total_verification_time += (verify_end - verify_start) * 1000  # Convert to ms

            # Store timing data in verification_stats
            if verification_stats is not None and 'time_breakdown' in verification_stats:
                if 'ar_model_forward_time' not in verification_stats['time_breakdown']:
                    verification_stats['time_breakdown']['ar_model_forward_time'] = []
                if 'verification_strategy_time' not in verification_stats['time_breakdown']:
                    verification_stats['time_breakdown']['verification_strategy_time'] = []
                verification_stats['time_breakdown']['ar_model_forward_time'].append((ar_end - ar_start) * 1000)
                verification_stats['time_breakdown']['verification_strategy_time'].append((verify_end - verify_start) * 1000)

            chosen_dream.extend(chosen_dream_chunk)
            total_accept_len += accept_len  # Accumulate total accepted tokens

            # Store last output and input for KV cache management outside loop
            last_out = out
            last_q_chunk_ar_inp = q_chunk_ar_inp

            # Debug logging
            logger.debug(f"Accepted {accept_len} tokens in this chunk")

            # # ── merge + crop KV-cache to keep only accepted cells ────────
            # pkv_cls = out.past_key_values.__class__
            # if self.kv is None:
            #     merged = out.past_key_values
            # else:
            #     merged_layers = [
            #         (
            #             torch.cat([k0, k1], dim=2),
            #             torch.cat([v0, v1], dim=2),
            #         )
            #         for (k0, v0), (k1, v1) in zip(self.kv, out.past_key_values)
            #     ]
            #     merged = pkv_cls(merged_layers)

            # # Keep KV cache for all accepted tokens - 1 
            # # if some tokens are accepted
            # if accept_len > 0: 
            #     # leave last token to next iteration
            #     len_incr = accept_len
            # else:
            #     len_incr = 1
            #     # assert chosen_dream is length 1
            #     # assert len(chosen_dream) == 1, f"chosen_dream must be length 1, but is {len(chosen_dream)}"
            #     # just warn
            #     # logger.warning(f"chosen_dream must is {len(chosen_dream)}, and no tokens were accepted")
            
            # new_len = len(self.cache_q) + len_incr

            # trimmed = _crop_past_key_values(self.model, merged, new_len)
            # self.kv = trimmed if isinstance(trimmed, pkv_cls) else pkv_cls(trimmed)
            
            # # Update cache_q to match the KV cache
            # self.cache_q = self.cache_q[:new_len]
            # self.cache_q.extend(q_chunk_ar_inp[0, :len_incr].tolist())

            # stop at first rejection inside this chunk
            if accept_len < len(q_chunk[0]):
                break
                
            # Update i to move to next chunk
            i += batch_size

        # ── merge + crop KV-cache to keep only accepted cells (moved outside loop) ────────
        if last_out is not None:
            pkv_cls = last_out.past_key_values.__class__
            if self.kv is None:
                merged = last_out.past_key_values
            else:
                merged_layers = [
                    (
                        torch.cat([k0, k1], dim=2),
                        torch.cat([v0, v1], dim=2),
                    )
                    for (k0, v0), (k1, v1) in zip(self.kv, last_out.past_key_values)
                ]
                merged = pkv_cls(merged_layers)

            # Keep KV cache for all accepted tokens - 1 
            # if some tokens are accepted
            if total_accept_len > 0: 
                # leave last token to next iteration
                len_incr = total_accept_len
            else:
                len_incr = 1
                # assert chosen_dream is length 1
                # assert len(chosen_dream) == 1, f"chosen_dream must be length 1, but is {len(chosen_dream)}"
                # just warn
                # logger.warning(f"chosen_dream must is {len(chosen_dream)}, and no tokens were accepted")
            
            new_len = len(self.cache_q) + len_incr

            trimmed = _crop_past_key_values(self.model, merged, new_len)
            self.kv = trimmed if isinstance(trimmed, pkv_cls) else pkv_cls(trimmed)
            
            # Update cache_q to match the KV cache
            self.cache_q = self.cache_q[:new_len]
            if last_q_chunk_ar_inp is not None:
                self.cache_q.extend(last_q_chunk_ar_inp[0, :len_incr].tolist())

        return chosen_dream, total_accept_len

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
def assisted_block_diffusion_generate(
    *,
    dream_model,
    ar_model,
    input_ids: Tensor,
    attention_mask: Optional[Tensor],
    config: AssistedDiffusionConfig,
    dream_tokenizer,
    ar_tokenizer,
    ar_verifier: ARAssistant = None,
    rng: Optional[torch.Generator] = None,
) -> Union[Tensor, AssistedDiffusionOutput]:
    """
    Assisted diffusion generation with block KV caching.
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
        verifier = ARAssistant(ar_model, dream_tokenizer, ar_tokenizer)
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
    print(f"DEBUG: use_block_boundary_caching = {use_block_boundary_caching}")
    use_sliding_window_caching = getattr(config, 'use_sliding_window_caching', False)
    print(f"DEBUG: use_sliding_window_caching = {use_sliding_window_caching}")
    print(f"DEBUG: config.use_sliding_window_caching = {getattr(config, 'use_sliding_window_caching', 'NOT_FOUND')}")
    sliding_window_size = getattr(config, 'sliding_window_size', 128)

    while (seq == mask_id).any():
        logger.debug("================================================")
        logger.debug(f"diffusion_step: {diffusion_step}")
        verification_stats['total_steps'] += 1

        # Start timing for this step
        # compute sliding window start index
        sliding_window_start = max(0, last_non_mask-sliding_window_size)
        sliding_window_end = min(max_len, last_non_mask+sliding_window_size)
        
        # Dream forward (draft) with block caching
        diffusion_start_time = time.perf_counter()
        
        if diffusion_step == 0:
            # First step: reset block cache and save cache up to prompt
            for layer_idx in range(dream_model.config.num_hidden_layers):
                dream_model.model.layers[layer_idx].reset_block_cache(bsz, L, block_size)
            
            # Dream model forward pass timing
            dream_forward_start_time = time.perf_counter()
            out = dream_model(seq, None, tok_idx if tok_idx is not None else None,
                            use_block_diffusion=True,
                            use_full_query_attn=False,
                            max_length=max_len,
                            block_size=block_size,
                            save_cache=True,
                            clean_idx=sliding_window_start if use_sliding_window_caching else prompt_len)
            dream_forward_end_time = time.perf_counter()
            dream_forward_time = (dream_forward_end_time - dream_forward_start_time) * 1000  # Convert to ms
            
            # Logits has shape [1, max_len, vocab_size]
            logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], 1)
            
            # OLD: mask_pos = (seq[0] == mask_id).nonzero(as_tuple=True)[0]
            # Optimized mask position finding (first step)
            first_mask = (seq[0] == mask_id).nonzero(as_tuple=True)[0]
            if len(first_mask) > 0:
                start_pos = first_mask[0]
                remaining_seq = seq[0, start_pos:]
                relative_mask_pos = (remaining_seq == mask_id).nonzero(as_tuple=True)[0]
                mask_pos = relative_mask_pos + start_pos
            else:
                mask_pos = torch.tensor([], dtype=torch.long, device=seq.device)
            
            drafts = logits[0, mask_pos, :].argmax(-1)
        else:
            if use_block_boundary_caching:
                # GREEN = '\033[1;32m'
                # BOLD = '\033[1m'
                # RESET = '\033[0m'
                # print(f"{GREEN}{BOLD}Using block boundary caching{RESET}")    
                # Calculate last clean block boundary
                last_clean_block = ((last_non_mask + 1) // block_size) * block_size
                
                # Start from last clean block
                gen_seq = seq[:, last_clean_block:]  # Only the generation part from last clean block
                gen_tok_idx = tok_idx[:, last_clean_block:] if tok_idx is not None else None
                
                # Dream model forward pass timing
                dream_forward_start_time = time.perf_counter()
                out = dream_model(gen_seq, None, gen_tok_idx,
                                use_block_diffusion=True,
                                use_full_query_attn=False,
                                max_length=max_len,
                                block_size=block_size,
                                save_cache=True,
                                clean_idx=last_clean_block + 1)  # Save cache up to last clean block
                dream_forward_end_time = time.perf_counter()
                dream_forward_time = (dream_forward_end_time - dream_forward_start_time) * 1000  # Convert to ms
                
                # Logits has shape [1, remaining_len, vocab_size]
                logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], 1)
                # OLD: gen_mask_pos = (gen_seq[0] == mask_id).nonzero(as_tuple=True)[0]
                # Get mask positions relative to the generation part (optimized)
                first_mask = (gen_seq[0] == mask_id).nonzero(as_tuple=True)[0]
                if len(first_mask) > 0:
                    start_pos = first_mask[0]
                    remaining_seq = gen_seq[0, start_pos:]
                    relative_mask_pos = (remaining_seq == mask_id).nonzero(as_tuple=True)[0]
                    gen_mask_pos = relative_mask_pos + start_pos
                else:
                    gen_mask_pos = torch.tensor([], dtype=torch.long, device=gen_seq.device)
                drafts = logits[0, gen_mask_pos, :].argmax(-1)
                
                # Convert gen_mask_pos back to full sequence positions for updating
                mask_pos = gen_mask_pos + last_clean_block

                
            elif aggressive_caching:
                # Start from last clean token
                gen_seq = seq[:, last_non_mask:]  # Only the generation part from last clean token
                gen_tok_idx = tok_idx[:, last_non_mask:] if tok_idx is not None else None
                
                # Dream model forward pass timing
                dream_forward_start_time = time.perf_counter()
                out = dream_model(gen_seq, None, gen_tok_idx,
                                use_block_diffusion=True,
                                use_full_query_attn=False,
                                max_length=max_len,
                                block_size=block_size,
                                save_cache=True,
                                clean_idx=last_non_mask + 1)  # Save cache up to last clean token
                dream_forward_end_time = time.perf_counter()
                dream_forward_time = (dream_forward_end_time - dream_forward_start_time) * 1000  # Convert to ms
                
                # Logits has shape [1, remaining_len, vocab_size]
                logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], 1)
                # OLD: gen_mask_pos = (gen_seq[0] == mask_id).nonzero(as_tuple=True)[0]
                # Get mask positions relative to the generation part (optimized)
                first_mask = (gen_seq[0] == mask_id).nonzero(as_tuple=True)[0]
                if len(first_mask) > 0:
                    start_pos = first_mask[0]
                    remaining_seq = gen_seq[0, start_pos:]
                    relative_mask_pos = (remaining_seq == mask_id).nonzero(as_tuple=True)[0]
                    gen_mask_pos = relative_mask_pos + start_pos
                else:
                    gen_mask_pos = torch.tensor([], dtype=torch.long, device=gen_seq.device)
                drafts = logits[0, gen_mask_pos, :].argmax(-1)
                
                # Convert gen_mask_pos back to full sequence positions for updating
                mask_pos = gen_mask_pos + last_non_mask
            elif use_sliding_window_caching:
                gen_seq = seq[:, sliding_window_start:sliding_window_end]
                gen_tok_idx = tok_idx[:, sliding_window_start:sliding_window_end] if tok_idx is not None else None
                
                # Dream model forward pass timing
                dream_forward_start_time = time.perf_counter()
                out = dream_model(gen_seq, None, gen_tok_idx,
                                use_block_diffusion=True,
                                use_full_query_attn=False,
                                max_length=max_len,
                                block_size=block_size,
                                save_cache=True,
                                clean_idx=sliding_window_start)
                dream_forward_end_time = time.perf_counter()
                dream_forward_time = (dream_forward_end_time - dream_forward_start_time) * 1000  # Convert to ms
                # Logits has shape [1, remaining_len, vocab_size]
                logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
                # Get mask positions relative to the generation part
                # OLD: gen_mask_pos = (gen_seq[0] == mask_id).nonzero(as_tuple=True)[0]
                
                # Optimized: find first mask and then get all subsequent masks
                first_mask = (gen_seq[0] == mask_id).nonzero(as_tuple=True)[0]
                if len(first_mask) > 0:
                    # Only scan from the first mask position onwards
                    start_pos = first_mask[0]
                    remaining_seq = gen_seq[0, start_pos:]
                    relative_mask_pos = (remaining_seq == mask_id).nonzero(as_tuple=True)[0]
                    gen_mask_pos = relative_mask_pos + start_pos
                else:
                    gen_mask_pos = torch.tensor([], dtype=torch.long, device=gen_seq.device)
                
                drafts = logits[0, gen_mask_pos, :].argmax(-1)

                # Convert gen_mask_pos back to full sequence positions for updating
                mask_pos = gen_mask_pos + sliding_window_start
                
                
            else:
                # Original behavior: start from prompt_len
                gen_seq = seq[:, prompt_len:]  # Only the generation part
                gen_tok_idx = tok_idx[:, prompt_len:] if tok_idx is not None else None
                
                # Dream model forward pass timing
                dream_forward_start_time = time.perf_counter()
                out = dream_model(gen_seq, None, gen_tok_idx,
                                use_block_diffusion=True,
                                use_full_query_attn=False,
                                max_length=max_len,
                                block_size=block_size,
                                save_cache=False,
                                clean_idx=None)
                dream_forward_end_time = time.perf_counter()
                dream_forward_time = (dream_forward_end_time - dream_forward_start_time) * 1000  # Convert to ms
                
                # Logits has shape [1, max_len - prompt_len (i.e., gen_len), vocab_size]
                logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], 1)
                # OLD: gen_mask_pos = (gen_seq[0] == mask_id).nonzero(as_tuple=True)[0]
                # Get mask positions relative to the generation part (optimized)
                first_mask = (gen_seq[0] == mask_id).nonzero(as_tuple=True)[0]
                if len(first_mask) > 0:
                    start_pos = first_mask[0]
                    remaining_seq = gen_seq[0, start_pos:]
                    relative_mask_pos = (remaining_seq == mask_id).nonzero(as_tuple=True)[0]
                    gen_mask_pos = relative_mask_pos + start_pos
                else:
                    gen_mask_pos = torch.tensor([], dtype=torch.long, device=gen_seq.device)
                # Use gen_mask_pos directly since logits corresponds to gen_seq
                drafts = logits[0, gen_mask_pos, :].argmax(-1)
                
                # Convert gen_mask_pos back to full sequence positions for updating
                mask_pos = gen_mask_pos + prompt_len

        logger.debug(f"step {diffusion_step} masks={mask_pos.numel()}")

        # Verifier in batched chunks
        mask_pos_ls = mask_pos.tolist()
        drafts = drafts.to(dev)  # Ensure drafts are on the correct device

        dream_prefix = seq[0, :mask_pos_ls[0]] # up to the first masked token

        # End diffusion timing and start AR verification timing
        diffusion_end_time = time.perf_counter()
        diffusion_time = (diffusion_end_time - diffusion_start_time) * 1000  # Convert to ms
        
        # Calculate Dream model overhead (diffusion time minus Dream forward time)
        dream_overhead_time = diffusion_time - dream_forward_time
        # uncomment to print diffusion time
        # RED = '\033[1;31m'
        # BOLD = '\033[1m'
        # RESET = '\033[0m'
        # print(f"{RED}{BOLD}DEBUG: diffusion_time = {diffusion_time}{RESET}")
        
        # AR verification timing
        ar_verify_start_time = time.perf_counter()
        chosen_chunk, accept_len = verifier.batch_verify(
            dream_prefix, drafts,
            batch_size=config.verify_batch_size,
            rng=rng,
            verification_stats=verification_stats,
            sampling_strategy=config.sampling_strategy,
            confidence_threshold=config.confidence_threshold,
            acceptance_top_k=config.acceptance_top_k,
            top_p=config.top_p
        )
        ar_verify_end_time = time.perf_counter()
        ar_verify_time = (ar_verify_end_time - ar_verify_start_time) * 1000  # Convert to ms
        
        # Start overhead timing (rejection/acceptance handling)
        overhead_start_time = time.perf_counter()

        # Update sequence with verified tokens
        mismatch_idx = None
        tokens_updated = 0
        assert len(chosen_chunk) > 0, f"chosen_chunk is empty at step {diffusion_step}"
        # import pdb; pdb.set_trace()
        # for off, (p, dr, ch) in enumerate(zip(mask_pos_ls, drafts.tolist(), chosen_chunk)):
        #     if accept_len == 0:
        #         # If nothing was accepted, only unmask the mismatched token
        #         # update the first token
        #         seq[0, p] = dr
        #         tokens_updated += 1
        #         last_non_mask = p
        #         logger.debug(f"    slot {p}: draft={dream_tokenizer.decode([dr])} ({dr})"
        #                 f" → chosen={dream_tokenizer.decode([ch])} ({ch}) (mismatch)")
        #         print(f"\033[1;33mStep {diffusion_step}\033[0m - \033[1;36mDraft: {dream_tokenizer.decode([dr])} ({dr})\033[0m | \033[1;35mAR: {dream_tokenizer.decode([ch])} ({ch})\033[0m")

        #         # if dr is eos, update the sequence, then break off and set stopped to true
        #         if dr == eos_id:
        #             verification_stats['unmasked_tokens_per_step'].append(tokens_updated)
        #             print(f"Early stop (eos) at step {diffusion_step}")
        #             seq[seq == mask_id] = eos_id
        #             stopped = True
        #         break
        #     else:
        #         # If something was accepted, unmask all accepted tokens
        #         if ch != dr and mismatch_idx is None:
        #             mismatch_idx = off  
        #             break

        #         # In case of a match, update the sequence
        #         assert ch == dr, f"ch != dr at step {diffusion_step}, slot {p}"
        #         seq[0, p] = dr
        #         tokens_updated += 1
        #         last_non_mask = p
        #         logger.debug(f"    slot {p}: draft={dream_tokenizer.decode([dr])} ({dr})"
        #               f" → chosen={dream_tokenizer.decode([ch])} ({ch})")
                
        #         # if dr is eos, update the sequence, then break off and set stopped to true
        #         # if dr == eos_id:
        #         #     # seq[0, p] = ch
        #         #     # # set all tokens after p to eos
        #         #     # seq[0, p+1:] = eos_id
        #         #     # stopped = True
        #         #     verification_stats['unmasked_tokens_per_step'].append(tokens_updated)
        #         #     print(f"Early stop (eos) at step {diffusion_step}")
        #         #     seq[seq == mask_id] = eos_id
        #         #     stopped = True
        #         #     break

        #         if dr == eos_id:
        #             verification_stats['unmasked_tokens_per_step'].append(tokens_updated)
        #             print(f"Early stop (eos) at step {diffusion_step}")
        #             seq[seq == mask_id] = eos_id
        #             stopped = True
        #             break

        for off, (p, dr, ch) in enumerate(zip(mask_pos_ls, drafts.tolist(), chosen_chunk)):
            if off == 0 and ch != dr:
                # when the first token is mismatched, unmask the first token
                # update the first token
                seq[0, p] = dr
                tokens_updated += 1
                last_non_mask = p
                logger.debug(f"    slot {p}: draft={dream_tokenizer.decode([dr])} ({dr})"
                        f" → chosen={dream_tokenizer.decode([ch])} ({ch}) (mismatch)")
                print(f"\033[1;33mStep {diffusion_step}\033[0m - \033[1;36mDraft: {dream_tokenizer.decode([dr])} ({dr})\033[0m | \033[1;35mAR: {dream_tokenizer.decode([ch])} ({ch})\033[0m")

                # if dr is eos, update the sequence, then break off and set stopped to true
                if dr == eos_id:
                    verification_stats['unmasked_tokens_per_step'].append(tokens_updated)
                    print(f"Early stop (eos) at step {diffusion_step}")
                    seq[seq == mask_id] = eos_id
                    stopped = True
                break
            else:
                # If something was accepted, unmask all accepted tokens
                if ch != dr and mismatch_idx is None:
                    mismatch_idx = off 
                    break

                # In case of a match, update the sequence
                assert ch == dr, f"ch != dr at step {diffusion_step}, slot {p}"
                seq[0, p] = dr
                tokens_updated += 1
                last_non_mask = p
                logger.debug(f"    slot {p}: draft={dream_tokenizer.decode([dr])} ({dr})"
                      f" → chosen={dream_tokenizer.decode([ch])} ({ch})")
                
                # if dr is eos, update the sequence, then break off and set stopped to true
                # if dr == eos_id:
                #     # seq[0, p] = ch
                #     # # set all tokens after p to eos
                #     # seq[0, p+1:] = eos_id
                #     # stopped = True
                #     verification_stats['unmasked_tokens_per_step'].append(tokens_updated)
                #     print(f"Early stop (eos) at step {diffusion_step}")
                #     seq[seq == mask_id] = eos_id
                #     stopped = True
                #     break

                if dr == eos_id:
                    verification_stats['unmasked_tokens_per_step'].append(tokens_updated)
                    print(f"Early stop (eos) at step {diffusion_step}")
                    seq[seq == mask_id] = eos_id
                    stopped = True
                    break
                
        if not stopped:
            verification_stats['unmasked_tokens_per_step'].append(tokens_updated)

        # End overhead timing and record all timing data
        overhead_end_time = time.perf_counter()
        overhead_time = (overhead_end_time - overhead_start_time) * 1000  # Convert to ms
        
        # Record timing breakdown for this step
        if 'time_breakdown' not in verification_stats:
            verification_stats['time_breakdown'] = {
                'dream_model_forward_time': [],
                'dream_overhead_time': [],
                'diffusion_prediction_time': [],
                'ar_model_forward_time': [],
                'verification_strategy_time': [],
                'overhead_time': [],
                'total_step_time': []
            }
        verification_stats['time_breakdown']['dream_model_forward_time'].append(dream_forward_time)
        verification_stats['time_breakdown']['dream_overhead_time'].append(dream_overhead_time)
        verification_stats['time_breakdown']['diffusion_prediction_time'].append(diffusion_time)
        verification_stats['time_breakdown']['overhead_time'].append(overhead_time)
        
        # Get the separate timing breakdowns from verification_stats
        ar_model_time = 0.0
        verification_strategy_time = 0.0
        if 'ar_model_forward_time' in verification_stats['time_breakdown']:
            ar_model_time = sum(verification_stats['time_breakdown']['ar_model_forward_time'][-1:])  # Last step only
        if 'verification_strategy_time' in verification_stats['time_breakdown']:
            verification_strategy_time = sum(verification_stats['time_breakdown']['verification_strategy_time'][-1:])  # Last step only
        
        total_ar_time = ar_model_time + verification_strategy_time
        verification_stats['time_breakdown']['total_step_time'].append(diffusion_time + total_ar_time + overhead_time)

        # If there was a mismatch and nothing was accepted, continue to next diffusion step
        if mismatch_idx is not None and len(chosen_chunk) == 0:
            logger.debug(f"Mismatch: {dream_tokenizer.decode([drafts.tolist()[mismatch_idx]])}"
                  f" ({drafts.tolist()[mismatch_idx]}) → "
                  f"{dream_tokenizer.decode([chosen_chunk[mismatch_idx]])}"
                  f" ({chosen_chunk[mismatch_idx]})")
            continue  # new Dream forward
            
        diffusion_step += 1

        if history is not None:
            history.append(seq.clone())
        # exit if last unmasked token is eos (last_non_mask)
        if seq[0, last_non_mask] == eos_id:
            print(f"Last unmasked token is eos at step {diffusion_step}")
            break

        if stopped:
            print(f"Stopped at step {diffusion_step}")
            break

        # if config.early_stop and _early_stop_ok(
        #         seq, eos_id, mask_id, config.early_stop_consecutive):
        #     print(f"Early stop: at step {diffusion_step}")
        #     seq[seq == mask_id] = eos_id
        #     break

    # Calculate final acceptance rate
    if verification_stats['total_tokens'] > 0:
        verification_stats['acceptance_rate'] = (
            verification_stats['accepted_tokens'] / verification_stats['total_tokens']
        )
    verification_stats['diffusion_steps'] = diffusion_step

    if config.return_dict_in_generate:
        return AssistedDiffusionOutput(sequences=seq,
                                 history=tuple(history) if history else None,
                                 verification_stats=verification_stats)
    return seq

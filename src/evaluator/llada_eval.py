"""
Evaluator of Llada
"""

import torch
# from src.model.llada.generate import llada_generate_v2 as llada_generate

from src.model.block_cached_llada.generate import generate as llada_generate
from src.model.opt_llada.generate import generate as opt_llada_generate
from src.model.llada_v2.generate import generate as llada_v2_generate

from src.evaluator.llada_evaluator import GSM8KEval

class GSM8KLlada(GSM8KEval):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir, model, tokenizer, draft_model, draft_tokenizer)
    
        llada_config = self.config["llada"]
        # self.steps = llada_config["steps"]
        self.steps = llada_config.get("steps", None)
        self.tokens_per_step = llada_config.get("tokens_per_step", None)
    
        self.gen_length = llada_config["gen_length"]
        self.block_length = llada_config["block_length"]
        self.temperature = llada_config["temperature"]
        self.cfg_scale = llada_config["cfg_scale"]
        self.remasking = llada_config["remasking"]

        # Block diffusion parameters
        self.use_block_caching = llada_config.get("use_block_caching", False)
        self.use_full_query_attn = llada_config.get("use_full_query_attn", False)

        # Early stop parameters
        self.early_stop = llada_config.get("early_stop", False)
        # This is not used in LLaDA v2
        # self.early_stop_consecutive = llada_config.get("early_stop_consecutive", 5)
        self.eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None

        if self.use_block_caching:
            self.logger.info(f"Block caching enabled with block length={self.block_length}")
            if self.use_full_query_attn:
                self.logger.info(f"Using full query attention mode")
        
        if self.early_stop:
            # self.logger.info(f"Early stopping enabled with consecutive={self.early_stop_consecutive}")
            self.logger.info(f"Early stopping enabled")
            if self.eos_token_id is None:
                self.logger.warning("EOS token ID not found in tokenizer, early stopping will be disabled")
                self.early_stop = False
            else:
                self.logger.info(f"EOS token ID found in tokenizer, early stopping will be enabled")
        else:
            self.logger.info(f"Early stopping disabled")

    def __name__(self):
        return "GSM8KLlada"
    
    def generate(self, input_ids, attn_mask, variable_gen_length=None):
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        gen_length = variable_gen_length if variable_gen_length else self.gen_length
        if not self.steps:
            steps = gen_length // self.tokens_per_step
        else:
            steps = self.steps

        if self.block_length == -1:
            # If block_length is -1, use gen_length as block_length
            # This is for pure autoregressive generation
            block_length = gen_length
        else:
            # Ensure block_length is used as specified
            block_length = self.block_length
        start.record()
        # out = opt_llada_generate(
        #     self.model, 
        #     input_ids,
        #     steps=steps,
        #     gen_length=gen_length,
        #     block_length=block_length,
        #     temperature=self.temperature,
        #     cfg_scale=self.cfg_scale,
        #     remasking=self.remasking,

        #     # Block caching parameters
        #     use_block_caching=self.use_block_caching,
        #     use_full_query_attn=self.use_full_query_attn,

        #     # Early stop parameters
        #     early_stop=self.early_stop,
        #     early_stop_consecutive=self.early_stop_consecutive,
        #     eos_token_id=self.eos_token_id
        # )
        out = llada_v2_generate(
            self.model, 
            input_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=self.temperature,
            cfg_scale=self.cfg_scale,
            remasking=self.remasking,

            # Block caching parameters
            use_block_caching=self.use_block_caching,
            use_full_query_attn=self.use_full_query_attn,

            # Early stop parameters
            early_stop=self.early_stop,
            eos_token_id=self.eos_token_id
        )
        end.record()
        torch.cuda.synchronize()
        latency = start.elapsed_time(end)

        return out, latency
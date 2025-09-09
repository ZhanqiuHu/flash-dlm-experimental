"""
Base evaluator class for Llada evaluation tasks
"""

import torch
import os
from src.stage.base import Execute
from src.utils.utils import stop_sequences_criteria
from src.evaluator.llada.logger import EvaluationLogger

class BaseLladaEvaluator(Execute):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir)
        
        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer

        # Configure model generation settings
        self.model.generation_config.spec_sampling = self.config["eval"].get("spec_sampling", False)
        
        if self.draft_model is not None:
            self.draft_model.generation_config.num_assistant_tokens = self.config["spec_dec"]["assistant_tokens"]
            self.draft_model.generation_config.spec_sampling = False

        # Common generation parameters
        self.max_gen_toks = self.config["eval"]["max_gen_toks"]
        self.gen_until = ['<|eot_id|>', '<|start_header_id|>user<|end_header_id|>', 'Q:', '</s>', '<|im_end|>']

        # Initialize evaluation logger
        self.eval_logger = EvaluationLogger(os.path.join(self.run_dir, "eval_results.json"))

    def tokenize(self, prompt: str, truncation=False):
        """Common tokenization method"""
        encoding = self.tokenizer(
            prompt,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False
        )
        return encoding["input_ids"], encoding["attention_mask"]

    def generate(self, input_ids, attention_mask, variable_gen_length=None):
        """Base generation method that can be overridden by specific evaluators"""
        max_length = input_ids.shape[1] + (variable_gen_length if variable_gen_length else self.max_gen_toks)
        stop_criteria = stop_sequences_criteria(
            self.tokenizer, self.gen_until, input_ids.shape[1], input_ids.shape[0]
        )

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        if self.config["model"].get("spec_dec", False):
            start.record()
            out = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                attention_mask=attention_mask,
                do_sample=True,
                top_k=1,
                assistant_model=self.draft_model,
            )
            end.record()
        else:
            start.record()
            out = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                stopping_criteria=stop_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                attention_mask=attention_mask,
                do_sample=True,
                top_k=1,
                assistant_model=self.draft_model,
            )
            end.record()

        torch.cuda.synchronize()
        latency = start.elapsed_time(end)
        return out, latency

    def metric(self, model_pred, gt):
        """Base metric method to be implemented by specific evaluators"""
        raise NotImplementedError("Metric method must be implemented by specific evaluator")

    def run(self):
        """Base run method to be implemented by specific evaluators"""
        raise NotImplementedError("Run method must be implemented by specific evaluator") 
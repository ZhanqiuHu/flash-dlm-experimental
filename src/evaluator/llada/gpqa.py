"""
GPQA evaluator for Llada model
"""

import re
import torch
import os
from tqdm import tqdm
from src.evaluator.llada.base import BaseLladaEvaluator
from src.dataset.gpqa import GPQA
from src.model.llada_v2.generate import generate as llada_v2_generate

class GPQALlada(BaseLladaEvaluator):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir, model, tokenizer, draft_model, draft_tokenizer)
        
        # Load GPQA dataset
        self.datastage = GPQA(config_dir, tokenizer)
        self.trainset, self.testset = self.datastage.run()

        # LLaDA specific configs
        llada_config = self.config["llada"]
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
        self.eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None

        if self.use_block_caching:
            self.logger.info(f"Block caching enabled with block length={self.block_length}")
            if self.use_full_query_attn:
                self.logger.info(f"Using full query attention mode")
        
        if self.early_stop:
            self.logger.info(f"Early stopping enabled")
            if self.eos_token_id is None:
                self.logger.warning("EOS token ID not found in tokenizer, early stopping will be disabled")
                self.early_stop = False
            else:
                self.logger.info(f"EOS token ID found in tokenizer, early stopping will be enabled")
        else:
            self.logger.info(f"Early stopping disabled")

        # Create output directory for detailed results
        self.output_dir = os.path.join(self.run_dir, "detailed_results")
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = os.path.join(self.output_dir, "output.txt")
        self.output_f = open(self.output_file, "w")

    def __name__(self):
        return "GPQALlada"

    def metric(self, model_pred: str, gt: str):
        """
        Extract the "AnswerX" tag from the model's output text and compare
        against the ground truth label (also "Answer1"–"Answer4").
        """
        text = model_pred.strip()
        # Look for Answer1, Answer2, Answer3, or Answer4 (case‐insensitive)
        m = re.search(r'Answer\s*([1-4])', text, re.IGNORECASE)
        if m:
            pred = f"Answer{m.group(1)}"
        else:
            pred = ""

        correctness = (pred == gt)

        BOLD = "\033[1m"
        if not correctness:
            RED = "\033[91m"
            RESET = "\033[0m"
            print(f"{RED}❌  Pred: {BOLD}{pred}{RESET}{RED} != Gold: {BOLD}{gt}{RESET}")
            print(f"{RED}{BOLD}Generated text:\n {text}{RESET}")
        else:
            GREEN = "\033[92m"
            RESET = "\033[0m"
            print(f"{GREEN}✅  Pred: {BOLD}{pred}{RESET}{GREEN} == Gold: {BOLD}{gt}{RESET}")

        return correctness, pred, gt

    @torch.no_grad()
    def run(self):
        """Run the evaluation"""
        self.logger.info(f"Starting GPQA evaluation with {len(self.testset['dataset'])} examples")
        print(f"\033[92mEvaluation results will be saved in {self.run_dir}/eval_results.json\033[0m")
        print(f"\033[92mDetailed results will be saved in {self.output_dir}/output.txt\033[0m")
        
        correct = 0
        total = len(self.testset["dataset"])
        
        # Create progress bar
        pbar = tqdm(total=total, desc="Evaluating GPQA")
        
        for idx, (prompt, answer) in enumerate(zip(self.testset["dataset"], self.testset["label"])):
            # Tokenize input
            input_ids, attention_mask = self.tokenize(prompt)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Generate response using LLaDA v2
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            output_ids = llada_v2_generate(
                self.model, 
                input_ids,
                steps=self.steps if self.steps else self.gen_length // self.tokens_per_step,
                gen_length=self.gen_length,
                block_length=self.block_length if self.block_length != -1 else self.gen_length,
                temperature=self.temperature,
                cfg_scale=self.cfg_scale,
                remasking=self.remasking,
                use_block_caching=self.use_block_caching,
                use_full_query_attn=self.use_full_query_attn,
                early_stop=self.early_stop,
                eos_token_id=self.eos_token_id
            )
            end.record()
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)
            
            # Handle response exactly like dream implementation
            tok_list = output_ids.tolist()
            # Only take the generated part
            tok = tok_list[0][input_ids.shape[1]:]
            # Decode tokens
            dec_tok = self.tokenizer.decode(tok, skip_special_tokens=True)
            # Count tokens by re-tokenizing
            num_generated_tokens = len(self.tokenizer.encode(dec_tok))
            
            # Calculate metrics
            is_correct, pred, gt = self.metric(dec_tok, answer)
            correct += int(is_correct)
            
            # Log results
            self.eval_logger.log(
                prompt_index=idx,
                batch_size=1,
                steps=self.steps if self.steps else self.gen_length // self.tokens_per_step,
                max_gen_length=self.gen_length,
                block_length=self.block_length if self.block_length != -1 else self.gen_length,
                avg_latency_s=latency / 1000,  # Convert ms to seconds
                throughput_req_s=1000 / latency if latency > 0 else 0,  # Convert ms to req/s
                generated_token_len=num_generated_tokens,
                actual_generated_token_len=num_generated_tokens,
                input_token_len=len(input_ids[0]),
                prompt=prompt,
                response=dec_tok,
                answer=answer,
                correct=is_correct,
                avg_tokens_per_step=num_generated_tokens / (self.steps if self.steps else self.gen_length // self.tokens_per_step),
                early_stopping=self.early_stop,
                early_stop_consecutive=self.early_stop_consecutive if hasattr(self, 'early_stop_consecutive') else None,
                denoising_steps_dir=sample_dir if hasattr(self, 'enable_hook') and self.enable_hook else None,
                block_diffusion=self.use_block_caching,
                block_size=self.block_length if self.block_length != -1 else self.gen_length,
                full_query_attention=self.use_full_query_attn if self.use_block_caching else None
            )
            
            # Save detailed output
            self.output_f.write(f"\nExample {idx + 1}/{total}\n")
            self.output_f.write(f"Prompt: {prompt}\n")
            self.output_f.write(f"Response: {dec_tok}\n")
            self.output_f.write(f"Answer: {answer}\n")
            self.output_f.write(f"Correct: {is_correct}\n")
            self.output_f.write("-" * 80 + "\n")
            self.output_f.flush()
            
            # Update progress bar
            pbar.update(1)
            accuracy = (correct / (idx + 1)) * 100
            pbar.set_postfix({"Accuracy": f"{accuracy:.2f}%"})
            
            # Print progress every 10 examples
            if (idx + 1) % 10 == 0:
                self.logger.info(f"Processed {idx + 1}/{total} examples. Current accuracy: {accuracy:.2f}%")
        
        # Close progress bar and output file
        pbar.close()
        self.output_f.close()
        
        # Calculate final accuracy
        final_accuracy = (correct / total) * 100
        self.logger.info(f"Evaluation complete. Final accuracy: {final_accuracy:.2f}%")
        
        return {"accuracy": final_accuracy} 
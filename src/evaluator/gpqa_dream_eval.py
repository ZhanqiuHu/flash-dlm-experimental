"""
GPQA Dream evaluator
"""

import torch
import re
import os
import json
import time
import wandb
from tqdm import tqdm
from datetime import datetime
from src.stage.base import Execute
from src.dataset.gpqa import GPQA
from src.evaluator.evaluator import EvaluationLogger
from src.utils.utils import stop_sequences_criteria

# Standard fg colors
BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"
PURPLE = "\033[35m"
ORANGE = "\033[33m"
PINK = "\033[31m"

# Bright / "light" variants
BRIGHT_BLACK   = "\033[90m"  # often used for DEBUG
BRIGHT_RED     = "\033[91m"
BRIGHT_GREEN   = "\033[92m"
BRIGHT_YELLOW  = "\033[93m"
BRIGHT_BLUE    = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN    = "\033[96m"
BRIGHT_WHITE   = "\033[97m"

# Styles
BOLD      = "\033[1m"
DIM       = "\033[2m"
UNDERLINE = "\033[4m"
RESET     = "\033[0m"

class GPQAEval(Execute):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        self.model.generation_config.spec_sampling = self.config["eval"].get("spec_sampling", False)

        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer

        if self.draft_model is not None:
            self.draft_model.generation_config.num_assistant_tokens = self.config["spec_dec"]["assistant_tokens"]
            self.draft_model.generation_config.spec_sampling = False

        # prepare dataset
        self.datastage = GPQA(config_dir, tokenizer)
        self.trainset, self.testset = self.datastage.run()

        # condition for end of generation
        self.max_gen_toks = self.config["eval"]["max_gen_toks"]
        self.gen_until = ['<|eot_id|>', '<|start_header_id|>user<|end_header_id|>', 'Q:', '</s>', '<|im_end|>']

    def __name__(self):
        return "GPQAEval"
    
    def tokenize(self, prompt:str, truncation=False):
        encoding = self.tokenizer(
            prompt,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False
        )

        return encoding["input_ids"], encoding["attention_mask"]
    
    def generate(self, input_ids, attention_mask):
        max_length = input_ids.shape[1] + self.max_gen_toks

        stop_criteria = stop_sequences_criteria(
            self.tokenizer, self.gen_until, input_ids.shape[1], input_ids.shape[0]
        )

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)        
        
        if self.config["model"]["spec_dec"]:
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
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)
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
        else:
            GREEN = "\033[92m"
            RESET = "\033[0m"
            print(f"{GREEN}✅  Pred: {BOLD}{pred}{RESET}{GREEN} == Gold: {BOLD}{gt}{RESET}")

        return correctness, pred, gt


    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        output = []
        latency = []
        total_tokens = 0  # Track total generated tokens
        eval_logger = EvaluationLogger(os.path.join(self.run_dir, "eval_results.json"))
        print(f"\033[92mEvaluation results will be saved in {self.run_dir}/eval_results.json\033[0m")
        self.model.eval()

        # Create directory for denoising steps
        denoising_dir = os.path.join(self.run_dir, "denoising_steps")
        if self.enable_hook:
            os.makedirs(denoising_dir, exist_ok=True)
            
        # Create directory for attention weights if enabled
        if self.save_attention_weights and self.attention_weights_path:
            os.makedirs(self.attention_weights_path, exist_ok=True)
            self.logger.info(f"Attention weights will be saved to {self.attention_weights_path}")

        # export the generated output
        output_file = os.path.join(self.run_dir, "output.txt")
        output_f = open(output_file, "w")

        # Subsample the dataset if enabled
        if self.subsample:
            import random
            random.seed(self.subsample_seed)
            dataset_size = len(self.testset["dataset"])
            
            # Check if subsample_size is a list of indices
            if isinstance(self.subsample_size, list):
                # Use the provided indices directly
                subsample_indices = self.subsample_size
                subsample_size = len(subsample_indices)
                self.logger.info(f"Using specific indices for dataset: {subsample_indices}")
            else:
                # Random subsampling as before
                subsample_size = min(self.subsample_size, dataset_size)
                subsample_indices = random.sample(range(dataset_size), subsample_size)
                subsample_indices.sort()  # Sort for reproducibility
                self.logger.info(f"Subsampling {subsample_size} samples from dataset of size {dataset_size} with seed {self.subsample_seed}")
                self.logger.info(f"Subsample indices: {subsample_indices}")
            
            # Create subsampled dataset
            subsampled_dataset = [self.testset["dataset"][i] for i in subsample_indices]
            subsampled_labels = [self.testset["label"][i] for i in subsample_indices]
            
            # Create a dictionary with the subsampled data
            subsampled_testset = {
                "dataset": subsampled_dataset,
                "label": subsampled_labels,
                "original_indices": subsample_indices  # Store the original indices
            }
            
            # Use the subsampled dataset
            dataset_to_use = subsampled_testset
        else:
            dataset_to_use = self.testset

        pbar = tqdm(dataset_to_use["dataset"])
        for idx, sample in enumerate(pbar):
            # Create directory for this sample's steps
            if self.enable_hook:
                sample_dir = os.path.join(denoising_dir, f"sample_{idx:05d}")
                os.makedirs(sample_dir, exist_ok=True)
            else:
                sample_dir = None
                
            # Create directory for this sample's attention weights if enabled
            if self.save_attention_weights and self.attention_weights_path:
                sample_attn_dir = os.path.join(self.attention_weights_path, f"sample_{idx:05d}")
                os.makedirs(sample_attn_dir, exist_ok=True)
            
            if not self.config["dataset"]["apply_chat_template"]:
                sample = sample[0]["content"]

            input_ids, attn_mask = self.tokenize(sample)
            
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)

            # label context
            gt = self.testset["label"][idx]

            # generate
            tok, lat = self.generate(input_ids, attn_mask)
            tok_list = tok.tolist()

            # decoded tokens
            tok = tok_list[0][input_ids.shape[1] :]
            # decode tokens
            dec_tok = self.tokenizer.decode(tok, skip_special_tokens=True)
            # Count tokens by re-tokenizing
            num_generated_tokens = len(self.tokenizer.encode(dec_tok))
            total_tokens += num_generated_tokens  # Add to total

            # correctness
            correctness, pred, gt = self.metric(dec_tok, gt)

            # if wrong, print out the generated text
            if not correctness:
                print(f"{RED}{BOLD}Generated text:\n {dec_tok}{RESET}")

            output.append(int(correctness))
            latency.append(lat)

            acc = sum(output) / len(output)
            pbar.set_description(f"Accuracy: {acc:.4f} | latency = {lat:.4f} | tokens = {num_generated_tokens}")

            out_str = f"Test ID = [{idx}] | Correctness = {correctness} |  GT=[{gt}] | Extracted ANS = {pred} \n\n Predicted = {dec_tok}\nGenerated Tokens = {num_generated_tokens}"
            self.logger.info(out_str)
            output_f.write(out_str + "\n")
            output_f.flush()

            # Log to wandb only if wandb_flag is True
            if self.wandb_flag:
                wandb.log({
                    "accuracy": acc,
                    "latency": lat,
                    "correctness": correctness,
                    "prediction": pred,
                    "ground_truth": gt,
                    "generated_text": dec_tok,
                    "num_generated_tokens": num_generated_tokens,
                    "tokens_per_second": num_generated_tokens / lat if lat > 0 else 0
                })

            # save every 100 prompts
            if idx % 100 == 0:
                eval_logger.save()
        
        eval_logger.save()

        avg = sum(output) / len(self.testset["dataset"])
        avg_lat = sum(latency) / len(latency)
        avg_tokens = total_tokens / len(dataset_to_use["dataset"])
        avg_tokens_per_second = total_tokens / sum(latency) if sum(latency) > 0 else 0

        self.logger.info(f"Average Score (exact match) = {avg:.4f} | average latency = {avg_lat:.4f}s")
        self.logger.info(f"Average Generated Tokens = {avg_tokens:.2f} | Average Tokens/Second = {avg_tokens_per_second:.2f}")
        output_f.close()
        return output

class GPQADream(GPQAEval):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir, model, tokenizer, draft_model, draft_tokenizer)
    
        dream_config = self.config["dream"]
        self.temperature = dream_config["temperature"]
        self.top_p = dream_config["top_p"]
        self.steps = dream_config["steps"]
        self.alg = dream_config["alg"]
        self.alg_temp = dream_config["alg_temp"]

        # NEW: enable_hook
        self.enable_hook = dream_config.get("enable_hook", False)

        # NEW: save attention weights
        self.save_attention_weights = dream_config.get("save_attention_weights", False)
        self.attention_weights_path = dream_config.get("attention_weights_path", None)
        
        # NEW: subsampling parameters
        self.subsample = dream_config.get("subsample", False)
        self.subsample_size = dream_config.get("subsample_size", 10)
        self.subsample_seed = dream_config.get("subsample_seed", 42)

        # NEW: block diffusion
        self.use_block_diffusion = dream_config.get("use_block_diffusion", False)
        self.use_full_query_attn = dream_config.get("use_full_query_attn", False)
        self.block_size = dream_config.get("block_size", 256)

        # NEW: early stopping
        self.early_stop = dream_config.get("early_stop", False)
        self.early_stop_consecutive = dream_config.get("early_stop_consecutive", 5)

        # NEW: confidence-based adaptive unmasking
        self.confidence_based_adaptive_unmasking = dream_config.get("confidence_based_adaptive_unmasking", False)
        self.decay_algorithm = dream_config.get("decay_algorithm", "exp_unmasked_ratio")
        self.decay_params = dream_config.get("decay_params", {
            "alpha": 1.0,  # base decay rate
            "gamma": 1.0,  # controls how fast alpha decays with unmasking
        })

        # print out all the hyperparameters
        self.logger.info(f"Hyperparameters: {dream_config}")

        if self.use_block_diffusion:
            self.logger.info(f"\033[92mBlock diffusion enabled with block size={self.block_size}\033[0m")
            if self.use_full_query_attn:
                self.logger.info(f"Using full query attention mode")
        else:
            self.logger.warning(f"\033[91mBlock diffusion is disabled. Inference will be very slow. This is not recommended for GPQA.\033[0m")
        
        if self.early_stop:
            self.logger.info(f"\033[92mEarly stopping enabled with consecutive={self.early_stop_consecutive}\033[0m")
        else:
            self.logger.warning(f"\033[91mEarly stopping is disabled. This is not recommended for GPQA.\033[0m")

        if self.confidence_based_adaptive_unmasking:
            self.logger.info(f"Confidence-based adaptive unmasking enabled with decay algorithm={self.decay_algorithm}")
            self.logger.info(f"Decay parameters: {self.decay_params}")

        # For testing/debugging, ensure steps is at least 1
        if self.steps < 1:
            self.logger.warning(f"Steps value {self.steps} is too low, setting to 1")
            self.steps = 1

    def __name__(self):
        return "GPQADream"

    def _track_denoising_step(self, step, x, logits, mask_index, t, s, p_transfer, num_transfer):
        """Track information about a denoising step
        
        Args:
            step (int): Current denoising step number (0 to steps-1)
            x (torch.Tensor): Current token sequence [batch_size, seq_len]
            logits (torch.Tensor): Model's logits for each position [batch_size, seq_len, vocab_size]
            mask_index (torch.Tensor): Boolean mask indicating masked positions [batch_size, seq_len]
            t (float): Current time step in the noise schedule
            s (float): Next time step in the noise schedule
            p_transfer (float): Probability of transferring tokens at this step
            num_transfer (int): Number of tokens being transferred at this step
            
        Returns:
            dict: Dictionary containing tracked information for this step
        """
        # Get top-k tokens and their logits for each masked position
        k = 20  # Number of top tokens to track
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        
        # Apply softmax to normalize the top-k logits
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        
        # Convert tensors to float32 before converting to numpy arrays for JSON serialization
        return {
            "step": step,
            "current_tokens": x.clone().detach().cpu().numpy().tolist(),  # Current sequence state
            "masked_positions": mask_index.clone().detach().cpu().numpy().tolist(),  # Which positions are masked
            "logits": None,  # Skip saving full logits in JSON since we're saving as .pt
            "top_k_tokens": top_k_indices.clone().detach().cpu().numpy().tolist(),  # Top k token indices
            "top_k_logits": top_k_logits.clone().detach().cpu().float().numpy().tolist(),  # Raw logits for top k tokens
            "top_k_probs": top_k_probs.clone().detach().cpu().float().numpy().tolist(),  # Normalized probabilities for top k tokens
            "time_step": float(t.item()),  # Current time in noise schedule
            "next_time_step": float(s.item()),  # Next time in noise schedule
            "transfer_probability": float(p_transfer),  # Probability of token transfer
            "num_tokens_transferred": int(num_transfer)  # Number of tokens being transferred
        }

    def generate(self, input_ids, attention_mask, sample_dir=None):
        def generation_tokens_hook(step, x, logits):
            """Hook function called at each denoising step"""
            if step is not None:  # Not the initial step
                # Calculate noise schedule parameters
                timesteps = torch.linspace(1, 1e-3, self.steps + 1, device=x.device)
                t = timesteps[step]
                s = timesteps[step + 1]
                p_transfer = 1 - s/t if step < self.steps - 1 else 1

                # Get mask for current step
                mask_index = torch.ones_like(x, dtype=torch.bool)
                mask_index[:, :input_ids.shape[1]] = False

                # Track denoising step
                step_info = self._track_denoising_step(
                    step, x, logits, mask_index, t, s, p_transfer, 
                    num_transfer=int(p_transfer * mask_index.sum().item())
                )

                # Save step info
                if self.enable_hook:
                    step_file = os.path.join(sample_dir, f"step_{step:03d}.json")
                    with open(step_file, "w") as f:
                        json.dump(step_info, f, indent=2)

            return x

        # Set the hook function based on enable_hook flag
        hook_func = generation_tokens_hook if self.enable_hook else (lambda step, x, logits: x)

        # Common parameters for diffusion generation
        generation_params = {
            "inputs": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_gen_toks,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "steps": self.steps,
            "alg": self.alg,
            "alg_temp": self.alg_temp,
            "generation_tokens_hook_func": hook_func,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True
        }
        
        # Add block diffusion parameters if enabled
        if self.use_block_diffusion:
            generation_params.update({
                "use_block_diffusion": True,
                "block_size": self.block_size,
                "use_full_query_attn": self.use_full_query_attn,
                "max_length": None  # if None, prompt_len + max_new_tokens will be used
            })
        
        # Add early stopping parameters if enabled
        if self.early_stop:
            generation_params.update({
                "early_stop": True,
                "early_stop_consecutive": self.early_stop_consecutive
            })
        
        # Add confidence-based adaptive unmasking parameters if enabled
        if self.confidence_based_adaptive_unmasking:
            generation_params.update({
                "confidence_based_adaptive_unmasking": True,
                "decay_algorithm": self.decay_algorithm,
                "decay_params": self.decay_params
            })
        
        # Add attention weight saving parameters if enabled
        if self.save_attention_weights and self.attention_weights_path:
            sample_attn_dir = os.path.join(self.attention_weights_path, os.path.basename(sample_dir))
            os.makedirs(sample_attn_dir, exist_ok=True)
            generation_params.update({
                "save_attention_weights": True,
                "attention_weights_path": sample_attn_dir
            })

        # Start time
        start_time = time.time()

        # Generate response
        response = self.model.diffusion_generate(**generation_params)
        end_time = time.time()
        lat = end_time - start_time

        return response, lat

    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        output = []
        latency = []
        total_tokens = 0  # Track total generated tokens
        eval_logger = EvaluationLogger(os.path.join(self.run_dir, "eval_results.json"))
        print(f"\033[92mEvaluation results will be saved in {self.run_dir}/eval_results.json\033[0m")
        self.model.eval()

        # Create directory for denoising steps
        denoising_dir = os.path.join(self.run_dir, "denoising_steps")
        if self.enable_hook:
            os.makedirs(denoising_dir, exist_ok=True)
            
        # Create directory for attention weights if enabled
        if self.save_attention_weights and self.attention_weights_path:
            os.makedirs(self.attention_weights_path, exist_ok=True)
            self.logger.info(f"Attention weights will be saved to {self.attention_weights_path}")

        # export the generated output
        output_file = os.path.join(self.run_dir, "output.txt")
        output_f = open(output_file, "w")

        # Subsample the dataset if enabled
        if self.subsample:
            import random
            random.seed(self.subsample_seed)
            dataset_size = len(self.testset["dataset"])
            
            # Check if subsample_size is a list of indices
            if isinstance(self.subsample_size, list):
                # Use the provided indices directly
                subsample_indices = self.subsample_size
                subsample_size = len(subsample_indices)
                self.logger.info(f"Using specific indices for dataset: {subsample_indices}")
            else:
                # Random subsampling as before
                subsample_size = min(self.subsample_size, dataset_size)
                subsample_indices = random.sample(range(dataset_size), subsample_size)
                subsample_indices.sort()  # Sort for reproducibility
                self.logger.info(f"Subsampling {subsample_size} samples from dataset of size {dataset_size} with seed {self.subsample_seed}")
                self.logger.info(f"Subsample indices: {subsample_indices}")
            
            # Create subsampled dataset
            subsampled_dataset = [self.testset["dataset"][i] for i in subsample_indices]
            subsampled_labels = [self.testset["label"][i] for i in subsample_indices]
            
            # Create a dictionary with the subsampled data
            subsampled_testset = {
                "dataset": subsampled_dataset,
                "label": subsampled_labels,
                "original_indices": subsample_indices  # Store the original indices
            }
            
            # Use the subsampled dataset
            dataset_to_use = subsampled_testset
        else:
            dataset_to_use = self.testset

        pbar = tqdm(dataset_to_use["dataset"])
        for idx, sample in enumerate(pbar):
            # Create directory for this sample's steps
            if self.enable_hook:
                sample_dir = os.path.join(denoising_dir, f"sample_{idx:05d}")
                os.makedirs(sample_dir, exist_ok=True)
            else:
                sample_dir = None
                
            # Create directory for this sample's attention weights if enabled
            if self.save_attention_weights and self.attention_weights_path:
                sample_attn_dir = os.path.join(self.attention_weights_path, f"sample_{idx:05d}")
                os.makedirs(sample_attn_dir, exist_ok=True)
            
            if not self.config["dataset"]["apply_chat_template"]:
                sample = sample[0]["content"]

            input_ids, attn_mask = self.tokenize(sample)
            
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)

            # label context
            gt = dataset_to_use["label"][idx]

            # generate
            tok, lat = self.generate(input_ids, attn_mask, sample_dir)
            tok_list = tok.tolist()

            # decoded tokens
            tok = tok_list[0][input_ids.shape[1] :]
            # decode tokens
            dec_tok = self.tokenizer.decode(tok, skip_special_tokens=True)
            # Count tokens by re-tokenizing
            num_generated_tokens = len(self.tokenizer.encode(dec_tok))
            total_tokens += num_generated_tokens  # Add to total

            # correctness
            correctness, pred, gt = self.metric(dec_tok, gt)
            if not correctness:
                print(f"{RED}{BOLD}Generated answer:\n {dec_tok}{RESET}")

            output.append(int(correctness))
            latency.append(lat)

            acc = sum(output) / len(output)
            pbar.set_description(f"Accuracy: {acc:.4f} | latency = {lat:.4f} | tokens = {num_generated_tokens}")

            out_str = f"Test ID = [{idx}] | Correctness = {correctness} |  GT=[{gt}] | Extracted ANS = {pred} \n\n Predicted = {dec_tok}\nGenerated Tokens = {num_generated_tokens}"
            self.logger.info(out_str)
            output_f.write(out_str + "\n")
            output_f.flush()

            # Log to wandb only if wandb_flag is True
            if self.wandb_flag:
                wandb.log({
                    "accuracy": acc,
                    "latency": lat,
                    "correctness": correctness,
                    "prediction": pred,
                    "ground_truth": gt,
                    "generated_text": dec_tok,
                    "num_generated_tokens": num_generated_tokens,
                    "tokens_per_second": num_generated_tokens / lat if lat > 0 else 0
                })
        
        avg = sum(output) / len(dataset_to_use["dataset"])
        avg_lat = sum(latency) / len(latency)
        avg_tokens = total_tokens / len(dataset_to_use["dataset"])
        avg_tokens_per_second = total_tokens / sum(latency) if sum(latency) > 0 else 0

        self.logger.info(f"Average Score (exact match) = {avg:.4f} | average latency = {avg_lat:.4f}s")
        self.logger.info(f"Average Generated Tokens = {avg_tokens:.2f} | Average Tokens/Second = {avg_tokens_per_second:.2f}")
        output_f.close()
        return output 
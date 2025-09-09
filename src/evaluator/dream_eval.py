"""
Evaluator of Dream
"""

import torch
import re
import os
import json
import time
import wandb
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from src.stage.base import Execute
from src.dataset.gsm8k import GSM8K
from src.evaluator.evaluator import GSM8KEval, MMLUProEval, PiQAEval, OpenbookQAEval, HellaswagEval, WinoGrandeEval

# COLOR CONSTANTS
YELLOW = "\033[93m"
PURPLE = "\033[95m"
RED = "\033[91m"
BLUE = "\033[94m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"

class EvaluationLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.logs = []
    
    def log(self, item):
        item["Timestamp"] = datetime.now().isoformat()
        self.logs.append(item)
        self._write_logs()
    
    def _write_logs(self):
        with open(self.log_file, "w") as f:
            json.dump(self.logs, f, indent=2)
    
    def save(self):
        """Save the current logs to file"""
        self._write_logs() 

class GSM8KDream(GSM8KEval):
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
            # print in green color
            self.logger.info(f"\033[92mBlock diffusion enabled with block size={self.block_size}\033[0m")
            if self.use_full_query_attn:
                self.logger.info(f"Using full query attention mode")
        else:
            # Warn using in capital red color
            self.logger.warning(f"\033[91mBlock diffusion is disabled. Inference will be very slow. This is not recommended for GSM8K.\033[0m")
        
        if self.early_stop:
            # print in green color
            self.logger.info(f"\033[92mEarly stopping enabled with consecutive={self.early_stop_consecutive}\033[0m")
        else:
            self.logger.warning(f"\033[91mEarly stopping is disabled. This is not recommended for GSM8K.\033[0m")

        if self.confidence_based_adaptive_unmasking:
            self.logger.info(f"Confidence-based adaptive unmasking enabled with decay algorithm={self.decay_algorithm}")
            self.logger.info(f"Decay parameters: {self.decay_params}")

        # For testing/debugging, ensure steps is at least 1
        if self.steps < 1:
            self.logger.warning(f"Steps value {self.steps} is too low, setting to 1")
            self.steps = 1

    def __name__(self):
        return "GSM8KDream"

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

    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        output = []
        latency = []
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
                
            # Create directory for this sample's attention weights if enabled
            if self.save_attention_weights and self.attention_weights_path:
                sample_attn_dir = os.path.join(self.attention_weights_path, f"sample_{idx:05d}")
                os.makedirs(sample_attn_dir, exist_ok=True)
            
            def generation_tokens_hook(step, x, logits):
                """Hook function called at each denoising step
                
                This function is called by the model's diffusion_generate method at each step
                of the denoising process. It allows us to track:
                1. The current state of the sequence
                2. The model's predictions (logits) for each position
                3. The noise schedule parameters
                4. The token transfer process
                
                Args:
                    step (int): Current denoising step (None for initial step)
                    x (torch.Tensor): Current token sequence
                    logits (torch.Tensor): Model's logits for each position
                
                Returns:
                    torch.Tensor: The current sequence (unchanged)
                """
                if step is not None:  # Not the initial step
                    # Calculate noise schedule parameters
                    # t: current time step (starts at 1, decreases to eps)
                    # s: next time step
                    # p_transfer: probability of transferring tokens (1 - s/t)
                    timesteps = torch.linspace(1, 1e-3, self.steps + 1, device=x.device)
                    t = timesteps[step]
                    s = timesteps[step + 1]
                    p_transfer = 1 - s/t if step < self.steps - 1 else 1
                    
                    # Save full logits as a separate .pt file
                    logits_path = os.path.join(sample_dir, f"step_{step:03d}_logits.pt")
                    torch.save(logits.cpu(), logits_path)
                    
                    # Track information about this step and save immediately
                    if self.enable_hook:
                        step_info = self._track_denoising_step(
                            step=step,
                            x=x,
                            logits=logits,
                            mask_index=(x == self.model.config.mask_token_id),
                            t=t,
                            s=s,
                            p_transfer=p_transfer,
                            num_transfer=(x == self.model.config.mask_token_id).sum().item()
                        )
                    
                    # Save step info immediately to disk
                    step_file = os.path.join(sample_dir, f"step_{step:03d}.json")
                    with open(step_file, "w") as f_step:
                        json.dump(step_info, f_step)
                return x

            # Tokenize input
            inputs = self.tokenizer(sample, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)

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

            # Decode response
            dec_tok = self.tokenizer.decode(response[0], skip_special_tokens=True)
            
            # Calculate actual generated token length (excluding input tokens and special tokens)
            input_token_length = len(self.tokenizer.encode(sample))
            # Decode and re-encode to get clean token count
            decoded_text = self.tokenizer.decode(response[0], skip_special_tokens=True)
            clean_tokens = self.tokenizer.encode(decoded_text, add_special_tokens=False)
            actual_generated_token_length = len(clean_tokens) - input_token_length
            
            # Get ground truth
            gt = dataset_to_use["label"][idx]
            
            # Evaluate correctness
            correctness = self.metric(dec_tok, gt)
            output.append(correctness)
            latency.append(lat)


            # Calculate running averages
            running_acc = sum(output) / len(output)
            running_lat = sum(latency) / len(latency)
            

            # Log results
            log_item = {
                "Prompt Index": idx,
                "Batch Size": 1,
                "Steps": self.steps,
                "Max Gen Length": self.max_gen_toks,
                "Block Length": self.block_size if self.use_block_diffusion else None,
                "Avg Latency (s)": lat,
                "Throughput (req/s)": 1/lat if lat > 0 else 0,
                "Generated Token Length": len(response[0]),
                "Actual Generated Token Length": actual_generated_token_length,
                "Input Token Length": len(self.tokenizer.encode(sample)),
                "Prompt": sample,
                "Response": dec_tok,
                "Answer": gt,
                "Correct": correctness,
                "Avg Tokens per Step": len(response[0])/self.steps if self.steps > 0 else 1,
                "Early Stopping": self.early_stop,
                "Early Stop Consecutive": self.early_stop_consecutive,
                "Denoising Steps Directory": sample_dir if self.enable_hook else None,
                "Block Diffusion": self.use_block_diffusion,
                "Block Size": self.block_size if self.use_block_diffusion else None,
                "Full Query Attention": self.use_full_query_attn if self.use_block_diffusion else None
            }
            eval_logger.log(log_item)

            # Calculate accuracy and update progress bar
            acc = sum(output) / len(output)
            # Calculate average latency
            avg_lat = sum(latency) / len(latency)
            # Update progress bar with accuracy, generation length, latency and correctness
            pbar.set_description(f"Acc: {acc:.4f} | Gen Len: {actual_generated_token_length}/{self.max_gen_toks} | Lat: {lat:.4f}s (avg: {avg_lat:.4f}s) | Correct: {correctness}")
            
            # save the generated text
            out_str = f"Test ID = [{idx}] | [{correctness}] \n{dec_tok}"
            print(out_str, file=output_f)

            log_info = {
                "Accuracy": running_acc,
                "Latency": lat,
                "Correct": correctness,
                "Token Length": len(response[0]),
                "Input Token Length": len(self.tokenizer.encode(sample)),
                "Generated Token Length": len(response[0]),
                "Avg Tokens Per Step": len(response[0])/self.steps if self.steps > 0 else 1,
                "Avg Latency": running_lat,
                "Top K Size": 20,  # Added to track top-k size
                "Block Diffusion": self.use_block_diffusion,
                "Block Size": self.block_size if self.use_block_diffusion else None
            }
            if self.enable_hook:
                with open(os.path.join(sample_dir, "log_info.json"), "w") as f_log:
                    json.dump(log_info, f_log)

            if self.wandb_flag:
                wandb.log(log_info)
            
            # save the evaluation information also in sample_dir
            if self.enable_hook:    
                with open(os.path.join(sample_dir, "log_info.json"), "w") as f_log:
                    json.dump(log_info, f_log)

            # save every 100 prompts
            if idx % 100 == 0:
                eval_logger.save()
                self.logger.info(f"Current Index: {idx}")
                self.logger.info(f"Avg Acc: {running_acc:.4f} | Avg Lat: {running_lat:.4f}s | Lat: {lat:.2f}s")
        
        eval_logger.save()

        output_f.close()
        avg = sum(output) / len(self.testset["dataset"])
        avg_lat = sum(latency) / len(latency)

        self.logger.info(f"Average Score (exact match) = {avg:.2f} | average latency = {avg_lat:.2f}")
        return output, latency

class MMLUProDream(MMLUProEval):
    def __init__(self, 
        config_dir, 
        model, 
        tokenizer, 
        draft_model=None, 
        draft_tokenizer=None,
        subject_filter=None
    ):
        super().__init__(config_dir, model, tokenizer, draft_model, draft_tokenizer)
        if subject_filter:
            self.testset = {subject: self.testset[subject] for subject in subject_filter}
            print(f"Filtered testset to run only on {self.testset.keys()}")

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
        
        # Log configuration information
        if not self.enable_hook:
            self.logger.info(f"Generation tokens hook disabled")
            
        if self.use_block_diffusion:
            # print in green color
            self.logger.info(f"\033[92mBlock diffusion enabled with block size={self.block_size}\033[0m")
            if self.use_full_query_attn:
                self.logger.info(f"Using full query attention mode")
        else:
            # Warn using in capital red color
            self.logger.warning(f"\033[91mBlock diffusion is disabled. Inference will be very slow. This is not recommended for MMLU-Pro.\033[0m")

        if self.early_stop:
            # print in green color
            self.logger.info(f"\033[92mEarly stopping enabled with consecutive={self.early_stop_consecutive}\033[0m")
        else:
            self.logger.warning(f"\033[91mEarly stopping is disabled. This is not recommended for MMLU-Pro.\033[0m")

        if self.confidence_based_adaptive_unmasking:
            self.logger.info(f"Confidence-based adaptive unmasking enabled with decay algorithm={self.decay_algorithm}")
            self.logger.info(f"Decay parameters: {self.decay_params}")



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

    def generate(self, input_ids, attention_mask, sample_dir):

        def generation_tokens_hook(step, x, logits):
            """Hook function called at each denoising step
            
            This function is called by the model's diffusion_generate method at each step
            of the denoising process. It allows us to track:
            1. The current state of the sequence
            2. The model's predictions (logits) for each position
            3. The noise schedule parameters
            4. The token transfer process
            
            Args:
                step (int): Current denoising step (None for initial step)
                x (torch.Tensor): Current token sequence
                logits (torch.Tensor): Model's logits for each position
            
            Returns:
                torch.Tensor: The current sequence (unchanged)
            """
            if step is not None and sample_dir is not None:  # Not the initial step and enable_hook is True
                # Calculate noise schedule parameters
                # t: current time step (starts at 1, decreases to eps)
                # s: next time step
                # p_transfer: probability of transferring tokens (1 - s/t)
                timesteps = torch.linspace(1, 1e-3, self.steps + 1, device=x.device)
                t = timesteps[step]
                s = timesteps[step + 1]
                p_transfer = 1 - s/t if step < self.steps - 1 else 1
                
                # Save full logits as a separate .pt file
                logits_path = os.path.join(sample_dir, f"step_{step:03d}_logits.pt")
                torch.save(logits.cpu(), logits_path)
                
                # Track information about this step and save immediately
                if self.enable_hook:
                    step_info = self._track_denoising_step(
                        step=step,
                        x=x,
                        logits=logits,
                        mask_index=(x == self.model.config.mask_token_id),
                        t=t,
                        s=s,
                        p_transfer=p_transfer,
                        num_transfer=(x == self.model.config.mask_token_id).sum().item()
                    )
                
                # Save step info immediately to disk
                step_file = os.path.join(sample_dir, f"step_{step:03d}.json")
                with open(step_file, "w") as f:
                    json.dump(step_info, f)
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
        
        start_time = time.time()

        # Generate response
        response = self.model.diffusion_generate(**generation_params)
        
        end_time = time.time()
        lat = end_time - start_time
        
        return response, lat

    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        latency = []
        self.model.eval()

        for subject, data in self.testset.items():
            subject_output = []
            output_file = os.path.join(self.run_dir, f"output_{subject}.txt")
            # eval_logger = EvaluationLogger(os.path.join(self.run_dir, f"gsm8k_eval_results_{subject}.json"))
            eval_logger = EvaluationLogger(os.path.join(self.run_dir, f"eval_results_{subject}.json"))

            print(f"Running {subject} with {len(data['sample'])} samples")
            # Print: evaluation results will be saved in {self.run_dir}/eval_results_{subject}.json
            # with highlight color and emojis   
            print(f"\033[92mEvaluation results will be saved in {self.run_dir}/eval_results_{subject}.json\033[0m")
            
            # Only create denoising directory if enable_hook is True
            denoising_dir = os.path.join(self.run_dir, f"denoising_steps_{subject}")
            if self.enable_hook:
                os.makedirs(denoising_dir, exist_ok=True)
                
            # Create directory for attention weights if enabled
            if self.save_attention_weights and self.attention_weights_path:
                subject_attn_dir = os.path.join(self.attention_weights_path, subject)
                os.makedirs(subject_attn_dir, exist_ok=True)
                self.logger.info(f"Attention weights for {subject} will be saved to {subject_attn_dir}")

            f = open(output_file, "w")
            
            # Subsample the dataset if enabled
            if self.subsample:
                import random
                random.seed(self.subsample_seed)
                dataset_size = len(data["sample"])
                
                # Check if subsample_size is a list of indices
                if isinstance(self.subsample_size, list):
                    # Use the provided indices directly
                    subsample_indices = self.subsample_size
                    subsample_size = len(subsample_indices)
                    self.logger.info(f"Using specific indices for {subject} dataset: {subsample_indices}")
                else:
                    # Random subsampling as before
                    subsample_size = min(self.subsample_size, dataset_size)
                    subsample_indices = random.sample(range(dataset_size), subsample_size)
                    subsample_indices.sort()  # Sort for reproducibility
                    self.logger.info(f"Subsampling {subsample_size} samples from {subject} dataset of size {dataset_size} with seed {self.subsample_seed}")
                    self.logger.info(f"Subsample indices: {subsample_indices}")
                
                # Create subsampled dataset
                subsampled_samples = [data["sample"][i] for i in subsample_indices]
                subsampled_labels = [data["label"][i] for i in subsample_indices]
                
                # Create a dictionary with the subsampled data
                subsampled_data = {
                    "sample": subsampled_samples,
                    "label": subsampled_labels,
                    "original_indices": subsample_indices  # Store the original indices
                }
                
                # Use the subsampled dataset
                data_to_use = subsampled_data
            else:
                data_to_use = data
                
            pbar = tqdm(data_to_use["sample"])
            for idx, sample in enumerate(pbar):
                input_ids, attn_mask = self.tokenize(sample)
                gt = data_to_use["label"][idx]

                # Only create sample directory if enable_hook is True
                sample_dir = None
                if self.enable_hook:
                    sample_dir = os.path.join(denoising_dir, f"sample_{idx:05d}")
                    os.makedirs(sample_dir, exist_ok=True)

                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)

                tok, lat = self.generate(input_ids, attn_mask, sample_dir)
                tok = tok.tolist()
                tok = tok[0][input_ids.shape[1] :]
                
                generated_text = self.tokenizer.decode(tok, skip_special_tokens=True)
                correctness = self.metric(generated_text, gt)

                subject_output.append(correctness)
                latency.append(lat)

                acc = sum(subject_output) / len(subject_output)
                running_lat = sum(latency) / len(latency)

                # Calculate actual generated token length before the first eos token
                try:
                    eos_position = tok.index(self.tokenizer.eos_token_id)
                    actual_generated_token_length = eos_position
                except (ValueError, TypeError):
                    # If eos_token_id not found or tok is not a list, use full generated length
                    actual_generated_token_length = len(tok)

                # Log actual number of tokens generated and running average
                if not hasattr(self, 'total_tokens_generated'):
                    self.total_tokens_generated = 0
                    self.num_samples = 0
                
                self.total_tokens_generated += actual_generated_token_length
                self.num_samples += 1
                avg_tokens_generated = self.total_tokens_generated / self.num_samples
                
                self.logger.info(f"Sample {idx}: Generated {actual_generated_token_length} tokens (Running avg: {avg_tokens_generated:.2f})")

                pbar.set_description(f"Acc: {acc:.4f} | Gen Len: {actual_generated_token_length}/{self.max_gen_toks} | Lat: {lat:.4f}s (avg: {running_lat:.4f}s) | Correct: {correctness}")

                # Log results
                log_item = {
                    "Prompt Index": idx,
                    "Batch Size": 1,
                    "Steps": self.steps,
                    "Max Gen Length": self.max_gen_toks,
                    "Block Length": self.block_size if self.use_block_diffusion else None,
                    "Avg Latency (s)": lat,
                    "Throughput (req/s)": 1/lat if lat > 0 else 0,
                    "Generated Token Length": len(tok),
                    "Actual Generated Token Length": actual_generated_token_length,
                    "Input Token Length": len(self.tokenizer.encode(sample)),
                    "Prompt": sample,
                    "Response": generated_text,
                    "Answer": gt,
                    "Correct": correctness,
                    "Avg Tokens per Step": len(tok)/self.steps if self.steps > 0 else 1,
                    "Early Stopping": self.early_stop,
                    "Early Stop Consecutive": self.early_stop_consecutive if self.early_stop else None,
                    "Denoising Steps Directory": sample_dir,  # Reference to the directory containing step files
                    "Block Diffusion": self.use_block_diffusion,
                    "Block Size": self.block_size if self.use_block_diffusion else None,
                    "Full Query Attention": self.use_full_query_attn if self.use_block_diffusion else None
                }
                eval_logger.log(log_item)

                pbar.set_description(f"Acc: {acc:.4f} | Gen Len: {actual_generated_token_length}/{self.max_gen_toks} | Lat: {lat:.4f}s (avg: {running_lat:.4f}s) | Correct: {correctness}")

                # save the generated text
                out_str = f"Test ID = [{idx}] | [{correctness}] \n{generated_text}"
                print(out_str, file=f)

                log_info = {
                    "Accuracy": acc, 
                    "latency": lat, 
                    "Correct": float(correctness), 
                    "Token Length": len(tok),
                    "Actual Generated Token Length": actual_generated_token_length,
                    "Input Token Length": len(self.tokenizer.encode(sample)),
                    "Generated Token Length": len(tok),
                    "Avg Tokens Per Step": len(tok)/self.steps if self.steps > 0 else 1,
                    "Avg Latency": running_lat,
                    "Top K Size": 20,  # Added to track top-k size
                    "Block Diffusion": self.use_block_diffusion,
                    "Block Size": self.block_size if self.use_block_diffusion else None,
                    "Early Stopping": self.early_stop
                }

                # Save evaluation info in sample_dir if hook is enabled
                if self.enable_hook and sample_dir is not None:
                    with open(os.path.join(sample_dir, "log_info.json"), "w") as f_log:
                        json.dump(log_info, f_log)

                if self.wandb_flag:
                    wandb.log(log_info)


                # COLOR
                RED = "\033[91m"
                GREEN = "\033[92m"
                RESET = "\033[0m"
                if correctness == 1:
                    print(f"{GREEN}Correct{RESET}")
                else:
                    print(f"{RED}Incorrect: {generated_text} | {gt}{RESET}")

            f.close()

class PiQADream(PiQAEval):
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
        
        # Log configuration information
        if not self.enable_hook:
            self.logger.info(f"Generation tokens hook disabled")
            
        if self.use_block_diffusion:
            # print in green color
            self.logger.info(f"\033[92mBlock diffusion enabled with block size={self.block_size}\033[0m")
            if self.use_full_query_attn:
                self.logger.info(f"Using full query attention mode")
        else:
            # Warn using in capital red color
            self.logger.warning(f"\033[91mBlock diffusion is disabled. Inference will be very slow. This is not recommended for PiQA.\033[0m")

        if self.early_stop:
            # print in green color
            self.logger.info(f"\033[92mEarly stopping enabled with consecutive={self.early_stop_consecutive}\033[0m")
        else:
            self.logger.warning(f"\033[91mEarly stopping is disabled. This is not recommended for PiQA.\033[0m")

        if self.confidence_based_adaptive_unmasking:
            self.logger.info(f"Confidence-based adaptive unmasking enabled with decay algorithm={self.decay_algorithm}")
            self.logger.info(f"Decay parameters: {self.decay_params}")

    def __name__(self):
        return "PiQADream"
    
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

    def generate(self, input_ids, attention_mask, sample_dir):
        def generation_tokens_hook(step, x, logits):
            """Hook function called at each denoising step
            
            This function is called by the model's diffusion_generate method at each step
            of the denoising process. It allows us to track:
            1. The current state of the sequence
            2. The model's predictions (logits) for each position
            3. The noise schedule parameters
            4. The token transfer process
            
            Args:
                step (int): Current denoising step (None for initial step)
                x (torch.Tensor): Current token sequence
                logits (torch.Tensor): Model's logits for each position
            
            Returns:
                torch.Tensor: The current sequence (unchanged)
            """
            if step is not None and sample_dir is not None:  # Not the initial step and enable_hook is True
                # Calculate noise schedule parameters
                # t: current time step (starts at 1, decreases to eps)
                # s: next time step
                # p_transfer: probability of transferring tokens (1 - s/t)
                timesteps = torch.linspace(1, 1e-3, self.steps + 1, device=x.device)
                t = timesteps[step]
                s = timesteps[step + 1]
                p_transfer = 1 - s/t if step < self.steps - 1 else 1
                
                # Save full logits as a separate .pt file
                logits_path = os.path.join(sample_dir, f"step_{step:03d}_logits.pt")
                torch.save(logits.cpu(), logits_path)
                
                # Track information about this step and save immediately
                if self.enable_hook:
                    step_info = self._track_denoising_step(
                        step=step,
                        x=x,
                        logits=logits,
                        mask_index=(x == self.model.config.mask_token_id),
                        t=t,
                        s=s,
                        p_transfer=p_transfer,
                        num_transfer=(x == self.model.config.mask_token_id).sum().item()
                    )
                
                # Save step info immediately to disk
                step_file = os.path.join(sample_dir, f"step_{step:03d}.json")
                with open(step_file, "w") as f:
                    json.dump(step_info, f)
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
        self.model.eval()

        eval_logger = EvaluationLogger(os.path.join(self.run_dir, "eval_results.json"))
        self.model.eval()

        # Create directory for denoising steps
        denoising_dir = os.path.join(self.run_dir, "denoising_steps")
        if self.enable_hook:
            os.makedirs(denoising_dir, exist_ok=True)

        pbar = tqdm(self.testset["dataset"])
        for idx, sample in enumerate(pbar):
            sample_dir = None
            if self.enable_hook:
                sample_dir = os.path.join(denoising_dir, f"sample_{idx:05d}")
                os.makedirs(sample_dir, exist_ok=True)

            
            def generation_tokens_hook(step, x, logits):
                """Hook function called at each denoising step
                
                This function is called by the model's diffusion_generate method at each step
                of the denoising process. It allows us to track:
                1. The current state of the sequence
                2. The model's predictions (logits) for each position
                3. The noise schedule parameters
                4. The token transfer process
                
                Args:
                    step (int): Current denoising step (None for initial step)
                    x (torch.Tensor): Current token sequence
                    logits (torch.Tensor): Model's logits for each position
                
                Returns:
                    torch.Tensor: The current sequence (unchanged)
                """
                if step is not None:  # Not the initial step
                    # Calculate noise schedule parameters
                    # t: current time step (starts at 1, decreases to eps)
                    # s: next time step
                    # p_transfer: probability of transferring tokens (1 - s/t)
                    timesteps = torch.linspace(1, 1e-3, self.steps + 1, device=x.device)
                    t = timesteps[step]
                    s = timesteps[step + 1]
                    p_transfer = 1 - s/t if step < self.steps - 1 else 1
                    
                    # Track information about this step and save immediately
                    step_info = self._track_denoising_step(
                        step=step,
                        x=x,
                        logits=logits,
                        mask_index=(x == self.model.config.mask_token_id),
                        t=t,
                        s=s,
                        p_transfer=p_transfer,
                        num_transfer=(x == self.model.config.mask_token_id).sum().item()
                    )
                    
                    # Save step info immediately to disk
                    step_file = os.path.join(sample_dir, f"step_{step:03d}.json")
                    with open(step_file, "w") as f:
                        json.dump(step_info, f)
                return x

            
            input_ids, attention_mask = self.tokenize(sample)
            
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            # label context
            gt = self.testset["label"][idx]

            tok, lat = self.generate(input_ids, attention_mask, sample_dir)
            tok = tok.tolist()
            tok = tok[0][input_ids.shape[1] :]

            # Decode response
            dec_tok = self.tokenizer.decode(tok, skip_special_tokens=True)
            
            # Get ground truth
            gt = self.testset["label"][idx]
            
            # Evaluate correctness
            correctness, pred, gt = self.metric(dec_tok, gt)
            output.append(correctness)
            latency.append(lat)

            # Calculate running averages
            running_acc = sum(output) / len(output)
            running_lat = sum(latency) / len(latency)

            # Calculate actual generated token length before the first eos token
            try:
                eos_position = tok.index(self.tokenizer.eos_token_id)
                actual_generated_token_length = eos_position
            except (ValueError, TypeError):
                # If eos_token_id not found or tok is not a list, use full generated length
                actual_generated_token_length = len(tok)

            # Log actual number of tokens generated and running average
            if not hasattr(self, 'total_tokens_generated'):
                self.total_tokens_generated = 0
                self.num_samples = 0
            
            self.total_tokens_generated += actual_generated_token_length
            self.num_samples += 1
            avg_tokens_generated = self.total_tokens_generated / self.num_samples
            
            self.logger.info(f"Sample {idx}: Generated {actual_generated_token_length} tokens (Running avg: {avg_tokens_generated:.2f})")

            pbar.set_description(f"Acc: {running_acc:.2f} | Avg Lat: {running_lat:.2f}s | Curr Lat: {lat:.2f}s | Tokens: {actual_generated_token_length} (avg: {avg_tokens_generated:.2f})")

            # Log results
            log_item = {
                "Prompt Index": idx,
                "Batch Size": 1,
                "Steps": self.steps,
                "Max Gen Length": self.max_gen_toks,
                "Block Length": self.block_size if self.use_block_diffusion else None,
                "Avg Latency (s)": lat,
                "Throughput (req/s)": 1/lat if lat > 0 else 0,
                "Generated Token Length": len(tok),
                "Actual Generated Token Length": actual_generated_token_length,
                "Input Token Length": len(self.tokenizer.encode(sample)),
                "Prompt": sample,
                "Response": dec_tok,
                "Answer": gt,
                "Correct": correctness,
                "Avg Tokens per Step": len(tok)/self.steps if self.steps > 0 else 1,
                "Early Stopping": self.early_stop,
                "Early Stop Consecutive": self.early_stop_consecutive if self.early_stop else None,
                "Denoising Steps Directory": sample_dir,  # Reference to the directory containing step files
                "Block Diffusion": self.use_block_diffusion,
                "Block Size": self.block_size if self.use_block_diffusion else None,
                "Full Query Attention": self.use_full_query_attn if self.use_block_diffusion else None
            }
            eval_logger.log(log_item)

            pbar.set_description(f"Acc: {running_acc:.4f} | Gen Len: {actual_generated_token_length}/{self.max_gen_toks} | Lat: {lat:.4f}s (avg: {running_lat:.4f}s) | Correct: {correctness}")


            # save the generated text
            out_str = f"Test ID = [{idx}] | Correctness = {correctness} |  GT=[{gt}] | Extracted ANS = {pred} \n\n Predicted = {dec_tok}"
            self.logger.info(out_str)

            log_info = {
                "Accuracy": running_acc, 
                "latency": lat, 
                "Correct": float(correctness), 
                "Token Length": len(tok),
                "Actual Generated Token Length": actual_generated_token_length,
                "Input Token Length": len(self.tokenizer.encode(sample)),
                "Generated Token Length": len(tok),
                "Avg Tokens Per Step": len(tok)/self.steps if self.steps > 0 else 1,
                "Avg Latency": running_lat,
                "Top K Size": 20,  # Added to track top-k size
                "Block Diffusion": self.use_block_diffusion,
                "Block Size": self.block_size if self.use_block_diffusion else None,
                "Early Stopping": self.early_stop
            }

            with open(os.path.join(sample_dir, "log_info.json"), "w") as f:
                json.dump(log_info, f)

            if self.wandb_flag:
                wandb.log(log_info)
            
            # save the evaluation information also in sample_dir
            with open(os.path.join(sample_dir, "log_info.json"), "w") as f:
                json.dump(log_info, f)

            # save every 100 prompts
            if idx % 100 == 0:
                eval_logger.save()
        
        eval_logger.save()

        avg = sum(output) / len(self.testset["dataset"])
        avg_lat = sum(latency) / len(latency)

        self.logger.info(f"Average Score (exact match) = {avg:.4f} | average latency = {avg_lat:.2f}")
        return output, latency

class OBQADream(OpenbookQAEval):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir, model, tokenizer, draft_model, draft_tokenizer)

        dream_config = self.config["dream"]
        self.temperature = dream_config["temperature"]
        self.top_p = dream_config["top_p"]
        self.steps = dream_config["steps"]
        # self.alg = dream_config["alg"]
        self.alg = dream_config.get("alg", None)
        # self.alg_temp = dream_config["alg_temp"]
        self.alg_temp = dream_config.get("alg_temp", None)

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

        # Log configuration information
        if not self.enable_hook:
            self.logger.info(f"Generation tokens hook disabled")
            
        if self.use_block_diffusion:
            # print in green color
            self.logger.info(f"\033[92mBlock diffusion enabled with block size={self.block_size}\033[0m")
            if self.use_full_query_attn:
                self.logger.info(f"Using full query attention mode")
        else:
            # Warn using in capital red color
            self.logger.warning(f"\033[91mBlock diffusion is disabled. Inference will be very slow. This is not recommended for OBQA.\033[0m")

        if self.early_stop:
            # print in green color
            self.logger.info(f"\033[92mEarly stopping enabled with consecutive={self.early_stop_consecutive}\033[0m")
        else:
            self.logger.warning(f"\033[91mEarly stopping is disabled. This is not recommended for OBQA.\033[0m")

        if self.confidence_based_adaptive_unmasking:
            self.logger.info(f"Confidence-based adaptive unmasking enabled with decay algorithm={self.decay_algorithm}")
            self.logger.info(f"Decay parameters: {self.decay_params}")

    def __name__(self):
        return "OBQADream"

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
    
    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        output = []
        latency = []
        self.model.eval()

        eval_logger = EvaluationLogger(os.path.join(self.run_dir, "eval_results.json"))
        self.model.eval()

        # Create directory for denoising steps
        denoising_dir = os.path.join(self.run_dir, "denoising_steps")
        if self.enable_hook:
            os.makedirs(denoising_dir, exist_ok=True)

        pbar = tqdm(self.testset["dataset"])
        for idx, sample in enumerate(pbar):
            sample_dir = None
            if self.enable_hook:
                sample_dir = os.path.join(denoising_dir, f"sample_{idx:05d}")
                os.makedirs(sample_dir, exist_ok=True)

            
            def generation_tokens_hook(step, x, logits):
                """Hook function called at each denoising step
                
                This function is called by the model's diffusion_generate method at each step
                of the denoising process. It allows us to track:
                1. The current state of the sequence
                2. The model's predictions (logits) for each position
                3. The noise schedule parameters
                4. The token transfer process
                
                Args:
                    step (int): Current denoising step (None for initial step)
                    x (torch.Tensor): Current token sequence
                    logits (torch.Tensor): Model's logits for each position
                
                Returns:
                    torch.Tensor: The current sequence (unchanged)
                """
                if step is not None:  # Not the initial step
                    # Calculate noise schedule parameters
                    # t: current time step (starts at 1, decreases to eps)
                    # s: next time step
                    # p_transfer: probability of transferring tokens (1 - s/t)
                    timesteps = torch.linspace(1, 1e-3, self.steps + 1, device=x.device)
                    t = timesteps[step]
                    s = timesteps[step + 1]
                    p_transfer = 1 - s/t if step < self.steps - 1 else 1
                    
                    # Track information about this step and save immediately
                    step_info = self._track_denoising_step(
                        step=step,
                        x=x,
                        logits=logits,
                        mask_index=(x == self.model.config.mask_token_id),
                        t=t,
                        s=s,
                        p_transfer=p_transfer,
                        num_transfer=(x == self.model.config.mask_token_id).sum().item()
                    )
                    
                    # Save step info immediately to disk
                    step_file = os.path.join(sample_dir, f"step_{step:03d}.json")
                    with open(step_file, "w") as f:
                        json.dump(step_info, f)
                return x

            
            input_ids, attention_mask = self.tokenize(sample)
            
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            # label context
            gt = self.testset["label"][idx]

            # Generate response using diffusion
            start_time = time.time()
            response = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_gen_toks,
                temperature=self.temperature,
                top_p=self.top_p,
                steps=self.steps,
                alg=self.alg,
                alg_temp=self.alg_temp,
                generation_tokens_hook_func=generation_tokens_hook  # Pass our hook function
            )
            end_time = time.time()
            lat = end_time - start_time

            # Decode response
            dec_tok = self.tokenizer.decode(response[0], skip_special_tokens=True)
            
            # Get ground truth
            gt = self.testset["label"][idx]
            
            # Evaluate correctness
            correctness, pred, gt = self.metric(dec_tok, gt)
            output.append(correctness)
            latency.append(lat)

            # Calculate running averages
            running_acc = sum(output) / len(output)
            running_lat = sum(latency) / len(latency)

            # Log results
            log_item = {
                "Prompt Index": idx,
                "Batch Size": 1,
                "Steps": self.steps,
                "Max Gen Length": self.max_gen_toks,
                "Block Length": None,
                "Avg Latency (s)": lat,
                "Throughput (req/s)": 1/lat if lat > 0 else 0,
                "Generated Token Length": len(response[0]),
                "Input Token Length": len(self.tokenizer.encode(sample)),
                "Prompt": sample,
                "Response": dec_tok,
                "Answer": gt,
                "Correct": correctness,
                "Avg Tokens per Step": len(response[0])/self.steps if self.steps > 0 else 1,
                "Denoising Steps Directory": sample_dir  # Reference to the directory containing step files
            }
            eval_logger.log(log_item)
            
            # Update progress bar with running averages
            pbar.set_description(f"Acc: {running_acc:.2f} | Avg Lat: {running_lat:.2f}s | Curr Lat: {lat:.2f}s")

            # save the generated text
            out_str = f"Test ID = [{idx}] | Correctness = {correctness} |  GT=[{gt}] | Extracted ANS = {pred} \n\n Predicted = {dec_tok}"
            self.logger.info(out_str)

            log_info = {
                "Accuracy": running_acc,
                "Latency": lat,
                "Correct": correctness,
                "Token Length": len(response[0]),
                "Input Token Length": len(self.tokenizer.encode(sample)),
                "Generated Token Length": len(response[0]),
                "Avg Tokens Per Step": len(response[0])/self.steps if self.steps > 0 else 1,
                "Avg Latency": running_lat,
                # "Top K Size": 20,  # Added to track top-k size
                "Block Diffusion": self.use_block_diffusion,
                "Block Size": self.block_size if self.use_block_diffusion else None,
                "Early Stopping": self.early_stop
            }
            with open(os.path.join(sample_dir, "log_info.json"), "w") as f:
                json.dump(log_info, f)

            if self.wandb_flag:
                wandb.log(log_info)
            
            # save the evaluation information also in sample_dir
            with open(os.path.join(sample_dir, "log_info.json"), "w") as f:
                json.dump(log_info, f)

            # save every 100 prompts
            if idx % 100 == 0:
                eval_logger.save()
        
        eval_logger.save()

        avg = sum(output) / len(self.testset["dataset"])
        avg_lat = sum(latency) / len(latency)

        self.logger.info(f"Average Score (exact match) = {avg:.4f} | average latency = {avg_lat:.2f}")
        return output, latency

class HellaswagDream(HellaswagEval):
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
            # print in green color
            self.logger.info(f"\033[92mBlock diffusion enabled with block size={self.block_size}\033[0m")
            if self.use_full_query_attn:
                self.logger.info(f"Using full query attention mode")
        else:
            # Warn using in capital red color
            self.logger.warning(f"\033[91mBlock diffusion is disabled. Inference will be very slow. This is not recommended for Hellaswag.\033[0m")
        
        if self.early_stop:
            # print in green color
            self.logger.info(f"\033[92mEarly stopping enabled with consecutive={self.early_stop_consecutive}\033[0m")
        else:
            self.logger.warning(f"\033[91mEarly stopping is disabled. This is not recommended for Hellaswag.\033[0m")

        if self.confidence_based_adaptive_unmasking:
            self.logger.info(f"Confidence-based adaptive unmasking enabled with decay algorithm={self.decay_algorithm}")
            self.logger.info(f"Decay parameters: {self.decay_params}")

        # For testing/debugging, ensure steps is at least 1
        if self.steps < 1:
            self.logger.warning(f"Steps value {self.steps} is too low, setting to 1")
            self.steps = 1

    def __name__(self):
        return "HellaswagDream"

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
    
    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        output = []
        latency = []
        self.model.eval()

        eval_logger = EvaluationLogger(os.path.join(self.run_dir, "eval_results.json"))
        self.model.eval()

        # Create directory for denoising steps
        denoising_dir = os.path.join(self.run_dir, "denoising_steps")
        if self.enable_hook:
            os.makedirs(denoising_dir, exist_ok=True)

        pbar = tqdm(self.testset["dataset"])
        for idx, sample in enumerate(pbar):
            sample_dir = None
            if self.enable_hook:
                sample_dir = os.path.join(denoising_dir, f"sample_{idx:05d}")
                os.makedirs(sample_dir, exist_ok=True)

            
            def generation_tokens_hook(step, x, logits):
                """Hook function called at each denoising step
                
                This function is called by the model's diffusion_generate method at each step
                of the denoising process. It allows us to track:
                1. The current state of the sequence
                2. The model's predictions (logits) for each position
                3. The noise schedule parameters
                4. The token transfer process
                
                Args:
                    step (int): Current denoising step (None for initial step)
                    x (torch.Tensor): Current token sequence
                    logits (torch.Tensor): Model's logits for each position
                
                Returns:
                    torch.Tensor: The current sequence (unchanged)
                """
                if step is not None and self.enable_hook:  # Not the initial step and enable_hook is True
                    # Calculate noise schedule parameters
                    # t: current time step (starts at 1, decreases to eps)
                    # s: next time step
                    # p_transfer: probability of transferring tokens (1 - s/t)
                    timesteps = torch.linspace(1, 1e-3, self.steps + 1, device=x.device)
                    t = timesteps[step]
                    s = timesteps[step + 1]
                    p_transfer = 1 - s/t if step < self.steps - 1 else 1
                    
                    # Track information about this step and save immediately
                    step_info = self._track_denoising_step(
                        step=step,
                        x=x,
                        logits=logits,
                        mask_index=(x == self.model.config.mask_token_id),
                        t=t,
                        s=s,
                        p_transfer=p_transfer,
                        num_transfer=(x == self.model.config.mask_token_id).sum().item()
                    )
                    
                    # Save step info immediately to disk
                    step_file = os.path.join(sample_dir, f"step_{step:03d}.json")
                    with open(step_file, "w") as f:
                        json.dump(step_info, f)
                return x

            
            input_ids, attention_mask = self.tokenize(sample)
            
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            # label context
            gt = self.testset["label"][idx]

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

            # Generate response using diffusion
            start_time = time.time()
            # Note: fixed the generation_params to include block diffusion and early stopping
            response = self.model.diffusion_generate(**generation_params)

            
            end_time = time.time()
            lat = end_time - start_time

            # Decode response
            dec_tok = self.tokenizer.decode(response[0], skip_special_tokens=True)

            # remove the prompt from the response
            gen_tok = self.tokenizer.decode(
                response[0][input_ids.shape[1]:], 
                skip_special_tokens=False)

            # remove the eos token from the response
            gen_tok = gen_tok.replace(self.tokenizer.eos_token, "")
            
            YELLOW = "\033[93m"
            PINK = "\033[95m"
            RESET = "\033[0m"
            # print the gen_tok in yellow
            print(f"{PINK}Generated Token: {gen_tok}{RESET}", flush=True)
            
            # Get ground truth
            gt = self.testset["label"][idx]

            # Evaluate correctness
            correctness, pred, gt = self.metric(dec_tok, gt)

            # add at top
            BOLD  = "\033[1m"
            GREEN = "\033[92m"
            RED   = "\033[91m"
            RESET = "\033[0m"
            CHECK_EMOJI = "✅"
            CROSS_EMOJI = "❌"

            # later in your loop
            if correctness:
                # green, bold, with emoji
                print(f"{GREEN}{BOLD}{CHECK_EMOJI} Pred: {pred} | GT: {gt}{RESET}", flush=True)
            else:
                # red, bold, with emoji
                print(f"{RED}{BOLD}{CROSS_EMOJI} Pred: {pred} | GT: {gt}{RESET}", flush=True)



            output.append(correctness)
            latency.append(lat)

            # Calculate running averages
            running_acc = sum(output) / len(output)
            running_lat = sum(latency) / len(latency)

    
            GREEN = "\033[92m"
            BLUE = "\033[94m"
            YELLOW = "\033[93m"
            RESET = "\033[0m"
            # # Update progress bar with running averages
            pbar.set_description(f"{YELLOW}Acc: {running_acc:.2f} | Avg Lat: {running_lat:.2f}s | Curr Lat: {lat:.2f}s{RESET}")

            ############################################################
            # TODO:
            # extract generated tokens (no prompt) before eos token, include special tokens
            gen_ids = response[0].tolist()[input_ids.shape[1]:]
            eos_id = self.tokenizer.eos_token_id
            if eos_id in gen_ids:
                eos_idx = gen_ids.index(eos_id)
                gen_ids = gen_ids[:eos_idx]

            # decode including special tokens
            generated_answer = self.tokenizer.decode(gen_ids, skip_special_tokens=False)

            # count tokens in generated answer
            actual_generated_tok_len = len(gen_ids)

            # maintain running totals for average
            if not hasattr(self, 'total_tokens_generated'):
                self.total_tokens_generated = 0
                self.num_samples = 0
            self.total_tokens_generated += actual_generated_tok_len
            self.num_samples += 1
            running_avg_tokens = self.total_tokens_generated / self.num_samples

            # log generated answer and token counts
            # self.logger.info(f"{BLUE}Generated Answer (before eos): {generated_answer} vs. GT: {gt}{RESET}")
            self.logger.info(
                f"{BLUE}Actual Generated Token Length: "
                f"{actual_generated_tok_len} "
                f"(running avg: {running_avg_tokens:.2f}){RESET}"
            )
            ############################################################

            # Log results
            log_item = {
                "Prompt Index": idx,
                "Batch Size": 1,
                "Steps": self.steps,
                "Max Gen Length": self.max_gen_toks,
                "Block Length": self.block_size if self.use_block_diffusion else None,
                "Avg Latency (s)": lat,
                "Throughput (req/s)": 1/lat if lat > 0 else 0,
                "Generated Token Length": actual_generated_tok_len,
                "Input Token Length": len(self.tokenizer.encode(sample)),
                "Prompt": sample,
                "Response": dec_tok,
                "Answer": gt,
                "Correct": correctness,
                "Avg Tokens per Step": actual_generated_tok_len/self.steps if self.steps > 0 else 1,
                "Early Stopping": self.early_stop,
                "Early Stop Consecutive": self.early_stop_consecutive if self.early_stop else None,
                "Denoising Steps Directory": sample_dir,  # Reference to the directory containing step files
                "Block Diffusion": self.use_block_diffusion,
                "Block Size": self.block_size if self.use_block_diffusion else None,
                "Full Query Attention": self.use_full_query_attn if self.use_block_diffusion else None
            }
            eval_logger.log(log_item)



            # save the generated text
            # out_str = f"Test ID = [{idx}] | Correctness = {correctness} |  GT=[{gt}] | Extracted ANS = {pred} \n\n Predicted = {dec_tok}"
            out_str = f"Test ID = [{idx}] | Correctness = {correctness} |  GT=[{gt}] | Extracted ANS = {pred}"
            self.logger.info(out_str)

            log_info = {
                "Accuracy": running_acc,
                "Latency": lat,
                "Correct": correctness,
                "Token Length": len(response[0]),
                "Input Token Length": len(self.tokenizer.encode(sample)),
                "Generated Token Length": len(response[0]),
                "Avg Tokens Per Step": len(response[0])/self.steps if self.steps > 0 else 1,
                "Avg Latency": running_lat,
                # "Top K Size": 20,  # Added to track top-k size
                "Block Diffusion": self.use_block_diffusion,
                "Block Size": self.block_size if self.use_block_diffusion else None,
                "Early Stopping": self.early_stop
            }
            if self.enable_hook:
                with open(os.path.join(sample_dir, "log_info.json"), "w") as f:
                    json.dump(log_info, f)

            if self.wandb_flag:
                wandb.log(log_info)

            # save every 100 prompts
            if idx % 100 == 0:
                eval_logger.save()
        
        eval_logger.save()

        avg = sum(output) / len(self.testset["dataset"])
        avg_lat = sum(latency) / len(latency)

        self.logger.info(f"Average Score (exact match) = {avg:.4f} | average latency = {avg_lat:.2f}")
        return output, latency

class WinoGrandeDream(WinoGrandeEval):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir, model, tokenizer, draft_model, draft_tokenizer)

        dream_config = self.config["dream"]
        self.temperature = dream_config["temperature"]
        self.top_p = dream_config["top_p"]
        self.steps = dream_config["steps"]
        self.alg = dream_config["alg"]
        self.alg_temp = dream_config["alg_temp"]

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
    
    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        output = []
        latency = []
        self.model.eval()

        eval_logger = EvaluationLogger(os.path.join(self.run_dir, "eval_results.json"))
        self.model.eval()

        # Create directory for denoising steps
        denoising_dir = os.path.join(self.run_dir, "denoising_steps")
        os.makedirs(denoising_dir, exist_ok=True)

        pbar = tqdm(self.testset["dataset"])
        for idx, sample in enumerate(pbar):
            sample_dir = os.path.join(denoising_dir, f"sample_{idx:05d}")
            os.makedirs(sample_dir, exist_ok=True)

            
            def generation_tokens_hook(step, x, logits):
                """Hook function called at each denoising step
                
                This function is called by the model's diffusion_generate method at each step
                of the denoising process. It allows us to track:
                1. The current state of the sequence
                2. The model's predictions (logits) for each position
                3. The noise schedule parameters
                4. The token transfer process
                
                Args:
                    step (int): Current denoising step (None for initial step)
                    x (torch.Tensor): Current token sequence
                    logits (torch.Tensor): Model's logits for each position
                
                Returns:
                    torch.Tensor: The current sequence (unchanged)
                """
                if step is not None:  # Not the initial step
                    # Calculate noise schedule parameters
                    # t: current time step (starts at 1, decreases to eps)
                    # s: next time step
                    # p_transfer: probability of transferring tokens (1 - s/t)
                    timesteps = torch.linspace(1, 1e-3, self.steps + 1, device=x.device)
                    t = timesteps[step]
                    s = timesteps[step + 1]
                    p_transfer = 1 - s/t if step < self.steps - 1 else 1
                    
                    # Track information about this step and save immediately
                    step_info = self._track_denoising_step(
                        step=step,
                        x=x,
                        logits=logits,
                        mask_index=(x == self.model.config.mask_token_id),
                        t=t,
                        s=s,
                        p_transfer=p_transfer,
                        num_transfer=(x == self.model.config.mask_token_id).sum().item()
                    )
                    
                    # Save step info immediately to disk
                    step_file = os.path.join(sample_dir, f"step_{step:03d}.json")
                    with open(step_file, "w") as f:
                        json.dump(step_info, f)
                return x

            
            input_ids, attention_mask = self.tokenize(sample)
            
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            # label context
            gt = self.testset["label"][idx]

            # Generate response using diffusion
            start_time = time.time()
            response = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_gen_toks,
                temperature=self.temperature,
                top_p=self.top_p,
                steps=self.steps,
                alg=self.alg,
                alg_temp=self.alg_temp,
                generation_tokens_hook_func=generation_tokens_hook  # Pass our hook function
            )
            end_time = time.time()
            lat = end_time - start_time

            # Decode response
            dec_tok = self.tokenizer.decode(response[0], skip_special_tokens=True)
            
            # Get ground truth
            gt = self.testset["label"][idx]
            
            # Evaluate correctness
            correctness, pred, gt = self.metric(dec_tok, gt)
            output.append(correctness)
            latency.append(lat)

            # Calculate running averages
            running_acc = sum(output) / len(output)
            running_lat = sum(latency) / len(latency)

            # Log results
            log_item = {
                "Prompt Index": idx,
                "Batch Size": 1,
                "Steps": self.steps,
                "Max Gen Length": self.max_gen_toks,
                "Block Length": None,
                "Avg Latency (s)": lat,
                "Throughput (req/s)": 1/lat if lat > 0 else 0,
                "Generated Token Length": len(response[0]),
                "Input Token Length": len(self.tokenizer.encode(sample)),
                "Prompt": sample,
                "Response": dec_tok,
                "Answer": gt,
                "Correct": correctness,
                "Avg Tokens per Step": len(response[0])/self.steps if self.steps > 0 else 1,
                "Denoising Steps Directory": sample_dir  # Reference to the directory containing step files
            }
            eval_logger.log(log_item)
            
            # Update progress bar with running averages
            pbar.set_description(f"Acc: {running_acc:.2f} | Avg Lat: {running_lat:.2f}s | Curr Lat: {lat:.2f}s")

            # save the generated text
            out_str = f"Test ID = [{idx}] | Correctness = {correctness} |  GT=[{gt}] | Extracted ANS = {pred} \n\n Predicted = {dec_tok}"
            self.logger.info(out_str)

            log_info = {
                "Accuracy": running_acc,
                "Latency": lat,
                "Correct": correctness,
                "Token Length": len(response[0]),
                "Input Token Length": len(self.tokenizer.encode(sample)),
                "Generated Token Length": len(response[0]),
                "Avg Tokens Per Step": len(response[0])/self.steps if self.steps > 0 else 1,
                "Avg Latency": running_lat,
                # "Top K Size": 20,  # Added to track top-k size
                "Block Diffusion": self.use_block_diffusion,
                "Block Size": self.block_size if self.use_block_diffusion else None,
                "Early Stopping": self.early_stop
            }
            with open(os.path.join(sample_dir, "log_info.json"), "w") as f:
                json.dump(log_info, f)

            if self.wandb_flag:
                wandb.log(log_info)
            
            # save the evaluation information also in sample_dir
            with open(os.path.join(sample_dir, "log_info.json"), "w") as f:
                json.dump(log_info, f)

            # save every 100 prompts
            if idx % 100 == 0:
                eval_logger.save()
        
        eval_logger.save()

        avg = sum(output) / len(self.testset["dataset"])
        avg_lat = sum(latency) / len(latency)

        self.logger.info(f"Average Score (exact match) = {avg:.4f} | average latency = {avg_lat:.2f}")
        return output, latency

class ARCCDream(OBQADream):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir, model, tokenizer, draft_model, draft_tokenizer)

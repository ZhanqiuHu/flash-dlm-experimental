"""
Common utilities for Dream evaluators
"""

import os
import json
import time
import torch
import wandb
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

# COLOR CONSTANTS
YELLOW = "\033[93m"
PURPLE = "\033[95m"
RED = "\033[91m"
BLUE = "\033[94m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

class DreamEvaluationLogger:
    """Enhanced logger for Dream evaluations with detailed tracking"""
    def __init__(self, run_dir: str, experiment_name: str):
        self.run_dir = run_dir
        self.experiment_name = experiment_name
        
        # Main evaluation results
        self.eval_results_file = os.path.join(run_dir, "eval_results.json")
        self.output_file = os.path.join(run_dir, "output.txt")
        self.logs = []
        
        # Create necessary directories
        self.denoising_dir = os.path.join(run_dir, "denoising_steps")
        self.attention_weights_dir = os.path.join(run_dir, "attention_weights")
        
        # Initialize files
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize all necessary files and directories"""
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.denoising_dir, exist_ok=True)
        os.makedirs(self.attention_weights_dir, exist_ok=True)
        
        # Initialize output file with header
        with open(self.output_file, "w") as f:
            f.write(f"=== {self.experiment_name} Evaluation Results ===\n\n")
    
    def log_step(self, 
                 step_idx: int,
                 sample: Any,
                 input_ids: torch.Tensor,
                 generated_text: str,
                 ground_truth: str,
                 prediction: str,
                 correctness: bool,
                 latency: float,
                 accuracy: float,
                 num_generated_tokens: int = 0,
                 sample_dir: Optional[str] = None,
                 wandb_flag: bool = False):
        """Log detailed information for each evaluation step"""
        # Create step info
        step_info = {
            "step_idx": step_idx,
            "timestamp": datetime.now().isoformat(),
            "correctness": correctness,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "generated_text": generated_text,
            "latency": latency,
            "accuracy": accuracy,
            "num_generated_tokens": num_generated_tokens,
            "tokens_per_second": num_generated_tokens / latency if latency > 0 else 0
        }
        
        # Add to logs
        self.logs.append(step_info)
        
        # Write to output file
        with open(self.output_file, "a") as f:
            f.write(f"\nTest ID = [{step_idx}]\n")
            f.write(f"Correctness = {correctness}\n")
            f.write(f"GT = [{ground_truth}]\n")
            f.write(f"Extracted ANS = {prediction}\n")
            f.write(f"Predicted = {generated_text}\n")
            f.write(f"Latency = {latency:.4f}s\n")
            f.write(f"Current Accuracy = {accuracy:.4f}\n")
            f.write(f"Generated Tokens = {num_generated_tokens}\n")
            f.write(f"Tokens/Second = {num_generated_tokens / latency:.2f}\n" if latency > 0 else "Tokens/Second = 0.00\n")
            f.write("-" * 80 + "\n")
        
        # Log to wandb if enabled
        if wandb_flag:
            wandb.log({
                "accuracy": accuracy,
                "latency": latency,
                "correctness": correctness,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "generated_text": generated_text,
                "num_generated_tokens": num_generated_tokens,
                "tokens_per_second": num_generated_tokens / latency if latency > 0 else 0,
                "step": step_idx
            })
        
        # Save intermediate results every 100 steps
        if step_idx % 100 == 0:
            self.save()
    
    def save(self):
        """Save current evaluation results"""
        # Save logs to JSON
        with open(self.eval_results_file, "w") as f:
            json.dump(self.logs, f, indent=2)
    
    def log_final_results(self, 
                         total_samples: int,
                         correct_predictions: int,
                         total_latency: float,
                         wandb_flag: bool = False):
        """Log final evaluation results"""
        final_accuracy = correct_predictions / total_samples
        avg_latency = total_latency / total_samples
        
        # Write final results to output file
        with open(self.output_file, "a") as f:
            f.write("\n=== Final Results ===\n")
            f.write(f"Total Samples: {total_samples}\n")
            f.write(f"Correct Predictions: {correct_predictions}\n")
            f.write(f"Final Accuracy: {final_accuracy:.4f}\n")
            f.write(f"Average Latency: {avg_latency:.4f}s\n")
        
        # Log final results to wandb
        if wandb_flag:
            wandb.log({
                "final_accuracy": final_accuracy,
                "average_latency": avg_latency,
                "total_samples": total_samples,
                "correct_predictions": correct_predictions
            })
        
        # Save final results
        self.save()
        
        return final_accuracy, avg_latency

def setup_dream_evaluation(config: Dict,
                          run_dir: str,
                          experiment_name: str,
                          enable_hook: bool = False,
                          save_attention_weights: bool = False) -> Tuple[DreamEvaluationLogger, Dict]:
    """Setup common Dream evaluation components"""
    # Initialize logger
    logger = DreamEvaluationLogger(run_dir, experiment_name)
    
    # Extract common Dream parameters
    dream_config = config["dream"]
    eval_params = {
        "temperature": dream_config["temperature"],
        "top_p": dream_config["top_p"],
        "steps": dream_config["steps"],
        "alg": dream_config["alg"],
        "alg_temp": dream_config["alg_temp"],
        "enable_hook": enable_hook,
        "save_attention_weights": save_attention_weights,
        "use_block_diffusion": dream_config.get("use_block_diffusion", False),
        "block_size": dream_config.get("block_size", 256),
        "use_full_query_attn": dream_config.get("use_full_query_attn", False),
        "early_stop": dream_config.get("early_stop", False),
        "early_stop_consecutive": dream_config.get("early_stop_consecutive", 5),
        "confidence_based_adaptive_unmasking": dream_config.get("confidence_based_adaptive_unmasking", False),
        "decay_algorithm": dream_config.get("decay_algorithm", "exp_unmasked_ratio"),
        "decay_params": dream_config.get("decay_params", {
            "alpha": 1.0,
            "gamma": 1.0
        })
    }
    
    return logger, eval_params

def prepare_generation_params(input_ids: torch.Tensor,
                            attention_mask: torch.Tensor,
                            eval_params: Dict,
                            tokenizer: Any,
                            sample_dir: Optional[str] = None) -> Dict:
    """Prepare generation parameters for Dream model"""
    # Common parameters
    generation_params = {
        "inputs": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": eval_params.get("max_gen_toks", 512),
        "temperature": eval_params["temperature"],
        "top_p": eval_params["top_p"],
        "steps": eval_params["steps"],
        "alg": eval_params["alg"],
        "alg_temp": eval_params["alg_temp"],
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True
    }
    
    # Add block diffusion parameters if enabled
    if eval_params["use_block_diffusion"]:
        generation_params.update({
            "use_block_diffusion": True,
            "block_size": eval_params["block_size"],
            "use_full_query_attn": eval_params["use_full_query_attn"],
            "max_length": None
        })
    
    # Add early stopping parameters if enabled
    if eval_params["early_stop"]:
        generation_params.update({
            "early_stop": True,
            "early_stop_consecutive": eval_params["early_stop_consecutive"]
        })
    
    # Add confidence-based adaptive unmasking parameters if enabled
    if eval_params["confidence_based_adaptive_unmasking"]:
        generation_params.update({
            "confidence_based_adaptive_unmasking": True,
            "decay_algorithm": eval_params["decay_algorithm"],
            "decay_params": eval_params["decay_params"]
        })
    
    # Add attention weight saving parameters if enabled
    if eval_params["save_attention_weights"] and sample_dir:
        sample_attn_dir = os.path.join(eval_params["attention_weights_path"], os.path.basename(sample_dir))
        os.makedirs(sample_attn_dir, exist_ok=True)
        generation_params.update({
            "save_attention_weights": True,
            "attention_weights_path": sample_attn_dir
        })
    
    return generation_params

def track_denoising_step(step: int,
                        x: torch.Tensor,
                        logits: torch.Tensor,
                        mask_index: torch.Tensor,
                        t: torch.Tensor,
                        s: torch.Tensor,
                        p_transfer: float,
                        num_transfer: int,
                        sample_dir: Optional[str] = None) -> Dict:
    """Track information about a denoising step"""
    # Get top-k tokens and their logits
    k = 20
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    top_k_probs = torch.softmax(top_k_logits, dim=-1)
    
    # Create step info
    step_info = {
        "step": step,
        "current_tokens": x.clone().detach().cpu().numpy().tolist(),
        "masked_positions": mask_index.clone().detach().cpu().numpy().tolist(),
        "logits": None,  # Skip saving full logits in JSON
        "top_k_tokens": top_k_indices.clone().detach().cpu().numpy().tolist(),
        "top_k_logits": top_k_logits.clone().detach().cpu().float().numpy().tolist(),
        "top_k_probs": top_k_probs.clone().detach().cpu().float().numpy().tolist(),
        "time_step": float(t.item()),
        "next_time_step": float(s.item()),
        "transfer_probability": float(p_transfer),
        "num_tokens_transferred": int(num_transfer)
    }
    
    # Save step info if directory provided
    if sample_dir:
        step_file = os.path.join(sample_dir, f"step_{step:03d}.json")
        with open(step_file, "w") as f:
            json.dump(step_info, f, indent=2)
        
        # Save full logits separately
        logits_path = os.path.join(sample_dir, f"step_{step:03d}_logits.pt")
        torch.save(logits.cpu(), logits_path)
    
    return step_info 
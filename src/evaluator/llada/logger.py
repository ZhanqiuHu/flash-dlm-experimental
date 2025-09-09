"""
Evaluation logger for tracking and saving evaluation results
"""

import json
import os
import platform
import torch
from datetime import datetime
from pathlib import Path

def get_gpu_info():
    """Get GPU type and count."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        return f"{gpu_count}x{gpu_name.replace(' ', '_')}"
    return "cpu"

def get_node_name():
    """Get the node/host name."""
    return platform.node().replace('.', '_')

class EvaluationLogger:
    def __init__(self, output_path):
        # Get node and GPU info
        node_name = get_node_name()
        gpu_info = get_gpu_info()
        
        # Modify output path to include node and GPU info
        base_path = Path(output_path)
        self.output_path = str(base_path.parent / f"{node_name}_{gpu_info}" / base_path.name)
        
        # Create directory if it doesn't exist
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.entries = []
        # Initialize running averages
        self.total_actual_generated_tokens = 0
        self.total_input_tokens = 0
        self.total_examples = 0

    def log(
        self,
        prompt_index,
        batch_size,
        steps,
        max_gen_length,
        block_length,
        avg_latency_s,
        throughput_req_s,
        generated_token_len,
        actual_generated_token_len,
        input_token_len,
        prompt,
        response,
        answer,
        correct,
        avg_tokens_per_step,
        early_stopping=None,
        early_stop_consecutive=None,
        denoising_steps_dir=None,
        block_diffusion=None,
        block_size=None,
        full_query_attention=None
    ):
        """Log a single evaluation result"""
        # Update running averages
        self.total_actual_generated_tokens += actual_generated_token_len
        self.total_input_tokens += input_token_len
        self.total_examples += 1
        
        avg_actual_generated_tokens = self.total_actual_generated_tokens / self.total_examples
        avg_input_tokens = self.total_input_tokens / self.total_examples

        entry = {
            "Prompt Index": prompt_index,
            "Batch Size": batch_size,
            "Steps": steps,
            "Max Gen Length": max_gen_length,
            "Block Length": block_length,
            "Avg Latency (s)": avg_latency_s,
            "Throughput (req/s)": throughput_req_s,
            "Generated Token Length": generated_token_len,
            "Actual Generated Token Length": actual_generated_token_len,
            "Input Token Length": input_token_len,
            "Prompt": prompt,
            "Response": response,
            "Answer": answer,
            "Correct": correct,
            "Avg Tokens per Step": avg_tokens_per_step,
            "Early Stopping": early_stopping,
            "Early Stop Consecutive": early_stop_consecutive,
            "Denoising Steps Directory": denoising_steps_dir,
            "Block Diffusion": block_diffusion,
            "Block Size": block_size,
            "Full Query Attention": full_query_attention,
            "Timestamp": datetime.now().isoformat()
        }

        # Print running statistics
        print(f"\nToken Statistics:")
        print(f"Current Example - Actual Generated: {actual_generated_token_len}, Input: {input_token_len}")
        print(f"Running Average - Actual Generated: {avg_actual_generated_tokens:.2f}, Input: {avg_input_tokens:.2f}")
        print(f"Total Examples Processed: {self.total_examples}\n")

        self.entries.append(entry)
        # Save after each entry to ensure we don't lose data
        self.save()

    def save(self):
        """Save the current entries to file"""
        with open(self.output_path, "w") as f:
            json.dump(self.entries, f, indent=4) 
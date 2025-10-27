#!/usr/bin/env python3
"""
Flash-DLM Simple API Script
A streamlined version that takes a prompt and generates a response using Flash-DLM.
"""

import os
import sys
import argparse
import torch
import time
from pathlib import Path

# Add project paths
sys.path += [str(Path(__file__).parent), str(Path(__file__).parent / "src")]

from src.model.auto_map import ModelMap
from guided_diffusion.dream_eval.base_guided_evaluator import load_tokenizer
from guided_diffusion.guided_diff_utils import assisted_block_diffusion_generate, AssistedDiffusionConfig

class FlashDLMAPI:
    def __init__(self, dream_model_path="Dream-org/Dream-Flash-Instruct-7B", ar_model_path="Qwen/Qwen2.5-1.5B-Instruct"):
        """Initialize the Flash-DLM API with the specified models."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load models
        self._load_models(dream_model_path, ar_model_path)
        
        # Set up tokenizers
        self.tok_d = load_tokenizer(dream_model_path)
        self.tok_a = load_tokenizer(ar_model_path)
        
        # Set mask token
        self.mask_token_id = self.tok_d.mask_token_id if self.tok_d.mask_token_id is not None else -100
        self.mask_token_str = "[MASK]"
        
        print("Flash-DLM API initialized successfully!")
    
    def _load_models(self, dream_model_path, ar_model_path):
        """Load the Dream and AR models."""
        try:
            print("Loading Dream Flash model...")
            self.model_d = ModelMap(dream_model_path).fetch()
            print(f"Dream model loaded successfully.")
        except Exception as e:
            print(f"Fatal Error loading Dream model: {e}")
            raise
        
        try:
            print("Loading AR model...")
            self.model_a = ModelMap(ar_model_path).fetch()
            print(f"AR model loaded successfully.")
        except Exception as e:
            print(f"Fatal Error loading AR model: {e}")
            raise
        
        # Move models to device
        try:
            self.model_d = self.model_d.to(self.device).eval()
            self.model_a = self.model_a.to(self.device).eval()
            print(f"Models moved to {self.device} and set to eval mode.")
        except Exception as e:
            print(f"Fatal Error moving models to device: {e}")
            raise
    
    def generate(self, prompt, max_new_tokens=128, temperature=0.2, sampling_strategy="deterministic", verbose=False):
        """
        Generate a response to the given prompt.
        
        Args:
            prompt (str): The input prompt
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature
            sampling_strategy (str): Token matching strategy - "deterministic" or "topk_relative"
            verbose (bool): Whether to print generation progress
        
        Returns:
            str: The generated response
        """
        if verbose:
            print(f"\n--- Starting Flash-DLM generation ---")
            print(f"Prompt: {prompt}")
            print(f"Parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, sampling_strategy={sampling_strategy}")
        
        # Format prompt as messages
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Tokenize input using Dream tokenizer
            inputs = self.tok_d.apply_chat_template(
                messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
            )
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            prompt_length = input_ids.shape[1]
            
            if verbose:
                print(f"Prompt length: {prompt_length}, input_ids device: {input_ids.device}")
        
        except Exception as e:
            print(f"Error during input tokenization/processing: {e}")
            return f"Error: {e}"
        
        # Set parameters based on sampling strategy
        if sampling_strategy == "topk_relative":
            relative_threshold = 0.5  # 50% for topk_relative
            top_k = 2    # top2 relative
        else:  # deterministic
            assert sampling_strategy == "deterministic"
            relative_threshold = None
            top_k = 2    # Still need a value, even if not used
        
        # Create config for Flash-DLM generation
        config = AssistedDiffusionConfig(
            max_length=input_ids.shape[1] + max_new_tokens,
            max_new_tokens=max_new_tokens,
            mask_token_id=self.tok_d.mask_token_id,
            eos_token_id=self.tok_d.eos_token_id,
            early_stop=True,
            early_stop_consecutive=1,
            temperature=temperature,
            # top_p=top_p,
            sampling_strategy=sampling_strategy,
            confidence_threshold=0.1,
            acceptance_top_k=top_k,
            relative_threshold=relative_threshold,
            verify_batch_size=32,
            use_sliding_window_caching=True,
            sliding_window_size=(64, max_new_tokens),
            return_dict_in_generate=True,
        )
        
        # Generate response
        if verbose:
            print("Starting Flash-DLM generation...")
        
        try:
            start_time = time.time()
            output = assisted_block_diffusion_generate(
                dream_model=self.model_d,
                ar_model=self.model_a,
                input_ids=input_ids,
                attention_mask=attention_mask,
                config=config,
                dream_tokenizer=self.tok_d,
                ar_tokenizer=self.tok_a,
            )
            generation_time = time.time() - start_time
            
            if verbose:
                print(f"Flash-DLM generation completed in {generation_time:.2f} seconds")
        
        except Exception as e:
            print(f"Error during generation: {e}")
            return f"Error during model generation: {e}"
        
        # Process final result
        try:
            # Extract final tokens
            if hasattr(output, 'sequences'):
                final_tokens_tensor = output.sequences[0][prompt_length:]
            else:
                final_tokens_tensor = output[0][prompt_length:]
            
            final_tokens_list = final_tokens_tensor.tolist()
            
            # Decode final text
            final_text = self.tok_d.decode(final_tokens_list, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            
            # Count tokens generated (excluding EOS token)
            eos_token_id = self.tok_d.eos_token_id
            # Filter out EOS tokens from the count
            filtered_tokens = [token for token in final_tokens_list if token != eos_token_id]
            num_tokens_generated = len(filtered_tokens)
            
            if verbose:
                # Show generation stats
                verification_stats = getattr(output, 'verification_stats', {})
                unmasked_per_step = verification_stats.get('unmasked_tokens_per_step', [])
                total_steps = verification_stats.get('total_steps', 0)
                print(f"Generation stats: {total_steps} steps, tokens per step: {unmasked_per_step}")
                print(f"Tokens generated: {num_tokens_generated}")
                print(f"Final text generated: {final_text[:100]}{'...' if len(final_text) > 100 else ''}")
            
            # Always report token count
            print(f"\nGenerated {num_tokens_generated} tokens")
            
            return final_text
        
        except Exception as e:
            print(f"Error processing final output: {e}")
            return f"Error processing final output: {e}"

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Flash-DLM Simple API")
    parser.add_argument("prompt", nargs="?", default="Hello! How are you today?", help="The input prompt (default: 'Hello! How are you today?')")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--sampling-strategy", choices=["deterministic", "topk_relative"], default="deterministic", 
                       help="Token matching strategy: 'deterministic' or 'topk_relative' (uses top2 relative 50%)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--dream-model", default="Dream-org/Dream-Flash-Instruct-7B", help="Dream model path")
    parser.add_argument("--ar-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="AR model path")
    
    args = parser.parse_args()
    
    # Set CUDA environment variable
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    try:
        # Initialize API
        api = FlashDLMAPI(args.dream_model, args.ar_model)
        
        # Generate response
        response = api.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            sampling_strategy=args.sampling_strategy,
            verbose=args.verbose
        )
        
        print("\n" + "="*50)
        print("RESPONSE:")
        print("="*50)
        print(response)
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

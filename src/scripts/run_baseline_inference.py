import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model.opt_dream.modeling_dream import DreamForCausalLM
from src.model.opt_dream.configuration_dream import ODreamConfig
from src.model.opt_dream.generation_utils import DreamGenerationConfig

def load_step_file(step_file_path):
    """Load a step file and return the token IDs"""
    with open(step_file_path, 'r') as f:
        data = json.load(f)
    return data['input_ids']  # Assuming this is the key for token IDs

def run_baseline_inference(model, token_ids, device):
    """Run a single forward pass without KV caching"""
    # Convert token IDs to tensor
    input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
    
    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Run forward pass without KV caching
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,  # Disable KV caching
            output_attentions=True,  # Get attention weights
            return_dict=True
        )
    
    return outputs

def main():
    # Load model
    model_name = "facebook/opt-6.7b"  # Adjust based on your model
    config = ODreamConfig.from_pretrained(model_name)
    model = DreamForCausalLM.from_pretrained(model_name, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Directory containing step files
    steps_dir = "save/dream_opt/gsm8k/with_attentions/stepsDiv=1_max_gen_toks=128_alg=maskgit_alg_temp=0_temperature=0p2_block_cached_nshot=0_early_stop=1_subsample_seed=42/7/denoising_steps/sample_00000"
    
    # Create output directory
    output_dir = os.path.join(steps_dir, "baseline_inference")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each step file
    for step_file in sorted(os.listdir(steps_dir)):
        if not step_file.startswith("step_") or not step_file.endswith(".json"):
            continue
            
        step_path = os.path.join(steps_dir, step_file)
        step_num = int(step_file.split("_")[1].split(".")[0])
        
        print(f"Processing step {step_num}")
        
        # Load token IDs
        token_ids = load_step_file(step_path)
        
        # Run inference
        outputs = run_baseline_inference(model, token_ids, device)
        
        # Save results
        step_output_dir = os.path.join(output_dir, f"step_{step_num:03d}")
        os.makedirs(step_output_dir, exist_ok=True)
        
        # Save input IDs as JSON
        with open(os.path.join(step_output_dir, "input_ids.json"), 'w') as f:
            json.dump({"input_ids": token_ids}, f)
        
        # Save logits as .pt file
        torch.save(outputs.logits.cpu(), os.path.join(step_output_dir, "logits.pt"))
        
        # Save attention weights as .pt file
        torch.save(outputs.attentions, os.path.join(step_output_dir, "attentions.pt"))

if __name__ == "__main__":
    main() 
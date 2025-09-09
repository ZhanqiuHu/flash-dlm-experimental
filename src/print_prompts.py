"""
Script to print prompts from different data stages using Qwen2.5 tokenizer
"""

import os
import sys
import json
import traceback
from datasets import load_dataset

# ANSI color codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from transformers import AutoTokenizer
from src.dataset.gsm8k import GSM8K
from src.dataset.hellaswag import Hellaswag
from src.dataset.piqa import PiQA
from src.dataset.winogrande import WinoGrande
from src.dataset.gpqa import GPQA
from src.dataset.math500 import MATH500
from src.dataset.mmlu_pro import MMLUPro

# Example configs for each dataset
EXAMPLE_CONFIGS = {
    "gsm8k": {
        "dataset": {
            "name": "gsm8k",
            "split": "test",
            "nshot": 8,
            "apply_chat_template": True,
            "train": "data/gsm8k/train.jsonl",
            "test": "data/gsm8k/test.jsonl"
        },
        "eval": {
            "cot": True
        },
        "train": {
            "batch_size": 1,
            "use_accelerate": False
        },
        "save": {
            "run_dir": "runs/gsm8k",
            "logger": "gsm8k.log"
        },
        "wandb": {
            "project_name": "gsm8k",
            "experiment_name": "gsm8k",
            "flag": False
        }
    },
    "hellaswag": {
        "dataset": {
            "name": "hellaswag",
            "split": "test",
            "apply_chat_template": True,
            "train": "data/hellaswag/train.jsonl",
            "test": "data/hellaswag/test.jsonl"
        },
        "train": {
            "batch_size": 1,
            "use_accelerate": False
        },
        "save": {
            "run_dir": "runs/hellaswag",
            "logger": "hellaswag.log"
        },
        "wandb": {
            "project_name": "hellaswag",
            "experiment_name": "hellaswag",
            "flag": False
        }
    },
    "piqa": {
        "dataset": {
            "name": "piqa",
            "split": "test",
            "apply_chat_template": True,
            "train": "data/piqa/train.jsonl",
            "test": "data/piqa/test.jsonl"
        },
        "train": {
            "batch_size": 1,
            "use_accelerate": False
        },
        "save": {
            "run_dir": "runs/piqa",
            "logger": "piqa.log"
        },
        "wandb": {
            "project_name": "piqa",
            "experiment_name": "piqa",
            "flag": False
        }
    },
    "winogrande": {
        "dataset": {
            "name": "winogrande",
            "split": "test",
            "apply_chat_template": True,
            "train": "data/winogrande/train.jsonl",
            "test": "data/winogrande/test.jsonl"
        },
        "train": {
            "batch_size": 1,
            "use_accelerate": False
        },
        "save": {
            "run_dir": "runs/winogrande",
            "logger": "winogrande.log"
        },
        "wandb": {
            "project_name": "winogrande",
            "experiment_name": "winogrande",
            "flag": False
        }
    },
    "gpqa": {
        "dataset": {
            "name": "gpqa",
            "split": "test",
            "apply_chat_template": True
        },
        "train": {
            "batch_size": 1,
            "use_accelerate": False
        },
        "save": {
            "run_dir": "runs/gpqa",
            "logger": "gpqa.log"
        },
        "wandb": {
            "project_name": "gpqa",
            "experiment_name": "gpqa",
            "flag": False
        }
    },
    "math500": {
        "dataset": {
            "name": "math500",
            "split": "test",
            "apply_chat_template": True
        },
        "train": {
            "batch_size": 1,
            "use_accelerate": False
        },
        "save": {
            "run_dir": "runs/math500",
            "logger": "math500.log"
        },
        "wandb": {
            "project_name": "math500",
            "experiment_name": "math500",
            "flag": False
        }
    },
    "mmlu-pro": {
        "dataset": {
            "name": "mmlu-pro",
            "split": "test",
            "apply_chat_template": True
        },
        "train": {
            "batch_size": 1,
            "use_accelerate": False
        },
        "save": {
            "run_dir": "runs/mmlu-pro",
            "logger": "mmlu-pro.log"
        },
        "wandb": {
            "project_name": "mmlu-pro",
            "experiment_name": "mmlu-pro",
            "flag": False
        }
    }
}

def colorize_prompt(prompt):
    """Colorize different parts of the prompt."""
    if isinstance(prompt, list):
        # For MATH500 which returns a list of chat messages
        return f"{BOLD}{WHITE}{json.dumps(prompt[0] if isinstance(prompt[0], list) else prompt, indent=2)}{RESET}"
        
    # Color system message
    prompt = prompt.replace("<|im_start|>system", f"{BLUE}<|im_start|>system{RESET}")
    prompt = prompt.replace("<|im_end|>", f"{BLUE}<|im_end|>{RESET}")
    
    # Color user message
    prompt = prompt.replace("<|im_start|>user", f"{GREEN}<|im_start|>user{RESET}")
    
    # Color assistant message
    prompt = prompt.replace("<|im_start|>assistant", f"{YELLOW}<|im_start|>assistant{RESET}")
    
    # Make the entire prompt bold white
    return f"{BOLD}{WHITE}{prompt}{RESET}"

def print_dataset_info(ds):
    """Print information about the dataset structure."""
    print(f"\n{YELLOW}Dataset Structure:{RESET}")
    print(f"Available splits: {list(ds.keys())}")
    if len(ds) > 0:
        first_split = list(ds.keys())[0]
        print(f"\nExample from {first_split} split:")
        print(json.dumps(ds[first_split][0], indent=2))

def main():
    # Initialize Qwen2.5 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    
    # Initialize and print prompts for each dataset
    datasets = {
        "GSM8K": (GSM8K, "gsm8k"),
        "Hellaswag": (Hellaswag, "hellaswag"),
        "PiQA": (PiQA, "piqa"),
        "WinoGrande": (WinoGrande, "winogrande"),
        "GPQA": (GPQA, "gpqa"),
        "MATH500": (MATH500, "math500"),
        "MMLUPro": (MMLUPro, "mmlu-pro")
    }

    for name, (dataset_class, config_name) in datasets.items():
        print(f"\n{CYAN}{'='*50}")
        print(f"Dataset: {name}")
        print(f"{'='*50}{RESET}")
        
        try:
            # Get config from our example configs
            config = EXAMPLE_CONFIGS[config_name]
                
            # Initialize dataset with correct parameters
            if name in ["PiQA", "WinoGrande", "Hellaswag"]:
                dataset = dataset_class(tokenizer, config)
            elif name in ["MATH500", "MMLUPro"]:
                dataset = dataset_class(config)
            else:
                dataset = dataset_class(config, tokenizer)
            
            # Get wrapped text
            if name == "GSM8K":
                # For GSM8K, load from HuggingFace
                ds = load_dataset("gsm8k", "main")
                print_dataset_info(ds)
                # Get first test example
                test_example = ds["test"][0]
                example = {"instruction": test_example["question"], "output": test_example["answer"]}
                # Get n-shot examples from train set
                train_examples = ds["train"][:dataset.nshot]
                nshot_examples = [{"instruction": ex["question"], "output": ex["answer"]} for ex in train_examples]
                # Format the prompt with n-shot examples
                wrapped = dataset.wrap_text(example, nshot_examples)
                # Add prompt head and tail
                wrapped = dataset.prompt_head + wrapped + dataset.prompt_tail
                # Create chat message
                wrapped = [{"role": "user", "content": wrapped}]
            elif name == "GPQA":
                # For GPQA, use wrap_single to format the question and answers
                ds = load_dataset("idavidrein/gpqa", "gpqa_main")
                print_dataset_info(ds)
                wrapped = dataset.wrap_single(ds["train"][0])
                # Add prompt head and tail
                wrapped = dataset.prompt_head + wrapped + dataset.prompt_tail
                # Create chat message
                wrapped = [{"role": "user", "content": wrapped}]
            elif name == "MMLUPro":
                # For MMLUPro, we need to load the dataset first to get validation examples
                testset, validset = dataset.load_dataset()
                # Get first test example
                first_test = testset[0]
                # Get validation examples for the same category
                validset = dataset.select_by_category(validset, first_test["category"])
                # Get wrapped text with validation examples
                wrapped, _ = dataset.wrap_text(validset, first_test)
            elif name == "PiQA":
                # For PiQA, use the test set from HuggingFace
                ds = load_dataset("piqa")
                print_dataset_info(ds)
                testset = ds["validation"]  # PiQA uses validation as test
                wrapped = dataset.wrap_text({"goal": testset[0]["goal"], "sol1": testset[0]["sol1"], "sol2": testset[0]["sol2"]})
            elif name == "Hellaswag":
                # For Hellaswag, load from HuggingFace
                ds = load_dataset("hellaswag")
                print_dataset_info(ds)
                testset = ds["validation"]  # Hellaswag uses validation as test
                wrapped = dataset.wrap_text({"instruction": testset[0]["ctx"] + " " + testset[0]["endings"][0], "answer": testset[0]["label"]})
            elif name == "WinoGrande":
                # For WinoGrande, load from HuggingFace
                ds = load_dataset("winogrande", "winogrande_xl")
                print_dataset_info(ds)
                testset = ds["validation"]  # WinoGrande uses validation as test
                wrapped = dataset.wrap_text({"instruction": testset[0]["sentence"], "answer": testset[0]["answer"]})
            else:
                # For other datasets, get the test set and use first example
                testset = dataset.run()
                wrapped = dataset.wrap_text(testset["dataset"][0])
            
            # Apply chat template
            if hasattr(dataset, 'apply_chat_template') and dataset.apply_chat_template:
                if name == "MATH500":
                    # MATH500 already returns chat format
                    wrapped = wrapped
                else:
                    wrapped = tokenizer.apply_chat_template(wrapped, tokenize=False, add_generation_prompt=True)
            
            # Colorize and print the prompt
            colored_prompt = colorize_prompt(wrapped)
            print(f"\n{MAGENTA}Prompt:{RESET}")
            print("-"*30)
            print(colored_prompt)
            print("-"*30)
            
        except Exception as e:
            print(f"{RED}Error processing {name}:{RESET}")
            print(f"{RED}Error type: {type(e).__name__}{RESET}")
            print(f"{RED}Error message: {str(e)}{RESET}")
            print(f"\n{RED}Full traceback:{RESET}")
            traceback.print_exc()
            print(f"\n{RED}Config used:{RESET}")
            print(json.dumps(config, indent=2))

if __name__ == "__main__":
    main() 
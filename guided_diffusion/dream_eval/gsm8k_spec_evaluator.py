#!/usr/bin/env python
import re
import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parent.parent.parent
sys.path += [str(root), str(root / "src")]

from src.dataset.gsm8k import GSM8K
from guided_diffusion.dream_eval.base_spec_evaluator import BaseSpecEvaluator

# ANSI color codes for output formatting
BLUE = "\033[94m"  # bright blue
RESET = "\033[0m"  # reset color

def exact_match(ans: str, ref: str) -> bool:
    """Extract numerical answer from prediction and compare with ground truth."""
    ANS_RE = re.compile(r"####\s*(-?[0-9\.,]+)")
    match = ANS_RE.search(ref)
    if not match:
        return False
    
    gt_str = match.group(1).strip()
    gt_str = gt_str.replace(",", "")

    # extract numerical answer from prediction
    preds = ans.split("The final answer is")
    valid_ans = True if len(preds) > 1 else False

    if valid_ans:
        pred = preds[1]
    else:
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return False

    if valid_ans:
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    if pred[-1] == ".":
        pred = pred[:-1]

    # Try numerical comparison first
    try:
        pred_float = float(pred)
        gt_float = float(gt_str)
        correctness = pred_float == gt_float
    except:
        correctness = pred == gt_str

    # Print colored output
    BOLD = "\033[1m"
    if not correctness:
        RED = "\033[91m"  # bright red    
        print(f"{RED}❌  Pred: {BOLD}{pred}{RESET}{RED} != Gold: {BOLD}{gt_str}{RESET}",
            flush=True)
    else:
        GREEN = "\033[92m"  # bright green
        print(f"{GREEN}✅  Pred: {BOLD}{pred}{RESET}{GREEN} == Gold: {BOLD}{gt_str}{RESET}",
            flush=True)

    return correctness

def extract_answer(text: str) -> str:
    """Extract the numerical answer from generated text."""
    # Look for "The final answer is" pattern
    if "The final answer is" in text:
        parts = text.split("The final answer is")
        if len(parts) > 1:
            answer_part = parts[1].strip()
            # Extract the first number from the answer part
            numbers = re.findall(r"-?\d+\.?\d*", answer_part)
            if numbers:
                return numbers[0]
    
    # Fallback: extract the last number from the entire text
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
    
    return ""

class GSM8KSpecEvaluator(BaseSpecEvaluator):
    """GSM8K speculative diffusion evaluator using the base evaluator."""
    
    def __init__(self, cfg_path: str):
        super().__init__(
            cfg_path=cfg_path,
            DatasetClass=GSM8K,
            answer_extractor=extract_answer,
            match_fn=exact_match
        )
        self._tqdm_desc = f"GSM8K-{self.cfg['model']['model_type']}"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("GSM8K speculative evaluation using base evaluator")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    evaluator = GSM8KSpecEvaluator(args.config)
    evaluator.run_evaluation() 
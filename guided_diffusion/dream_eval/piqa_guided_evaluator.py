#!/usr/bin/env python
import re
import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parent.parent.parent
sys.path += [str(root), str(root / "src")]

from src.dataset.piqa import PiQA
from guided_diffusion.dream_eval.base_guided_evaluator import (
    setup_logging,
    parse_args,
    BaseGuidedEvaluator,
)

class PiQADatasetWrapper:
    """Wrapper to make PiQA dataset compatible with base evaluator interface."""
    def __init__(self, cfg_path, tokenizer):
        self.dataset = PiQA(tokenizer, cfg_path)
    
    def run(self):
        trainset, testset = self.dataset.run()
        # Return only the test set in the format expected by base evaluator
        return testset

# ANSI color codes for output formatting
BLUE = "\033[94m"  # bright blue
RED = "\033[91m"   # bright red
GREEN = "\033[92m" # bright green
BOLD = "\033[1m"   # bold
RESET = "\033[0m"  # reset color

def exact_match(pred: str, gt: str) -> bool:
    """Check if the predicted answer matches the ground truth."""
    # Ground truth is like "solution1" or "solution2"
    gt_solution = gt  # e.g., "solution1"
    
    # Extract solution1 or solution2 from the prediction (case insensitive)
    pred_answers = re.findall(r'solution1|solution2', pred.lower())

    if len(pred_answers) > 0:
        pred_answer = pred_answers[0]  # e.g., "solution1"
        correctness = pred_answer == gt_solution
    else:
        pred_answer = ""
        correctness = False

    # Print colored output
    if not correctness:
        print(f"{RED}❌  Pred: {BOLD}{pred_answer}{RESET}{RED} != Gold: {BOLD}{gt_solution}{RESET}",
            flush=True)
    else:
        print(f"{GREEN}✅  Pred: {BOLD}{pred_answer}{RESET}{GREEN} == Gold: {BOLD}{gt_solution}{RESET}",
            flush=True)

    return correctness

def extract_answer(text: str) -> str:
    """Extract the answer from generated text."""
    # Look for "The final answer is" pattern (case insensitive)
    if "the final answer is" in text.lower():
        parts = text.lower().split("the final answer is")
        if len(parts) > 1:
            answer_part = parts[1].strip()
            # Extract solution1 or solution2 (case insensitive)
            solutions = re.findall(r'solution1|solution2|solution 1|solution 2', answer_part.lower())
            if solutions:
                # Convert back to lowercase to match ground truth format
                return solutions[0].lower()  # Return the full solution string in lowercase
    
    return ""

class PiQAGuidedEvaluator(BaseGuidedEvaluator):
    """PiQA guided diffusion evaluator using Dream Flash model with AR verifier."""
    
    def __init__(self, cfg_path: str):
        super().__init__(
            cfg_path,
            PiQADatasetWrapper,
            answer_extractor=extract_answer,
            match_fn=exact_match,
        )

    def run_evaluation(self):
        # Load PiQA data
        # PiQA.run() returns (trainset, testset) where testset is the dict we need
        all_data = self.DatasetClass(self.cfg_path, self.tok_d).run()

        # PiQA returns a single dataset, not per-subject like MMLU-Pro
        self.pairs = list(zip(all_data["dataset"], all_data["label"]))

        # header + bar description
        print(f"\n=== Evaluating PiQA ({len(self.pairs)} examples) ===\n")
        self._tqdm_desc = f"{self.cfg['model']['model_type']} on PiQA"

        # run base logic (with its own progress bar)
        super().run_evaluation()


def main():
    setup_logging()
    cfg = parse_args()
    PiQAGuidedEvaluator(cfg).run_evaluation()


if __name__ == "__main__":
    main()

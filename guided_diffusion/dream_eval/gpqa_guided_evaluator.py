#!/usr/bin/env python
import re
import sys
from pathlib import Path

# ensure repo root on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from guided_diffusion.dream_eval.base_guided_evaluator import (
    setup_logging,
    parse_args,
    BaseGuidedEvaluator,
)
from src.dataset.gpqa import GPQA


def extract_answer(text: str) -> str:
    """
    Extract the "AnswerX" tag from the model's output text.
    Look for "answer is AnswerX" pattern.
    """
    text = text.strip()
    # Look for "The final answer is AnswerX" pattern (case-insensitive)
    m = re.search(r'answer is\s+(Answer[1-4])', text, re.IGNORECASE)
    if m:
        return m.group(1)
    else:
        # Fallback: look for standalone Answer1, Answer2, Answer3, or Answer4
        m = re.search(r'Answer\s*([1-4])', text, re.IGNORECASE)
        if m:
            return f"Answer{m.group(1)}"
        else:
            return ""


def exact_match(ans: str, ref: str) -> bool:
    """
    Check if the predicted answer matches the reference answer.
    Both should be in format "Answer1", "Answer2", etc.
    """
    return ans == ref


class GPQAGuidedEvaluator(BaseGuidedEvaluator):
    """
    GPQA guided diffusion evaluator using Dream Flash model with AR verifier.
    """
    
    def __init__(self, cfg_path: str):
        super().__init__(
            cfg_path,
            GPQA,
            answer_extractor=extract_answer,
            match_fn=exact_match,
        )

    def run_evaluation(self):
        # Load GPQA data
        # GPQA.run() returns (None, testset) where testset is the dict we need
        _, all_data = self.DatasetClass(self.cfg_path, self.tok_d).run()

        # GPQA returns a single dataset, not per-subject like MMLU-Pro
        self.pairs = list(zip(all_data["dataset"], all_data["label"]))

        # header + bar description
        print(f"\n=== Evaluating GPQA ({len(self.pairs)} examples) ===\n")
        self._tqdm_desc = f"{self.cfg['model']['model_type']} on GPQA"

        # run base logic (with its own progress bar)
        super().run_evaluation()


def main():
    setup_logging()
    cfg = parse_args()
    GPQAGuidedEvaluator(cfg).run_evaluation()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
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
from src.dataset.mmlu_pro import MMLUPro, extract_answer


class MMLUProGuidedEvaluator(BaseGuidedEvaluator):
    """
    MMLU-Pro guided diffusion evaluator using Dream Flash model with AR verifier.
    """
    
    def __init__(self, cfg_path: str):
        # Initialize the base class but override the dataset loading
        self.cfg_path = cfg_path
        
        # Load config first
        import yaml
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        
        # Initialize the base class components manually to avoid the dataset loading issue
        from guided_diffusion.dream_eval.base_guided_evaluator import setup_logging, get_node_name, get_gpu_info
        from pathlib import Path
        import logging
        from logging import Formatter
        import torch
        import wandb
        
        # Setup logging
        setup_logging()
        
        # Create run directory
        base = Path(self.cfg["save"]["run_dir"])
        run_dir = base / f"{get_node_name()}_{get_gpu_info()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        
        # Setup file logger
        log_fname = self.cfg["save"].get("logger", "evaluation.log")
        log_path = run_dir / log_fname
        print(f"Logs will be saved to: {log_path}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(fh)
        
        # Setup eval logger
        from guided_diffusion.dream_eval.base_guided_evaluator import GuidedDiffusionLogger
        self.eval_logger = GuidedDiffusionLogger(str(run_dir/"eval_results.json"))
        self.eval_logger.save()
        
        # Setup wandb
        wandb.init(
            project=self.cfg["save"].get("wandb_project","spec_diffusion"),
            config=self.cfg,
            dir=str(run_dir),
        )
        
        # Setup devices
        if torch.cuda.is_available() and torch.cuda.device_count()>=2:
            self.draft_dev = torch.device("cuda:0")
            self.ar_dev    = torch.device("cuda:1")
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.draft_dev = self.ar_dev = dev
        
        # Load models and tokenizers
        from src.model.auto_map import ModelMap
        from guided_diffusion.dream_eval.base_guided_evaluator import load_tokenizer
        
        dname = self.cfg["model"]["draft_model"]
        aname = self.cfg["model"]["model_type"]
        self.model_d = ModelMap(dname).fetch().to(self.draft_dev)
        self.tok_d   = load_tokenizer(dname)
        self.model_a = ModelMap(aname).fetch(device_map={"":self.ar_dev}).to(self.ar_dev)
        self.tok_a   = load_tokenizer(aname)
        
        # Setup verifier
        from guided_diffusion.guided_diff_utils import ARAssistant
        from guided_diffusion.spec_diff_utils import ARVerifier
        
        if self.cfg["dream"].get("use_assisted",False):
            self.verifier = ARAssistant(self.model_a, self.tok_d, self.tok_a)
        else:
            self.verifier = ARVerifier(self.model_a, self.tok_d, self.tok_a)
        
        # Set up dataset class and answer functions
        self.DatasetClass = MMLUPro
        self.answer_extractor = extract_answer
        self.match_fn = lambda p, g: p == g

    def run_evaluation(self):
        # Load per-subject data
        all_data = self.DatasetClass(self.cfg_path).run()

        for subject, data in all_data.items():
            # reset pairs
            self.pairs = list(zip(data["sample"], data["label"]))

            # new JSON logger per subject
            subj_dir = self.run_dir / subject
            subj_dir.mkdir(parents=True, exist_ok=True)
            self.eval_logger.log_path = str(subj_dir / "eval_results.json")

            # header + bar description
            print(f"\n=== Evaluating subject \"{subject}\" ({len(self.pairs)} examples) ===\n")
            self._tqdm_desc = f"{self.cfg['model']['model_type']} on {subject}"

            # run base logic (with its own progress bar)
            super().run_evaluation()


def main():
    setup_logging()
    cfg = parse_args()
    MMLUProGuidedEvaluator(cfg).run_evaluation()


if __name__ == "__main__":
    main()

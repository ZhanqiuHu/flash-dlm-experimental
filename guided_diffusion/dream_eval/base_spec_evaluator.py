#!/usr/bin/env python
import logging
from logging import Formatter
import warnings
import argparse
import torch
import platform
import yaml
import time
from pathlib import Path
from tqdm import tqdm
import wandb
from transformers.utils import logging as hf_logging
from transformers import AutoTokenizer, set_seed

# ANSI color codes
BLUE = "\033[94m"  # bright blue
RED = "\033[91m"   # bright red
GREEN = "\033[92m" # bright green
BOLD = "\033[1m"   # bold
RESET = "\033[0m"  # reset color

from src.model.auto_map import ModelMap
from guided_diffusion.diff_spec_utils_shifted import (
    speculative_diffusion_generate,
    speculative_block_diffusion_generate,
    SpecDiffusionConfig,
    ARVerifier,
)
from guided_diffusion.diff_spec_utils_assisted import (
    # assisted_diffusion_generate,
    assisted_block_diffusion_generate,
    AssistedDiffusionConfig,
    ARAssistant,
)


def setup_logging(level=logging.INFO):
    # root = logging.getLogger()
    # for h in list(root.handlers):
    #     root.removeHandler(h)
    # ch = logging.StreamHandler()
    # ch.setFormatter(Formatter("%(message)s"))
    # root.addHandler(ch)
    # root.setLevel(level)
    # hf_logging.set_verbosity_info()
    # hf_logging.disable_default_handler()
    # warnings.filterwarnings("ignore", category=UserWarning)
    pass

def parse_args():
    p = argparse.ArgumentParser(description="Speculative Diffusion Evaluator")
    p.add_argument("--config", required=True, help="YAML config path")
    return p.parse_args().config

def load_tokenizer(name: str):
    canon = (name.replace("-Custom","")
                 .replace("-Opt","")
                 .replace("-Block-Cached","")
                 .replace("-Flash","-v0")
                 .replace("-v2","-v0")
                 .replace("-v3","-v0"))
    tok = AutoTokenizer.from_pretrained(canon, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    return tok

def get_gpu_info() -> str:
    if torch.cuda.is_available():
        cnt = torch.cuda.device_count()
        nm  = torch.cuda.get_device_name(0).replace(" ","_")
        return f"{cnt}x{nm}"
    return "cpu"

def get_node_name() -> str:
    return platform.node().replace(".","_")

class GuidedDiffusionLogger:
    """Custom logger for guided diffusion evaluation results."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.results = []
        
    def log(self, 
            idx: int, 
            sample_id: int,
            steps: int,
            max_new_tokens: int,
            latency_ms: float,
            input_length: int,
            generated_length: int,
            prompt: str,
            generated_text: str,
            ground_truth: str,
            is_correct: bool,
            time_breakdown: dict = None):
        """Log a single evaluation result."""
        result = {
            "idx": idx,
            "sample_id": sample_id,
            "steps": steps,
            "max_new_tokens": max_new_tokens,
            "latency_ms": latency_ms,
            "input_length": input_length,
            "generated_length": generated_length,
            "prompt": prompt,
            "generated_text": generated_text,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "time_breakdown": time_breakdown or {},
            "timestamp": time.time()
        }
        self.results.append(result)
        
    def save(self):
        """Save all logged results to JSON file."""
        import json
        with open(self.log_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {self.log_path}")

class BaseSpecEvaluator:
    """
    Shared speculative-diffusion evaluator.
    Subclasses must pass:
      - cfg_path
      - DatasetClass
      - answer_extractor: fn(text)->str
      - match_fn: fn(predicted, gold)->bool
    """
    def __init__(self,
                 cfg_path: str,
                 DatasetClass,
                 answer_extractor: callable,
                 match_fn: callable):
        self.cfg_path         = cfg_path
        self.DatasetClass     = DatasetClass
        self.answer_extractor = answer_extractor
        self.match_fn         = match_fn
        self._tqdm_desc       = None

        # load config
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        # logging + run_dir
        setup_logging()
        base = Path(self.cfg["save"]["run_dir"])
        run_dir = base / f"{get_node_name()}_{get_gpu_info()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir

        # file logger
        log_fname = self.cfg["save"].get("logger", "evaluation.log")
        log_path = run_dir / log_fname
        print(f"Logs will be saved to: {log_path}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(fh)

        # JSON logger + wandb
        self.eval_logger = GuidedDiffusionLogger(str(run_dir/"eval_results.json"))
        self.eval_logger.save()
        wandb.init(
          project=self.cfg["save"].get("wandb_project","spec_diffusion"),
          config=self.cfg,
          dir=str(run_dir),
        )

        # devices
        if torch.cuda.is_available() and torch.cuda.device_count()>=2:
            self.draft_dev = torch.device("cuda:0")
            self.ar_dev    = torch.device("cuda:1")
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.draft_dev = self.ar_dev = dev

        # models + tokenizers
        dname = self.cfg["model"]["draft_model"]
        aname = self.cfg["model"]["model_type"]
        self.model_d = ModelMap(dname).fetch().to(self.draft_dev)
        self.tok_d   = load_tokenizer(dname)
        self.model_a = ModelMap(aname).fetch(device_map={"":self.ar_dev}).to(self.ar_dev)
        self.tok_a   = load_tokenizer(aname)

        # verifier
        if self.cfg["dream"].get("use_assisted",False):
            self.verifier = ARAssistant(self.model_a, self.tok_d, self.tok_a)
        else:
            self.verifier = ARVerifier(self.model_a, self.tok_d, self.tok_a)

        # load & flatten dataset into self.pairs
        raw = DatasetClass(cfg_path, self.tok_d).run()
        if isinstance(raw, dict):
            # Check if it's MMLU-Pro style (nested dicts with "sample"/"label")
            if any(isinstance(v, dict) and "sample" in v and "label" in v for v in raw.values()):
                # MMLU-Pro style
                self.pairs = [
                    (p,l)
                    for v in raw.values()
                    for p,l in zip(v["sample"], v["label"])
                ]
            elif "dataset" in raw and "label" in raw:
                # PiQA style: simple dict with "dataset"/"label" keys
                self.pairs = list(zip(raw["dataset"], raw["label"]))
            else:
                # fallback for other dict formats
                raise ValueError(f"Unsupported dataset format: {raw.keys()}")
        else:
            # GSM8K style: tuple(_, split)
            if isinstance(raw, tuple):
                _, split = raw
                self.pairs = list(zip(split["dataset"], split["label"]))
            else:
                # fallback
                self.pairs = raw

        # Apply random subsampling if max_samples is specified
        max_samples = self.cfg["eval"].get("max_samples", None)
        if max_samples is not None and max_samples > 0:
            if max_samples < len(self.pairs):
                # Set seed for reproducible random sampling
                subsample_seed = self.cfg["eval"].get("subsample_seed", 42)
                import random
                random.seed(subsample_seed)
                
                # Randomly sample without replacement
                self.pairs = random.sample(self.pairs, max_samples)
                logging.info(f"Randomly subsampled dataset to {max_samples} samples (seed: {subsample_seed}) out of {len(raw) if isinstance(raw, tuple) else len(self.pairs)} total")
            else:
                logging.info(f"max_samples ({max_samples}) >= dataset size ({len(self.pairs)}), using full dataset")

        set_seed(self.cfg.get("seed",5000))

    def run_evaluation(self):
        # print config
        logging.info("="*60)
        logging.info("Configuration:")
        logging.info("="*60)
        for k,v in self.cfg.items():
            if isinstance(v, dict):
                logging.info(f"{k}:")
                for kk,vv in v.items(): logging.info(f"  {kk}: {vv}")
            else:
                logging.info(f"{k}: {v}")
        logging.info("="*60)

        # stats
        tot_corr = tot_lat = tot_act = tot_in = tot_steps = tot_ar_calls = tot_rejections = 0
        max_new     = self.cfg["eval"]["max_gen_toks"]
        use_block   = self.cfg["dream"].get("use_block_diffusion",False)
        use_assist  = self.cfg["dream"].get("use_assisted",False)
        block_size  = self.cfg["dream"].get("block_size",None)
        
        desc = self._tqdm_desc or f"{self.cfg['model']['model_type']}"
        bar = tqdm(self.pairs,
                    total=len(self.pairs),
                    desc=desc,
                    postfix={"acc":"0.00%","avg_lat":"0.00s"})

        for idx,(prompt,gold) in enumerate(bar):
            # encode
            enc = self.tok_d(prompt, return_tensors="pt")
            inp = enc.input_ids.to(self.draft_dev)
            att = enc.attention_mask.to(self.draft_dev)
            curr_in = inp.shape[1]; tot_in += curr_in

            # Log current input token length and running average
            logging.info(f"Sample {idx+1}/{len(self.pairs)} - Input tokens: {curr_in}, Running avg: {tot_in/(idx+1):.4f}")

            # build config
            C = AssistedDiffusionConfig if use_assist else SpecDiffusionConfig
            cfg = C(
                max_length=inp.shape[1]+max_new,
                max_new_tokens=max_new,
                mask_token_id=self.tok_d.mask_token_id,
                eos_token_id=self.tok_d.eos_token_id,
                early_stop=self.cfg["dream"].get("early_stop",False),
                early_stop_consecutive=self.cfg["dream"].get("early_stop_consecutive",1),
                temperature=self.cfg["dream"].get("temperature",0.2),
                top_p=self.cfg["dream"].get("top_p",0.95),
                sampling_strategy=self.cfg["dream"].get("sampling_strategy","deterministic"),
                confidence_threshold=self.cfg["dream"].get("confidence_threshold",0.1),
                use_sliding_window_caching=self.cfg["dream"].get("use_sliding_window_caching",False),
                sliding_window_size=self.cfg["dream"].get("sliding_window_size",128),
                use_block_boundary_caching=self.cfg["dream"].get("use_block_boundary_caching",False),
                stop_on_dream_eos=self.cfg["dream"].get("stop_on_dream_eos",False),
                return_dict_in_generate=True,
            )
            if use_block: cfg.block_size = block_size

            # generate with detailed timing breakdown
            t0 = time.perf_counter()
            if use_block:
                fn = assisted_block_diffusion_generate if use_assist else speculative_block_diffusion_generate
            else:
                raise NotImplementedError("Assisted diffusion is not implemented for non-block diffusion")
                fn = assisted_diffusion_generate if use_assist else speculative_diffusion_generate

            out = fn(
                dream_model=self.model_d,
                ar_model=self.model_a,
                input_ids=inp,
                attention_mask=att,
                config=cfg,
                dream_tokenizer=self.tok_d,
                ar_tokenizer=self.tok_a,
                ar_verifier=self.verifier,
            )
            total_lat = (time.perf_counter()-t0)*1000; tot_lat += total_lat

            # Log current latency
            logging.info(f"Sample {idx+1} - Generation time: {total_lat/1000:.4f}s")

            # decode with proper length tracking
            seq = out.sequences if not isinstance(out, torch.Tensor) else out
            gen_ids = seq[0, inp.size(1):].tolist()
            input_length = inp.size(1)
            
            # Find first EOS token and include it in generated length
            try:
                eos_idx = gen_ids.index(self.tok_d.eos_token_id)
                generated_length = eos_idx + 1  # Include the EOS token
                gen_ids_for_text = gen_ids[:eos_idx]  # Exclude EOS for text decoding
            except ValueError:
                generated_length = len(gen_ids)  # No EOS found
                gen_ids_for_text = gen_ids
            
            txt = self.tok_d.decode(gen_ids_for_text, skip_special_tokens=True)
            tot_act += generated_length

            # Get verification stats
            verification_stats = None
            if hasattr(out, 'verification_stats'):
                verification_stats = out.verification_stats
                curr_steps = verification_stats.get('total_steps', 0)
                curr_ar_calls = verification_stats.get('ar_model_calls', 0)
                curr_rejections = verification_stats.get('rejection_correction_calls', 0)
                tot_steps += curr_steps
                tot_ar_calls += curr_ar_calls
                tot_rejections += curr_rejections

            # Log generation length and statistics
            logging.info(f"Sample {idx+1} - Generated {generated_length}/{max_new} tokens")
            if verification_stats:
                logging.info(f"Sample {idx+1} - Denoising steps: {curr_steps}, AR model calls: {curr_ar_calls}, Rejection corrections: {curr_rejections}")
                logging.info(f"Sample {idx+1} - Tokens per step: {generated_length/curr_steps:.4f}")
                
                # Calculate timing breakdown
                if curr_steps > 0:
                    avg_time_per_step = total_lat / curr_steps
                    avg_time_per_token = total_lat / generated_length if generated_length > 0 else 0
                    logging.info(f"Sample {idx+1} - Timing breakdown - Avg time/step: {avg_time_per_step:.2f}ms, Avg time/token: {avg_time_per_token:.2f}ms")
                
                if 'unmasked_tokens_per_step' in verification_stats:
                    logging.info(f"Sample {idx+1} - Unmasked tokens per step: {verification_stats['unmasked_tokens_per_step']}")
                    print(f"Sample {idx+1} - Unmasked tokens per step: {verification_stats['unmasked_tokens_per_step']}")
                    print(f"Sample {idx+1} - Input length: {input_length}, Output length: {generated_length}")

            # Log running averages
            if tot_steps > 0:
                avg_steps = tot_steps/(idx+1)
                avg_tokens_per_step = ((tot_act + (idx+1))/(idx+1))/avg_steps  # Add 1 EOS token per sample
                avg_actual_tokens = tot_act/(idx+1)
                avg_latency = tot_lat/(idx+1)
                logging.info(f"Sample {idx+1} - Running averages - Steps: {avg_steps:.4f}, Tokens/step: {avg_tokens_per_step:.4f}, Actual tokens/sample: {avg_actual_tokens:.4f}, Avg latency: {avg_latency:.2f}ms")
                print(f"Running averages - Steps: {avg_steps:.4f}, {BLUE}Tokens/step: {avg_tokens_per_step:.4f}, Actual tokens/sample: {avg_actual_tokens:.4f}, Avg latency: {avg_latency:.2f}ms{RESET}")

            # extract & match
            pred = self.answer_extractor(txt)
            corr = self.match_fn(pred, gold)
            tot_corr += int(corr)
            
            # Log prediction and correctness
            logging.info(f"Sample {idx+1} - Prediction: '{pred}', Ground truth: '{gold}', Correct: {corr}")

            # Print colored output for prediction
            if not corr:
                print(f"{RED}❌  Pred: {BOLD}{pred}{RESET}{RED} != Gold: {BOLD}{gold}{RESET}\n")
            else:
                print(f"{GREEN}✅  Pred: {BOLD}{pred}{RESET}{GREEN} == Gold: {BOLD}{gold}{RESET}")

            # log
            time_breakdown = verification_stats.get('time_breakdown', {}) if verification_stats else {}
            self.eval_logger.log(
                idx, 1,
                getattr(cfg,"steps",None),
                max_new,
                total_lat,
                input_length,
                generated_length,
                prompt,
                txt,
                gold,
                corr,
                time_breakdown
            )
            if (idx+1)%100==0:
                self.eval_logger.save()

            # update bar
            bar.set_postfix({
                "acc": f"{tot_corr/(idx+1):.2%}",
                "avg_lat": f"{tot_lat/(idx+1)/1000:.3f}s"
            })

        # final summary with detailed runtime breakdown
        acc = tot_corr/len(self.pairs)
        avg_lat = tot_lat/len(self.pairs)/1000
        avg_act = tot_act/len(self.pairs)
        avg_in = tot_in/len(self.pairs)
        
        logging.info("="*60)
        logging.info("FINAL EVALUATION SUMMARY")
        logging.info("="*60)
        logging.info(f"Accuracy: {acc*100:.2f}% ({tot_corr}/{len(self.pairs)})")
        logging.info(f"Average latency: {avg_lat:.3f}s per sample")
        logging.info(f"Average prompt length: {avg_in:.2f} tokens per sample")
        logging.info(f"Average generated length: {avg_act:.2f} tokens per sample")
        
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Accuracy: {acc*100:.2f}% ({tot_corr}/{len(self.pairs)})")
        print(f"Average latency: {avg_lat:.3f}s per sample")
        print(f"Average prompt length: {avg_in:.2f} tokens per sample")
        print(f"Average generated length: {avg_act:.2f} tokens per sample")
        
        # Runtime breakdown
        if tot_steps > 0:
            avg_steps = tot_steps/len(self.pairs)
            avg_tokens_per_step = (tot_act + len(self.pairs))/tot_steps  # Add 1 EOS token per sample
            avg_time_per_step = tot_lat/tot_steps
            avg_time_per_token = tot_lat/tot_act if tot_act > 0 else 0
            
            logging.info("TIMING METRICS:")
            logging.info(f"Average denoising steps: {avg_steps:.2f} per sample")
            logging.info(f"Average tokens per step: {avg_tokens_per_step:.3f}")
            logging.info(f"Average time per step: {avg_time_per_step:.2f}ms")
            logging.info(f"Average time per token: {avg_time_per_token:.2f}ms")
            logging.info(f"Average tokens per second: {1000/avg_time_per_token:.1f}" if avg_time_per_token > 0 else "Average tokens per second: 0.0")
            
            print(f"\nTIMING METRICS:")
            print(f"Average denoising steps: {avg_steps:.2f} per sample")
            print(f"Average tokens per step: {avg_tokens_per_step:.3f}")
            print(f"Average time per step: {avg_time_per_step:.2f}ms")
            print(f"Average time per token: {avg_time_per_token:.2f}ms")
            print(f"Average tokens per second: {1000/avg_time_per_token:.1f}" if avg_time_per_token > 0 else "Average tokens per second: 0.0")
            
            # Detailed time analysis from verification stats
            if verification_stats and 'time_breakdown' in verification_stats:
                time_bd = verification_stats['time_breakdown']
                if time_bd['diffusion_prediction_time']:
                    # All steps breakdown
                    avg_dream_model_forward_time = sum(time_bd['dream_model_forward_time']) / len(time_bd['dream_model_forward_time'])
                    avg_dream_overhead_time = sum(time_bd['dream_overhead_time']) / len(time_bd['dream_overhead_time'])
                    avg_diffusion_time = sum(time_bd['diffusion_prediction_time']) / len(time_bd['diffusion_prediction_time'])
                    avg_ar_model_time = sum(time_bd['ar_model_forward_time']) / len(time_bd['ar_model_forward_time'])
                    avg_verification_strategy_time = sum(time_bd['verification_strategy_time']) / len(time_bd['verification_strategy_time'])
                    avg_overhead_time = sum(time_bd['overhead_time']) / len(time_bd['overhead_time'])
                    avg_total_step_time = sum(time_bd['total_step_time']) / len(time_bd['total_step_time'])
                    
                    # Calculate total AR time
                    avg_total_ar_time = avg_ar_model_time + avg_verification_strategy_time
                    
                    logging.info("DETAILED TIME ANALYSIS PER STEP (ALL STEPS):")
                    logging.info(f"Average Dream model forward time: {avg_dream_model_forward_time:.2f}ms")
                    logging.info(f"Average Dream overhead time: {avg_dream_overhead_time:.2f}ms")
                    logging.info(f"Average diffusion prediction time: {avg_diffusion_time:.2f}ms")
                    logging.info(f"Average AR model forward time: {avg_ar_model_time:.2f}ms")
                    logging.info(f"Average verification strategy time: {avg_verification_strategy_time:.2f}ms")
                    logging.info(f"Average total AR time: {avg_total_ar_time:.2f}ms")
                    logging.info(f"Average overhead time: {avg_overhead_time:.2f}ms")
                    logging.info(f"Average total step time: {avg_total_step_time:.2f}ms")
                    
                    print(f"\nDETAILED TIME ANALYSIS PER STEP (ALL STEPS):")
                    print(f"Average Dream model forward time: {avg_dream_model_forward_time:.2f}ms")
                    print(f"Average Dream overhead time: {avg_dream_overhead_time:.2f}ms")
                    print(f"Average diffusion prediction time: {avg_diffusion_time:.2f}ms")
                    print(f"Average AR model forward time: {avg_ar_model_time:.2f}ms")
                    print(f"Average verification strategy time: {avg_verification_strategy_time:.2f}ms")
                    print(f"Average total AR time: {avg_total_ar_time:.2f}ms")
                    print(f"Average overhead time: {avg_overhead_time:.2f}ms")
                    print(f"Average total step time: {avg_total_step_time:.2f}ms")
                    
                    # Calculate percentages
                    total_time = avg_dream_model_forward_time + avg_dream_overhead_time + avg_total_ar_time + avg_overhead_time
                    if total_time > 0:
                        logging.info("TIME ANALYSIS PERCENTAGES (ALL STEPS):")
                        logging.info(f"Dream model forward: {avg_dream_model_forward_time/total_time*100:.1f}%")
                        logging.info(f"Dream overhead: {avg_dream_overhead_time/total_time*100:.1f}%")
                        logging.info(f"AR model forward: {avg_ar_model_time/total_time*100:.1f}%")
                        logging.info(f"Verification strategy: {avg_verification_strategy_time/total_time*100:.1f}%")
                        logging.info(f"Total AR time: {avg_total_ar_time/total_time*100:.1f}%")
                        logging.info(f"Overhead: {avg_overhead_time/total_time*100:.1f}%")
                        
                        print(f"\nTIME ANALYSIS PERCENTAGES (ALL STEPS):")
                        print(f"Dream model forward: {avg_dream_model_forward_time/total_time*100:.1f}%")
                        print(f"Dream overhead: {avg_dream_overhead_time/total_time*100:.1f}%")
                        print(f"AR model forward: {avg_ar_model_time/total_time*100:.1f}%")
                        print(f"Verification strategy: {avg_verification_strategy_time/total_time*100:.1f}%")
                        print(f"Total AR time: {avg_total_ar_time/total_time*100:.1f}%")
                        print(f"Overhead: {avg_overhead_time/total_time*100:.1f}%")
                    
                    # Excluding first step breakdown
                    if len(time_bd['diffusion_prediction_time']) > 1:
                        try:
                            # Skip first step (index 0)
                            avg_dream_model_forward_time_excl_first = sum(time_bd['dream_model_forward_time'][1:]) / (len(time_bd['dream_model_forward_time']) - 1)
                            avg_dream_overhead_time_excl_first = sum(time_bd['dream_overhead_time'][1:]) / (len(time_bd['dream_overhead_time']) - 1)
                            avg_diffusion_time_excl_first = sum(time_bd['diffusion_prediction_time'][1:]) / (len(time_bd['diffusion_prediction_time']) - 1)
                            avg_ar_model_time_excl_first = sum(time_bd['ar_model_forward_time'][1:]) / (len(time_bd['ar_model_forward_time']) - 1)
                            avg_verification_strategy_time_excl_first = sum(time_bd['verification_strategy_time'][1:]) / (len(time_bd['verification_strategy_time']) - 1)
                            avg_overhead_time_excl_first = sum(time_bd['overhead_time'][1:]) / (len(time_bd['overhead_time']) - 1)
                            avg_total_step_time_excl_first = sum(time_bd['total_step_time'][1:]) / (len(time_bd['total_step_time']) - 1)
                        except ZeroDivisionError:
                            # If any list has only one element, set all averages to 0
                            avg_dream_model_forward_time_excl_first = 0
                            avg_dream_overhead_time_excl_first = 0
                            avg_diffusion_time_excl_first = 0
                            avg_ar_model_time_excl_first = 0
                            avg_verification_strategy_time_excl_first = 0
                            avg_overhead_time_excl_first = 0
                            avg_total_step_time_excl_first = 0
                        
                        # Calculate total AR time excluding first step
                        avg_total_ar_time_excl_first = avg_ar_model_time_excl_first + avg_verification_strategy_time_excl_first
                        
                        logging.info("DETAILED TIME ANALYSIS PER STEP (EXCLUDING FIRST STEP):")
                        logging.info(f"Average Dream model forward time: {avg_dream_model_forward_time_excl_first:.2f}ms")
                        logging.info(f"Average Dream overhead time: {avg_dream_overhead_time_excl_first:.2f}ms")
                        logging.info(f"Average diffusion prediction time: {avg_diffusion_time_excl_first:.2f}ms")
                        logging.info(f"Average AR model forward time: {avg_ar_model_time_excl_first:.2f}ms")
                        logging.info(f"Average verification strategy time: {avg_verification_strategy_time_excl_first:.2f}ms")
                        logging.info(f"Average total AR time: {avg_total_ar_time_excl_first:.2f}ms")
                        logging.info(f"Average overhead time: {avg_overhead_time_excl_first:.2f}ms")
                        logging.info(f"Average total step time: {avg_total_step_time_excl_first:.2f}ms")
                        
                        print(f"\nDETAILED TIME ANALYSIS PER STEP (EXCLUDING FIRST STEP):")
                        print(f"Average Dream model forward time: {avg_dream_model_forward_time_excl_first:.2f}ms")
                        print(f"Average Dream overhead time: {avg_dream_overhead_time_excl_first:.2f}ms")
                        print(f"Average diffusion prediction time: {avg_diffusion_time_excl_first:.2f}ms")
                        print(f"Average AR model forward time: {avg_ar_model_time_excl_first:.2f}ms")
                        print(f"Average verification strategy time: {avg_verification_strategy_time_excl_first:.2f}ms")
                        print(f"Average total AR time: {avg_total_ar_time_excl_first:.2f}ms")
                        print(f"Average overhead time: {avg_overhead_time_excl_first:.2f}ms")
                        print(f"Average total step time: {avg_total_step_time_excl_first:.2f}ms")
                        
                        # Calculate percentages excluding first step
                        total_time_excl_first = avg_dream_model_forward_time_excl_first + avg_dream_overhead_time_excl_first + avg_total_ar_time_excl_first + avg_overhead_time_excl_first
                        if total_time_excl_first > 0:
                            logging.info("TIME ANALYSIS PERCENTAGES (EXCLUDING FIRST STEP):")
                            logging.info(f"Dream model forward: {avg_dream_model_forward_time_excl_first/total_time_excl_first*100:.1f}%")
                            logging.info(f"Dream overhead: {avg_dream_overhead_time_excl_first/total_time_excl_first*100:.1f}%")
                            logging.info(f"AR model forward: {avg_ar_model_time_excl_first/total_time_excl_first*100:.1f}%")
                            logging.info(f"Verification strategy: {avg_verification_strategy_time_excl_first/total_time_excl_first*100:.1f}%")
                            logging.info(f"Total AR time: {avg_total_ar_time_excl_first/total_time_excl_first*100:.1f}%")
                            logging.info(f"Overhead: {avg_overhead_time_excl_first/total_time_excl_first*100:.1f}%")
                            
                            print(f"\nTIME ANALYSIS PERCENTAGES (EXCLUDING FIRST STEP):")
                            print(f"Dream model forward: {avg_dream_model_forward_time_excl_first/total_time_excl_first*100:.1f}%")
                            print(f"Dream overhead: {avg_dream_overhead_time_excl_first/total_time_excl_first*100:.1f}%")
                            print(f"AR model forward: {avg_ar_model_time_excl_first/total_time_excl_first*100:.1f}%")
                            print(f"Verification strategy: {avg_verification_strategy_time_excl_first/total_time_excl_first*100:.1f}%")
                            print(f"Total AR time: {avg_total_ar_time_excl_first/total_time_excl_first*100:.1f}%")
                            print(f"Overhead: {avg_overhead_time_excl_first/total_time_excl_first*100:.1f}%")
        
        logging.info("="*60)
        print(f"{'='*60}")
        
        logging.info(f"Final Stats: Acc={acc*100:.2f}%, Lat={avg_lat:.3f}s, Tokens={avg_act:.2f}")
        # Prepare WandB logging data
        wandb_data = {
            "accuracy": acc,
            "avg_latency": avg_lat,
            "avg_tokens": avg_act,
            "avg_steps": avg_steps if tot_steps > 0 else 0,
            "avg_tokens_per_step": avg_tokens_per_step if tot_steps > 0 else 0,
            "avg_time_per_step": avg_time_per_step if tot_steps > 0 else 0,
            "avg_time_per_token": avg_time_per_token if tot_act > 0 else 0,
            "avg_tokens_per_second": 1000/avg_time_per_token if avg_time_per_token > 0 else 0
        }
        
        # Add detailed time analysis to WandB
        if verification_stats and 'time_breakdown' in verification_stats:
            time_bd = verification_stats['time_breakdown']
            if time_bd['diffusion_prediction_time']:
                # All steps breakdown
                avg_dream_model_forward_time = sum(time_bd['dream_model_forward_time']) / len(time_bd['dream_model_forward_time'])
                avg_dream_overhead_time = sum(time_bd['dream_overhead_time']) / len(time_bd['dream_overhead_time'])
                avg_diffusion_time = sum(time_bd['diffusion_prediction_time']) / len(time_bd['diffusion_prediction_time'])
                avg_ar_model_time = sum(time_bd['ar_model_forward_time']) / len(time_bd['ar_model_forward_time'])
                avg_verification_strategy_time = sum(time_bd['verification_strategy_time']) / len(time_bd['verification_strategy_time'])
                avg_overhead_time = sum(time_bd['overhead_time']) / len(time_bd['overhead_time'])
                avg_total_step_time = sum(time_bd['total_step_time']) / len(time_bd['total_step_time'])
                
                # Calculate total AR time
                avg_total_ar_time = avg_ar_model_time + avg_verification_strategy_time
                total_time = avg_dream_model_forward_time + avg_dream_overhead_time + avg_total_ar_time + avg_overhead_time
                
                wandb_data.update({
                    "avg_dream_model_forward_time_ms": avg_dream_model_forward_time,
                    "avg_dream_overhead_time_ms": avg_dream_overhead_time,
                    "avg_diffusion_prediction_time_ms": avg_diffusion_time,
                    "avg_ar_model_forward_time_ms": avg_ar_model_time,
                    "avg_verification_strategy_time_ms": avg_verification_strategy_time,
                    "avg_total_ar_time_ms": avg_total_ar_time,
                    "avg_overhead_time_ms": avg_overhead_time,
                    "avg_total_step_time_ms": avg_total_step_time,
                    "dream_model_forward_percentage": avg_dream_model_forward_time/total_time*100 if total_time > 0 else 0,
                    "dream_overhead_percentage": avg_dream_overhead_time/total_time*100 if total_time > 0 else 0,
                    "ar_model_forward_percentage": avg_ar_model_time/total_time*100 if total_time > 0 else 0,
                    "verification_strategy_percentage": avg_verification_strategy_time/total_time*100 if total_time > 0 else 0,
                    "total_ar_percentage": avg_total_ar_time/total_time*100 if total_time > 0 else 0,
                    "overhead_percentage": avg_overhead_time/total_time*100 if total_time > 0 else 0
                })
                
                # Excluding first step breakdown for WandB
                if len(time_bd['diffusion_prediction_time']) > 1:
                    try:
                        # Skip first step (index 0)
                        avg_dream_model_forward_time_excl_first = sum(time_bd['dream_model_forward_time'][1:]) / (len(time_bd['dream_model_forward_time']) - 1)
                        avg_dream_overhead_time_excl_first = sum(time_bd['dream_overhead_time'][1:]) / (len(time_bd['dream_overhead_time']) - 1)
                        avg_diffusion_time_excl_first = sum(time_bd['diffusion_prediction_time'][1:]) / (len(time_bd['diffusion_prediction_time']) - 1)
                        avg_ar_model_time_excl_first = sum(time_bd['ar_model_forward_time'][1:]) / (len(time_bd['ar_model_forward_time']) - 1)
                        avg_verification_strategy_time_excl_first = sum(time_bd['verification_strategy_time'][1:]) / (len(time_bd['verification_strategy_time']) - 1)
                        avg_overhead_time_excl_first = sum(time_bd['overhead_time'][1:]) / (len(time_bd['overhead_time']) - 1)
                        avg_total_step_time_excl_first = sum(time_bd['total_step_time'][1:]) / (len(time_bd['total_step_time']) - 1)
                    except ZeroDivisionError:
                        # If any list has only one element, set all averages to 0
                        avg_dream_model_forward_time_excl_first = 0
                        avg_dream_overhead_time_excl_first = 0
                        avg_diffusion_time_excl_first = 0
                        avg_ar_model_time_excl_first = 0
                        avg_verification_strategy_time_excl_first = 0
                        avg_overhead_time_excl_first = 0
                        avg_total_step_time_excl_first = 0
                    
                    # Calculate total AR time excluding first step
                    avg_total_ar_time_excl_first = avg_ar_model_time_excl_first + avg_verification_strategy_time_excl_first
                    total_time_excl_first = avg_dream_model_forward_time_excl_first + avg_dream_overhead_time_excl_first + avg_total_ar_time_excl_first + avg_overhead_time_excl_first
                    
                    wandb_data.update({
                        "avg_dream_model_forward_time_ms_excl_first": avg_dream_model_forward_time_excl_first,
                        "avg_dream_overhead_time_ms_excl_first": avg_dream_overhead_time_excl_first,
                        "avg_diffusion_prediction_time_ms_excl_first": avg_diffusion_time_excl_first,
                        "avg_ar_model_forward_time_ms_excl_first": avg_ar_model_time_excl_first,
                        "avg_verification_strategy_time_ms_excl_first": avg_verification_strategy_time_excl_first,
                        "avg_total_ar_time_ms_excl_first": avg_total_ar_time_excl_first,
                        "avg_overhead_time_ms_excl_first": avg_overhead_time_excl_first,
                        "avg_total_step_time_ms_excl_first": avg_total_step_time_excl_first,
                        "dream_model_forward_percentage_excl_first": avg_dream_model_forward_time_excl_first/total_time_excl_first*100 if total_time_excl_first > 0 else 0,
                        "dream_overhead_percentage_excl_first": avg_dream_overhead_time_excl_first/total_time_excl_first*100 if total_time_excl_first > 0 else 0,
                        "ar_model_forward_percentage_excl_first": avg_ar_model_time_excl_first/total_time_excl_first*100 if total_time_excl_first > 0 else 0,
                        "verification_strategy_percentage_excl_first": avg_verification_strategy_time_excl_first/total_time_excl_first*100 if total_time_excl_first > 0 else 0,
                        "total_ar_percentage_excl_first": avg_total_ar_time_excl_first/total_time_excl_first*100 if total_time_excl_first > 0 else 0,
                        "overhead_percentage_excl_first": avg_overhead_time_excl_first/total_time_excl_first*100 if total_time_excl_first > 0 else 0
                    })
        
        wandb.log(wandb_data)
        self.eval_logger.save()
        
        # Write summary directly to log file
        log_fname = self.cfg["save"].get("logger", "evaluation.log")
        log_path = self.run_dir / log_fname
        with open(log_path, 'a') as f:
            f.write("\n" + "="*60 + "\n")
            f.write("FINAL EVALUATION SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Accuracy: {acc*100:.2f}% ({tot_corr}/{len(self.pairs)})\n")
            f.write(f"Average latency: {avg_lat:.3f}s per sample\n")
            f.write(f"Average prompt length: {avg_in:.2f} tokens per sample\n")
            f.write(f"Average generated length: {avg_act:.2f} tokens per sample\n")
            
            if tot_steps > 0:
                avg_steps = tot_steps/len(self.pairs)
                avg_tokens_per_step = (tot_act + len(self.pairs))/tot_steps  # Add 1 EOS token per sample
                avg_time_per_step = tot_lat/tot_steps
                avg_time_per_token = tot_lat/tot_act if tot_act > 0 else 0
                
                f.write("\nTIMING METRICS:\n")
                f.write(f"Average denoising steps: {avg_steps:.2f} per sample\n")
                f.write(f"Average tokens per step: {avg_tokens_per_step:.3f}\n")
                f.write(f"Average time per step: {avg_time_per_step:.2f}ms\n")
                f.write(f"Average time per token: {avg_time_per_token:.2f}ms\n")
                f.write(f"Average tokens per second: {1000/avg_time_per_token:.1f}\n" if avg_time_per_token > 0 else "Average tokens per second: 0.0\n")
                
                # Detailed time analysis from verification stats
                if verification_stats and 'time_breakdown' in verification_stats:
                    time_bd = verification_stats['time_breakdown']
                    if time_bd['diffusion_prediction_time']:
                        # All steps breakdown
                        avg_dream_model_forward_time = sum(time_bd['dream_model_forward_time']) / len(time_bd['dream_model_forward_time'])
                        avg_dream_overhead_time = sum(time_bd['dream_overhead_time']) / len(time_bd['dream_overhead_time'])
                        avg_diffusion_time = sum(time_bd['diffusion_prediction_time']) / len(time_bd['diffusion_prediction_time'])
                        avg_ar_model_time = sum(time_bd['ar_model_forward_time']) / len(time_bd['ar_model_forward_time'])
                        avg_verification_strategy_time = sum(time_bd['verification_strategy_time']) / len(time_bd['verification_strategy_time'])
                        avg_overhead_time = sum(time_bd['overhead_time']) / len(time_bd['overhead_time'])
                        avg_total_step_time = sum(time_bd['total_step_time']) / len(time_bd['total_step_time'])
                        
                        # Calculate total AR time
                        avg_total_ar_time = avg_ar_model_time + avg_verification_strategy_time
                        
                        f.write("\nDETAILED TIME ANALYSIS PER STEP (ALL STEPS):\n")
                        f.write(f"Average Dream model forward time: {avg_dream_model_forward_time:.2f}ms\n")
                        f.write(f"Average Dream overhead time: {avg_dream_overhead_time:.2f}ms\n")
                        f.write(f"Average diffusion prediction time: {avg_diffusion_time:.2f}ms\n")
                        f.write(f"Average AR model forward time: {avg_ar_model_time:.2f}ms\n")
                        f.write(f"Average verification strategy time: {avg_verification_strategy_time:.2f}ms\n")
                        f.write(f"Average total AR time: {avg_total_ar_time:.2f}ms\n")
                        f.write(f"Average overhead time: {avg_overhead_time:.2f}ms\n")
                        f.write(f"Average total step time: {avg_total_step_time:.2f}ms\n")
                        
                        # Calculate percentages
                        total_time = avg_dream_model_forward_time + avg_dream_overhead_time + avg_total_ar_time + avg_overhead_time
                        if total_time > 0:
                            f.write("\nTIME ANALYSIS PERCENTAGES (ALL STEPS):\n")
                            f.write(f"Dream model forward: {avg_dream_model_forward_time/total_time*100:.1f}%\n")
                            f.write(f"Dream overhead: {avg_dream_overhead_time/total_time*100:.1f}%\n")
                            f.write(f"AR model forward: {avg_ar_model_time/total_time*100:.1f}%\n")
                            f.write(f"Verification strategy: {avg_verification_strategy_time/total_time*100:.1f}%\n")
                            f.write(f"Total AR time: {avg_total_ar_time/total_time*100:.1f}%\n")
                            f.write(f"Overhead: {avg_overhead_time/total_time*100:.1f}%\n")
                        
                        # Excluding first step breakdown
                        if len(time_bd['diffusion_prediction_time']) > 1:
                            # Skip first step (index 0)
                            avg_dream_model_forward_time_excl_first = sum(time_bd['dream_model_forward_time'][1:]) / (len(time_bd['dream_model_forward_time']) - 1)
                            avg_dream_overhead_time_excl_first = sum(time_bd['dream_overhead_time'][1:]) / (len(time_bd['dream_overhead_time']) - 1)
                            avg_diffusion_time_excl_first = sum(time_bd['diffusion_prediction_time'][1:]) / (len(time_bd['diffusion_prediction_time']) - 1)
                            avg_ar_model_time_excl_first = sum(time_bd['ar_model_forward_time'][1:]) / (len(time_bd['ar_model_forward_time']) - 1)
                            avg_verification_strategy_time_excl_first = sum(time_bd['verification_strategy_time'][1:]) / (len(time_bd['verification_strategy_time']) - 1)
                            avg_overhead_time_excl_first = sum(time_bd['overhead_time'][1:]) / (len(time_bd['overhead_time']) - 1)
                            avg_total_step_time_excl_first = sum(time_bd['total_step_time'][1:]) / (len(time_bd['total_step_time']) - 1)
                            
                            # Calculate total AR time excluding first step
                            avg_total_ar_time_excl_first = avg_ar_model_time_excl_first + avg_verification_strategy_time_excl_first
                            
                            f.write("\nDETAILED TIME ANALYSIS PER STEP (EXCLUDING FIRST STEP):\n")
                            f.write(f"Average Dream model forward time: {avg_dream_model_forward_time_excl_first:.2f}ms\n")
                            f.write(f"Average Dream overhead time: {avg_dream_overhead_time_excl_first:.2f}ms\n")
                            f.write(f"Average diffusion prediction time: {avg_diffusion_time_excl_first:.2f}ms\n")
                            f.write(f"Average AR model forward time: {avg_ar_model_time_excl_first:.2f}ms\n")
                            f.write(f"Average verification strategy time: {avg_verification_strategy_time_excl_first:.2f}ms\n")
                            f.write(f"Average total AR time: {avg_total_ar_time_excl_first:.2f}ms\n")
                            f.write(f"Average overhead time: {avg_overhead_time_excl_first:.2f}ms\n")
                            f.write(f"Average total step time: {avg_total_step_time_excl_first:.2f}ms\n")
                            
                            # Calculate percentages excluding first step
                            total_time_excl_first = avg_dream_model_forward_time_excl_first + avg_dream_overhead_time_excl_first + avg_total_ar_time_excl_first + avg_overhead_time_excl_first
                            if total_time_excl_first > 0:
                                f.write("\nTIME ANALYSIS PERCENTAGES (EXCLUDING FIRST STEP):\n")
                                f.write(f"Dream model forward: {avg_dream_model_forward_time_excl_first/total_time_excl_first*100:.1f}%\n")
                                f.write(f"Dream overhead: {avg_dream_overhead_time_excl_first/total_time_excl_first*100:.1f}%\n")
                                f.write(f"AR model forward: {avg_ar_model_time_excl_first/total_time_excl_first*100:.1f}%\n")
                                f.write(f"Verification strategy: {avg_verification_strategy_time_excl_first/total_time_excl_first*100:.1f}%\n")
                                f.write(f"Total AR time: {avg_total_ar_time_excl_first/total_time_excl_first*100:.1f}%\n")
                                f.write(f"Overhead: {avg_overhead_time_excl_first/total_time_excl_first*100:.1f}%\n")
        
        print(f"\nSummary is saved to: {log_path}")

import os
import re
import json
import torch
import wandb
import platform
from pathlib import Path

from tqdm import tqdm
from src.stage.base import Execute
from src.utils.utils import stop_sequences_criteria
from src.dataset.mmlu import MMLU
from src.dataset.gsm8k import GSM8K
from src.dataset.math500 import MATH500, MATH500Step
from src.dataset.mmlu_pro import MMLUPro, extract_answer
from src.dataset.piqa import PiQA
from src.dataset.hellaswag import Hellaswag
from src.dataset.obqa import OpenBookQA
from src.dataset.winogrande import WinoGrande
from src.utils.math.grader import MistralEqualityChecker, extract_final_answer
from src.utils.math.format import _strip_string
import math

import json
import os
from datetime import datetime

## COLOR
YELLOW = "\033[93m"
PURPLE = "\033[95m"
RED = "\033[91m"
BLUE = "\033[94m"
GREEN = "\033[92m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"

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
    def __init__(self, output_path: str):
        # Get node and GPU info
        node_name = get_node_name()
        gpu_info = get_gpu_info()
        
        # Check if node name is already in the path
        base_path = Path(output_path)
        if node_name not in str(base_path):
            # Modify output path to include node and GPU info
            self.output_path = str(base_path.parent / f"{node_name}_{gpu_info}" / base_path.name)
        else:
            self.output_path = output_path
        
        # Create directory if it doesn't exist
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        if os.path.exists(self.output_path):
            with open(self.output_path, 'r') as f:
                self.results = json.load(f)
        
        self.entries = []

    def log(
        self,
        prompt_index,
        batch_size,
        steps,
        gen_length,
        block_length,
        latency_ms,
        generated_token_len,
        input_token_len,
        prompt,
        response,
        answer,
        is_correct
    ):
        latency_sec = latency_ms / 1000.0
        throughput = 1.0 / latency_sec if latency_sec > 0 else 0.0

        try:
            avg_tokens_per_step = gen_length/steps
        except:
            avg_tokens_per_step = 1

        entry = {
            "Prompt Index": prompt_index,
            "Batch Size": batch_size,
            "Steps": steps,
            "Max Gen Length": gen_length,
            "Block Length": block_length,
            "Avg Latency (s)": latency_sec,
            "Throughput (req/s)": throughput,
            "Generated Token Length": generated_token_len,
            "Input Token Length": input_token_len,
            "Prompt": prompt,
            "Response": response,
            "Answer": answer,
            "Timestamp": datetime.now().isoformat(),
            "Avg Tokens per Step": avg_tokens_per_step,
            "Correct": is_correct,
        }

        self.results.append(entry)

    def save(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)


class GSM8KEval(Execute):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        self.model.generation_config.spec_sampling = self.config["eval"].get("spec_sampling", False)

        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer

        if self.draft_model is not None:
            self.draft_model.generation_config.num_assistant_tokens = self.config["spec_dec"]["assistant_tokens"]
            self.draft_model.generation_config.spec_sampling = False

        # prepare dataset
        self.datastage = GSM8K(config_dir, tokenizer)
        self.trainset, self.testset = self.datastage.run()

        # condition for end of generation
        self.max_gen_toks = self.config["eval"]["max_gen_toks"]
        self.gen_until = ['<|eot_id|>', '<|start_header_id|>user<|end_header_id|>', 'Q:', '</s>', '<|im_end|>']

    def __name__(self):
        return "GSM8KEval"
    
    def tokenize(self, prompt:str, truncation=False):
        encoding = self.tokenizer(
            prompt,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False
        )

        # TODO: add left_truncate_len (if necessary)
        return encoding["input_ids"], encoding["attention_mask"]
    
    def generate(self, input_ids, attention_mask):
        max_length = input_ids.shape[1] + self.max_gen_toks

        stop_criteria = stop_sequences_criteria(
            self.tokenizer, self.gen_until, input_ids.shape[1], input_ids.shape[0]
        )

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)        
        
        if self.config["model"]["spec_dec"]:
            start.record()
            out = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                attention_mask=attention_mask,
                do_sample=True,
                top_k=1,
                assistant_model=self.draft_model,
                # assistant_tokenizer=self.draft_tokenizer
            )
            end.record()
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)
        else:
            start.record()
            out = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                stopping_criteria=stop_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                attention_mask=attention_mask,
                do_sample=True,
                top_k=1,
                assistant_model=self.draft_model,
            )
            end.record()
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)

        return out, latency

    def metric(self, model_pred, gt):
        ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
        match = ANS_RE.search(gt)

        gt_str = match.group(1).strip()
        gt_str = gt_str.replace(",", "")

        # extract numerical answer from 
        preds = model_pred.split(self.datastage.ans_trigger.lower())
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
        
        correctness = gt_str == pred

        # if pred can be converted to float, check numerical correctness
        try:
            print(f"Pred: {pred} | GT: {gt_str}")
            pred_float = float(pred)
            gt_float = float(gt_str)
            correctness = pred_float == gt_float
        except:
            correctness = pred == gt_str

        BOLD  = "\033[1m"
        RESET = "\033[0m"

        if not correctness:
            RED   = "\033[91m"  # bright red    
            print(
                f"{RED}❌  Pred: {BOLD}{pred}{RESET}{RED} != Gold: {BOLD}{gt_str}{RESET}"
            )
        else:
            GREEN = "\033[92m"  # bright green
            print(
                f"{GREEN}✅  Pred: {BOLD}{pred}{RESET}{GREEN} == Gold: {BOLD}{gt_str}{RESET}"
            )

        return correctness

    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        output = []
        latency = []
        total_tokens = 0  # total tokens generated across all samples
        total_input_tokens = 0  # total input tokens across all samples
        
        # eval_logger = EvaluationLogger(os.path.join(self.run_dir, "gsm8k_eval_results.json"))
        eval_logger = EvaluationLogger(os.path.join(self.run_dir, "eval_results.json"))
        self.model.eval()

        json_gen_length_file = self.config.get("variable_length_file", None)

        if json_gen_length_file:
            with open(json_gen_length_file, "r") as f:
                variable_gen_lengths = json.load(f)
        else:
            variable_gen_lengths = None

        # json file format
        # [[
        # {
        #     "Prompt Index": 0,
        #     "Batch Size": 1,
        #     "Steps": null,
        #     "Max Gen Length": 1024,
        #     "Block Length": null,
        #     "Avg Latency (s)": 4.44046826171875,
        #     "Throughput (req/s)": 0.22520147449785735,
        #     "Generated Token Length": 147,
        #     "Input Token Length": 141,
        #     "Prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem. \nProblem:Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        #     "Response": "To find out how much Janet makes every day at the farmers' market, we need to calculate the number of eggs she has left after eating and baking, and then multiply that number by the price per egg.\n\nJanet's ducks lay 16 eggs per day. \nShe eats 3 eggs for breakfast, so she has 16 - 3 = 13 eggs left.\nShe bakes 4 eggs for muffins, so she has 13 - 4 = 9 eggs left.\n\nJanet sells the remaining 9 eggs at the farmers' market for $2 per egg. \nSo, she makes 9 * $2 = $18 every day at the farmers' market.\n\nThe final answer is $18.",
        #     "Answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\u2019s market.\n#### 18",
        #     "Timestamp": "2025-03-21T19:30:19.119704",
        #     "Avg Tokens per Step": 1,
        #     "Correct": true
        # },]]

        # get generated_token_length as a list
        if variable_gen_lengths:
            variable_gen_lengths = [item["Generated Token Length"] for item in variable_gen_lengths]

        

        # export the generated output
        output_file = os.path.join(self.run_dir, "output.txt")
        f = open(output_file, "w")

        pbar = tqdm(self.testset["dataset"])
        for idx, sample in enumerate(pbar):
            input_ids, attn_mask = self.tokenize(sample)
            
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)

            # Track input tokens
            curr_input_tokens = input_ids.shape[1]
            total_input_tokens += curr_input_tokens
            avg_input_tokens = total_input_tokens / (idx + 1)

            # label context
            gt = self.testset["label"][idx]

            max_gen_length = None

            # generate
            if variable_gen_lengths:
                variable_gen_length = variable_gen_lengths[idx]
                # Retrieve config values.
                steps = self.config.get("llada", {}).get("steps", None)
                block_length = self.config.get("llada", {}).get("block_length", None)
                tokens_per_step = self.config.get("llada", {}).get("tokens_per_step", None)
                
                # --- Pure Diffusion Case ---
                if block_length == -1:
                    # In pure diffusion, treat the whole generation as one block.
                    adjusted_gen_length = variable_gen_length
                    effective_block_length = variable_gen_length  # override block_length
                    # Compute steps if not provided, based on tokens_per_step.
                    if steps is None and tokens_per_step:
                        steps = math.ceil(variable_gen_length / tokens_per_step)
                    # (If steps is provided, leave it unchanged.)
                    print(f"[Pure Diffusion] Var Gen Len: {variable_gen_length} | Adjusted Gen Len: {adjusted_gen_length} | Steps: {steps}")
                
                # --- Non-Pure Diffusion Case ---
                else:
                    effective_block_length = block_length
                    if steps is None and tokens_per_step:
                        # Compute a target step count based on tokens_per_step.
                        target_steps = round(variable_gen_length / tokens_per_step)
                        # Adjust variable_gen_length to the nearest multiple of block_length.
                        candidate_down = (variable_gen_length // block_length) * block_length
                        candidate_up = math.ceil(variable_gen_length / block_length) * block_length
                        if abs(variable_gen_length - candidate_down) <= abs(candidate_up - variable_gen_length):
                            adjusted_gen_length = candidate_down
                        else:
                            adjusted_gen_length = candidate_up

                        # Determine the number of blocks.
                        num_blocks = adjusted_gen_length // block_length
                        # Adjust steps so that steps is a multiple of num_blocks and as close as possible to target_steps.
                        steps = round(target_steps / num_blocks) * num_blocks
                        if steps < num_blocks:
                            steps = num_blocks  # Ensure at least one step per block
                        print(f"[Semi-Autoregressive] Var Gen Len: {variable_gen_length} | Adjusted Gen Len: {adjusted_gen_length} | Steps: {steps}")
                    elif steps is not None:
                        adjusted_gen_length = variable_gen_length
                        # Increment adjusted_gen_length until the number of blocks divides steps evenly.
                        while True:
                            if adjusted_gen_length % block_length == 0:
                                num_blocks = adjusted_gen_length // block_length
                                if steps % num_blocks == 0:
                                    break
                            adjusted_gen_length += 1
                    else:
                        adjusted_gen_length = variable_gen_length

                max_gen_length = adjusted_gen_length
                print(f"Final: Var Gen Len: {variable_gen_length} | Adjusted Gen Len: {adjusted_gen_length}")
                tok, lat = self.generate(input_ids, attn_mask, variable_gen_length=adjusted_gen_length)
            else:
                tok, lat = self.generate(input_ids, attn_mask)

            tok_list = tok.tolist()

            # decoded tokens
            tok = tok_list[0][input_ids.shape[1] :]

            # Track generated tokens
            curr_tokens = len(tok)
            total_tokens += curr_tokens
            avg_tokens = total_tokens / (idx + 1)

            # decode tokens
            dec_tok = self.tokenizer.decode(tok, skip_special_tokens=True)

            #### New Code:
            # Count input token length
            input_token_len = input_ids.shape[1]

            # Count generated token length from tokenizer
            generated_token_len = len(self.tokenizer.encode(dec_tok, add_special_tokens=False))
            print(f"Generated Token Length: {generated_token_len} | Max Gen Length: {max_gen_length} | Input Token Length: {input_token_len}")

            # print(self.config["eval"])
            # print(self.config)

            # Evaluate correctness
            correctness = self.metric(dec_tok, gt)
            output.append(correctness)
            latency.append(lat)

            try:
                steps = self.config["llada"].get("steps", None)
                block_length = self.config["llada"].get("block_length", None)
                tokens_per_step = self.config["llada"].get("tokens_per_step", None)

                if not steps and tokens_per_step:
                    steps = math.ceil(max_gen_length / tokens_per_step)
            except:
                block_length = None
                steps = None


            eval_logger.log(
                prompt_index=idx,
                batch_size=input_ids.shape[0],
                steps=steps,
                # gen_length=len(tok),
                gen_length=len(tok) if max_gen_length is None else max_gen_length,
                block_length=block_length,
                latency_ms=lat,
                input_token_len=input_token_len,
                generated_token_len=generated_token_len,
                prompt=sample,
                response=dec_tok,
                answer=gt,
                is_correct=correctness,
            )


            acc = sum(output) / len(output)
            # Calculate average latency
            avg_lat = sum(latency) / len(latency)
            # Update progress bar with both current and average latency
            pbar.set_description(f"Accuracy: {acc:.4f} | Latency: {lat/1000:.4f}s (avg: {avg_lat/1000:.4f}s) | Avg gen: {avg_tokens:.4f} | Avg in: {avg_input_tokens:.4f}")

            # save the generated text
            out_str = f"Test ID = [{idx}] | [{correctness}] \n{dec_tok}"
            print(out_str, file=f)

            try:
                avg_tokens_per_step = len(tok)/steps
            except:
                avg_tokens_per_step = 1

            if self.wandb_flag:
                wandb.log({"Accuracy": acc, 
                           "latency": lat, 
                           "Correct": float(correctness), 
                           "Token Length": len(tok),
                           "Input Token Length": input_token_len,
                           "Generated Token Length": generated_token_len,
                           "Avg Tokens Per Step": avg_tokens_per_step,
                           "Avg Latency": avg_lat
                           })

            # save every 100 prompts
            if idx % 100 == 0:
                eval_logger.save()
        
        eval_logger.save()

        f.close()
        avg = sum(output) / len(self.testset["dataset"])
        avg_lat = sum(latency) / len(latency)

        self.logger.info(f"Average Score (exact match) = {avg:.4f} | average latency = {avg_lat:.4f}")
        self.logger.info(f"Average input tokens = {total_input_tokens/len(output):.4f} | Average generated tokens = {total_tokens/len(output):.4f}")
        return output

class MMLUEval(Execute):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer

        # condition for end of generation
        self.max_gen_toks = self.config["eval"]["max_gen_toks"]

        # dataset stage
        self.datastage = MMLU(config_dir, self.tokenizer)

        self.sub_task_list = [
            'abstract_algebra',
            'anatomy',
            'astronomy',
            'business_ethics',
            'clinical_knowledge',
            'college_biology',
            'college_chemistry',
            'college_computer_science',
            'college_mathematics',
            'college_medicine',
            'college_physics',
            'computer_security',
            'conceptual_physics',
            'econometrics',
            'electrical_engineering',
            'elementary_mathematics',
            'formal_logic',
            'global_facts',
            'high_school_biology',
            'high_school_chemistry',
            'high_school_computer_science',
            'high_school_european_history',
            'high_school_geography',
            'high_school_government_and_politics',
            'high_school_macroeconomics',
            'high_school_mathematics',
            'high_school_microeconomics',
            'high_school_physics',
            'high_school_psychology',
            'high_school_statistics',
            'high_school_us_history',
            'high_school_world_history',
            'human_aging',
            'human_sexuality',
            'international_law',
            'jurisprudence',
            'logical_fallacies',
            'machine_learning',
            'management',
            'marketing',
            'medical_genetics',
            'miscellaneous',
            'moral_disputes',
            'moral_scenarios',
            'nutrition',
            'philosophy',
            'prehistory',
            'professional_accounting',
            'professional_law',
            'professional_medicine',
            'professional_psychology',
            'public_relations',
            'security_studies', 
            'sociology',
            'us_foreign_policy',
            'virology',
            'world_religions'
        ]

    def compute_mmlu_score(self, output_filename):
        with open(output_filename, 'r') as f:
            run_results = json.load(f)
        total_acc = 0
        total_num = 0
        for task in run_results:
            acc = 0
            pred_answers = run_results[task]['pred_answers']
            gold_answers = run_results[task]['gold_answers']
            for pred, gold in zip(pred_answers, gold_answers):
                if pred == gold: acc += 1
            print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
            total_acc += acc
            total_num += len(gold_answers)
        self.logger.info("ACC-all: %.4f" % (total_acc/total_num))

    def tokenize(self, prompt):
        encoding = self.tokenizer.batch_encode_plus([prompt], return_tensors="pt", padding=True)
        return encoding["input_ids"], encoding["attention_mask"]

    def generate(self, input_ids, attention_mask):
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1, 
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
            temperature=1.0,
            top_p=1.0
        )

        return out

    def metric(self, model_pred, gt):
        answers = model_pred[-1]
        return answers == gt

    def run(self):
        output = []
        self.model.eval()
        run_results = {}

        for task in self.sub_task_list:
            self.logger.info(f"\nEvaluating Task: {task}")
            testset = self.datastage.run(task)

            pred, golden_output = [], []

            pbar = tqdm(testset["dataset"])
            for idx, sample in enumerate(pbar):
                input_ids, attn_mask = self.tokenize(sample)

                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)

                # label context
                gt = testset["label"][idx]

                # generate
                tok = self.generate(input_ids, attn_mask)
                dec_tok = self.tokenizer.batch_decode(tok, skip_special_tokens=True)

                pred.append(dec_tok[0][-1])
                golden_output.append(gt)

                correctness = self.metric(dec_tok[0], gt)
                output.append(int(correctness))

            run_results[task] = {'pred_answers':pred, 'gold_answers':golden_output}

        output_filename = os.path.join(self.run_dir, "accuracy.json")
        
        with open(output_filename, 'w') as f:
            json.dump(run_results, f, ensure_ascii=False, indent=2)

        self.compute_mmlu_score(output_filename)

        return output
    
class MATH500Eval(Execute):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        # MATH 500 returns the test set directly (partition = "test")
        self.datastage = MATH500(config_dir)
        self.testset = self.datastage.run()

        # condition for end of generation
        self.max_tokens = self.config["eval"]["max_tokens"]

        # initialize the euality checker
        self.equality_checker = MistralEqualityChecker()

    def generate(self, prepared_text, idx:int):
        templated_text = self.tokenizer.apply_chat_template(
                prepared_text, 
                add_generation_prompt=True,
                tokenize=False,
        )

        encoding = self.tokenizer(templated_text, return_tensors="pt").to(self.device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # generate output
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                do_sample=True
            )

        return out
    
    def grade_answer(self, pred_answer, gold_answer):
        gold_answer = _strip_string(gold_answer)
        print("\n=====")
        print(pred_answer)
        print("=====\n")
        extracted_answer = extract_final_answer(pred_answer)

        # Check equivalence using hybrid approach
        try:
            score, equality_checker_response = self.equality_checker.check(gold_answer, extracted_answer)
            score = float(score)
        except:
            score = 0

        print(f"Pred Answer: {extracted_answer} | Gold Answer: {gold_answer} | correctness = {score}")

        return score

    def run(self):
        answer = self.testset["target"]
        pbar = tqdm(self.testset["dataset"])

        results = {"problem": [], "prediction": [], "correctness": []}
        total_correct = 0

        for idx, sample in enumerate(pbar):
            out = self.generate(sample, idx)
            dec_tok = self.tokenizer.batch_decode(out, skip_special_tokens=True)

            pred_answer = dec_tok[0].strip()
            gold_answer = answer[idx].strip()

            is_correct = self.grade_answer(pred_answer, gold_answer)

            results["problem"].append(sample[0])
            results["prediction"].append(dec_tok)
            results["correctness"].append(int(is_correct))

            total_correct += is_correct

            instant_acc = (total_correct / (idx + 1)) * 100
            pbar.set_description(f"Accuracy = {instant_acc:.3f}")

        json_path = os.path.join(self.run_dir, "predictions.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)

        acc = total_correct / len(self.testset["dataset"])
        self.logger.info(f"\nInference Completed! Accuracy = {acc:.3f}")


class MATH500StepEval(MATH500Eval):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir, model, tokenizer, draft_model, draft_tokenizer)

        # MATH 500 returns the test set directly (partition = "test")
        self.datastage = MATH500Step(config_dir)
        self.testset = self.datastage.run()

    def generate(self, prepared_text, idx:int):
        templated_text = self.tokenizer.apply_chat_template(
                prepared_text, 
                add_generation_prompt=True,
                tokenize=False,
        )

        encoding = self.tokenizer(templated_text, return_tensors="pt").to(self.device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # generate output
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                do_sample=True
            )

        return out
    
    def run(self):
        return super().run()


class MMLUProEval(Execute):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer

        # mmlu-pro dataset
        datastage = MMLUPro(config_dir)
        self.testset = datastage.run()

        self.max_gen_toks = self.config["eval"]["max_gen_toks"]
        self.gen_until = ["Question:"]

    def __name__(self):
        return "MMLUProEval"
    
    def tokenize(self, prompt:str):
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt"
        )
        return encoding["input_ids"], encoding["attention_mask"]
    
    def generate(self, input_ids, attention_mask):
        max_length = input_ids.shape[1] + self.max_gen_toks

        stop_criteria = stop_sequences_criteria(
            self.tokenizer, self.gen_until, input_ids.shape[1], input_ids.shape[0]
        )

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            stopping_criteria=stop_criteria,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            attention_mask=attention_mask,
            do_sample=True,
            top_k=1
        )
        end.record()
        torch.cuda.synchronize()
        latency = start.elapsed_time(end)

        return out, latency
        
    def metric(self, model_pred, gt):
        ans = extract_answer(model_pred)
        return ans == gt
    
    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        output = []
        latency = []
        self.model.eval()

        for subject, data in self.testset.items():
            subject_output = []
            output_file = os.path.join(self.run_dir, f"output_{subject}.txt")
            f = open(output_file, "w")
    
            pbar = tqdm(data["sample"])
            for idx, sample in enumerate(pbar):
                input_ids, attn_mask = self.tokenize(sample)
                gt = data["label"][idx]

                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)

                tok, lat = self.generate(input_ids, attn_mask)
                tok_list = tok.tolist()

                # decoded tokens
                tok = tok_list[0][input_ids.shape[1] :]
                generated_text = self.tokenizer.decode(tok, skip_special_tokens=True)

                correctness = self.metric(generated_text, gt)
                subject_output.append(correctness)
                
                latency.append(lat)
                acc = sum(subject_output) / len(subject_output)
                pbar.set_description(f"Accuracy: {acc:.4f} | latency = {lat:.4f}")

                # save the generated text
                out_str = f"Test ID = [{idx}] | [{correctness}] \n{generated_text}"
                print(out_str, file=f)

            f.close()

class PiQAEval(Execute):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        self.model.generation_config.spec_sampling = self.config["eval"].get("spec_sampling", False)

        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer

        if self.draft_model is not None:
            self.draft_model.generation_config.num_assistant_tokens = self.config["spec_dec"]["assistant_tokens"]
            self.draft_model.generation_config.spec_sampling = False

        # datastage
        self.datastage = PiQA(tokenizer, config_dir)
        self.trainset, self.testset = self.datastage.run()

        self.max_gen_toks = self.config["eval"]["max_gen_toks"]

    def __name__(self):
        return "PiQAEval"
    
    def tokenize(self, prompt):
        encoding = self.tokenizer(
            prompt,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False
        )

        # TODO: add left_truncate_len (if necessary)
        return encoding["input_ids"], encoding["attention_mask"]
    
    def generate(self, input_ids, attention_mask):
        max_length = input_ids.shape[1] + self.max_gen_toks

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        if self.config["model"]["spec_dec"]:
            start.record()
            out = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.6,
                top_p=1.0,
                assistant_model=self.draft_model,
            )
            end.record()
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)
        else:
            start.record()
            out = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                attention_mask=attention_mask,
                do_sample=True,
                assistant_model=self.draft_model,
            )
            end.record()
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)

        return out, latency

    def metric(self, sentence:str, gt:str):
        gt_index = gt[-1]

        sentence = sentence.lower()
        sentence = sentence.split(self.datastage.ans_trigger.lower())

        valid_ans = True if len(sentence) > 1 else False

        pred = sentence[-1]
        pred_answers = re.findall(r'solution1|solution2|solution 1|solution 2', pred)

        if len(pred_answers) > 0:
            pred_answers = pred_answers[0][-1]
            correctness = pred_answers == gt_index
        else:
            pred_answers = ""
            correctness = False
        
        return correctness, pred_answers, gt_index
    
    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        output = []
        latency = []
        self.model.eval()

        pbar = tqdm(self.testset["dataset"])
        for idx, sample in enumerate(pbar):
            
            if not self.config["dataset"]["apply_chat_template"]:
                sample = sample[0]["content"]

            input_ids, attn_mask = self.tokenize(sample)
            
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)

            # label context
            gt = self.testset["label"][idx]

            # generate
            tok, lat = self.generate(input_ids, attn_mask)
            tok_list = tok.tolist()

            # decoded tokens
            tok = tok_list[0][input_ids.shape[1] :]

            # decode tokens
            dec_tok = self.tokenizer.decode(tok, skip_special_tokens=True)

            # correctness
            correctness, pred, gt = self.metric(dec_tok, gt)

            output.append(int(correctness))
            latency.append(lat)

            acc = sum(output) / len(output)
            pbar.set_description(f"Accuracy: {acc:.4f} | latency = {lat:.4f}")

            out_str = f"Test ID = [{idx}] | Correctness = {correctness} |  GT=[{gt}] | Extracted ANS = {pred} \n\n Predicted = {dec_tok}"
            self.logger.info(out_str)
        
        avg = sum(output) / len(self.testset["dataset"])
        avg_lat = sum(latency) / len(latency)

        self.logger.info(f"Average Score (exact match) = {avg:.4f} | average latency = {avg_lat:.4f}")

class OpenbookQAEval(Execute):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        self.model.generation_config.spec_sampling = self.config["eval"].get("spec_sampling", False)

        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer

        if self.draft_model is not None:
            self.draft_model.generation_config.num_assistant_tokens = self.config["spec_dec"]["assistant_tokens"]
            self.draft_model.generation_config.spec_sampling = False

        self.datastage = OpenBookQA(config_dir, tokenizer)
        self.testset = self.datastage.run()

        self.max_gen_toks = self.config["eval"]["max_gen_toks"]

    def __name__(self):
        return "OpenbookQAEval"
    
    def tokenize(self, prompt):
        encoding = self.tokenizer(
            prompt,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False
        )

        return encoding["input_ids"], encoding["attention_mask"]
    
    def generate(self, input_ids, attention_mask):
        max_length = input_ids.shape[1] + self.max_gen_toks

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        if self.config["model"]["spec_dec"]:
            start.record()
            out = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.6,
                top_p=1.0,
                assistant_model=self.draft_model,
            )
            end.record()
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)
        else:
            start.record()
            out = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                attention_mask=attention_mask,
                do_sample=True,
                assistant_model=self.draft_model,
            )
            end.record()
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)

        return out, latency
    
    def metric(self, sentence:str, gt:str):
        gt_index = gt[-1]

        sentence = sentence.lower()
        sentence = sentence.split(self.datastage.ans_trigger.lower())

        valid_ans = True if len(sentence) > 1 else False

        if valid_ans:
            pred = sentence[-1]
        else:
            pred = sentence[-1]

        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer 1|answer 2|answer 3|answer 4', pred)

        if len(pred_answers) > 0:
            pred_answers = pred_answers[0][-1]
            correctness = pred_answers == gt_index
        else:
            pred_answers = ""
            correctness = False
        
        return correctness, pred_answers, gt_index

    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        output = []
        latency = []
        self.model.eval()

        pbar = tqdm(self.testset["dataset"])
        for idx, sample in enumerate(pbar):
            
            if not self.config["dataset"]["apply_chat_template"]:
                sample = sample[0]["content"]

            input_ids, attn_mask = self.tokenize(sample)
            
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)

            # label context
            gt = self.testset["label"][idx]

            # generate
            tok, lat = self.generate(input_ids, attn_mask)
            tok_list = tok.tolist()

            # decoded tokens
            tok = tok_list[0][input_ids.shape[1] :]

            # decode tokens
            dec_tok = self.tokenizer.decode(tok, skip_special_tokens=True)

            # correctness
            correctness, pred, gt = self.metric(dec_tok, gt)

            output.append(int(correctness))
            latency.append(lat)

            acc = sum(output) / len(output)
            pbar.set_description(f"Accuracy: {acc:.4f} | latency = {lat:.4f}")

            out_str = f"Test ID = [{idx}] | Correctness = {correctness} |  GT=[{gt}] | Extracted ANS = {pred} \n\n Predicted = {dec_tok}"
            self.logger.info(out_str)

        avg = sum(output) / len(self.testset["dataset"])
        avg_lat = sum(latency) / len(latency)

        self.logger.info(f"Average Score (exact match) = {avg:.4f} | average latency = {avg_lat:.4f}")

class HellaswagEval(Execute):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        self.model.generation_config.spec_sampling = self.config["eval"].get("spec_sampling", False)

        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer

        if self.draft_model is not None:
            self.draft_model.generation_config.num_assistant_tokens = self.config["spec_dec"]["assistant_tokens"]
            self.draft_model.generation_config.spec_sampling = False

        # datastage
        self.datastage = Hellaswag(tokenizer, config_dir)
        self.testset = self.datastage.run()

        self.max_gen_toks = self.config["eval"]["max_gen_toks"]
     
    def __name__(self):
        return "HellaswagEval"
    
    def tokenize(self, prompt):
        encoding = self.tokenizer(
            prompt,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False
        )

        return encoding["input_ids"], encoding["attention_mask"]
    
    def generate(self, input_ids, attention_mask):
        max_length = input_ids.shape[1] + self.max_gen_toks

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        if self.config["model"]["spec_dec"]:
            start.record()
            out = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.6,
                top_p=1.0,
                assistant_model=self.draft_model,
            )
            end.record()
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)
        else:
            start.record()
            out = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                attention_mask=attention_mask,
                do_sample=True,
                assistant_model=self.draft_model,
            )
            end.record()
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)

        return out, latency
    
    def metric(self, sentence:str, gt:str):

        YELLOW = "\033[93m"
        PURPLE = "\033[95m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        CYAN = "\033[96m"
        WHITE = "\033[97m"
        RESET = "\033[0m"

        # print(f"{PURPLE}Sentence (original): {sentence}{RESET}", flush=True)

        gt_index = gt[-1]

        sentence = sentence.lower()

        # print(f"{PURPLE}Sentence (lowered): {sentence}{RESET}", flush=True)

        # print out answer trigger
        # print(f"{WHITE}Answer Trigger: {self.datastage.ans_trigger.lower()}{RESET}", flush=True)
        sentence = sentence.split(self.datastage.ans_trigger.lower())
        # print(f"{BLUE}Sentence (splited): {sentence}{RESET}", flush=True)

        valid_ans = True if len(sentence) > 1 else False

        if valid_ans:
            pred = sentence[-1]
        else:
            pred = sentence[-1]

        # print(f"{RED}Pred: {pred}{RESET}", flush=True)

        pred_answers = re.findall(r'ending1|ending2|ending3|ending4|ending 1|ending 2|ending 3|ending 4', pred)

        # print(f"{CYAN}Pred Answers: {pred_answers}{RESET}", flush=True)

        if len(pred_answers) > 0:
            pred_answers = pred_answers[0][-1]
            correctness = pred_answers == gt_index
        else:
            pred_answers = ""
            correctness = False
        
        # print(f"{GREEN}Correctness: {correctness}{RESET}", flush=True)
        # print(f"{GREEN}Pred Answers: {pred_answers}{RESET}", flush=True)
        # print(f"{GREEN}GT Index: {gt_index}{RESET}", flush=True)

        # import pdb; pdb.set_trace()
        return correctness, pred_answers, gt_index
    
    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        output = []
        latency = []
        self.model.eval()

        pbar = tqdm(self.testset["dataset"])
        for idx, sample in enumerate(pbar):
            
            if not self.config["dataset"]["apply_chat_template"]:
                sample = sample[0]["content"]

            input_ids, attn_mask = self.tokenize(sample)
            
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)

            # label context
            gt = self.testset["label"][idx]

            # generate
            tok, lat = self.generate(input_ids, attn_mask)
            tok_list = tok.tolist()

            # decoded tokens
            tok = tok_list[0][input_ids.shape[1] :]

            # decode tokens
            dec_tok = self.tokenizer.decode(tok, skip_special_tokens=True)

            # correctness
            correctness, pred, gt = self.metric(dec_tok, gt)

            output.append(int(correctness))
            latency.append(lat)

            acc = sum(output) / len(output)
            pbar.set_description(f"Accuracy: {acc:.4f} | latency = {lat:.4f}")

            out_str = f"Test ID = [{idx}] | Correctness = {correctness} |  GT=[{gt}] | Extracted ANS = {pred} \n\n Predicted = {dec_tok}"
            self.logger.info(out_str)

        avg = sum(output) / len(self.testset["dataset"])
        avg_lat = sum(latency) / len(latency)

        self.logger.info(f"Average Score (exact match) = {avg:.4f} | average latency = {avg_lat:.4f}")

class WinoGrandeEval(Execute):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        self.model.generation_config.spec_sampling = self.config["eval"].get("spec_sampling", False)

        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer

        if self.draft_model is not None:
            self.draft_model.generation_config.num_assistant_tokens = self.config["spec_dec"]["assistant_tokens"]
            self.draft_model.generation_config.spec_sampling = False

        # datastage
        self.datastage = WinoGrande(tokenizer, config_dir)
        self.testset = self.datastage.run()

        self.max_gen_toks = self.config["eval"]["max_gen_toks"]

    def __name__(self):
        return "WinoGrandeEval"
    
    def tokenize(self, prompt):
        encoding = self.tokenizer(
            prompt,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False
        )

        # TODO: add left_truncate_len (if necessary)
        return encoding["input_ids"], encoding["attention_mask"]
    
    def generate(self, input_ids, attention_mask):
        max_length = input_ids.shape[1] + self.max_gen_toks

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        if self.config["model"]["spec_dec"]:
            start.record()
            out = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.6,
                top_p=1.0,
                assistant_model=self.draft_model,
            )
            end.record()
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)
        else:
            start.record()
            out = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                attention_mask=attention_mask,
                do_sample=True,
                assistant_model=self.draft_model,
            )
            end.record()
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)

        return out, latency
    
    def metric(self, sentence:str, gt:str):
        gt_index = gt[-1]

        sentence = sentence.lower()
        sentence = sentence.split(self.datastage.ans_trigger.lower())

        valid_ans = True if len(sentence) > 1 else False

        if valid_ans:
            pred = sentence[-1]
        else:
            pred = sentence[-1]
    
        pred_answers = re.findall(r'option1|option2|option 1|option 2', pred)

        if len(pred_answers) > 0:
            pred_answers = pred_answers[0][-1]
            correctness = pred_answers == gt_index
        else:
            pred_answers = ""
            correctness = False
        
        return correctness, pred_answers, gt_index

    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        output = []
        latency = []
        self.model.eval()

        pbar = tqdm(self.testset["dataset"])
        for idx, sample in enumerate(pbar):
            
            if not self.config["dataset"]["apply_chat_template"]:
                sample = sample[0]["content"]

            input_ids, attn_mask = self.tokenize(sample)
            
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)

            # label context
            gt = self.testset["label"][idx]

            # generate
            tok, lat = self.generate(input_ids, attn_mask)
            tok_list = tok.tolist()

            # decoded tokens
            tok = tok_list[0][input_ids.shape[1] :]

            # decode tokens
            dec_tok = self.tokenizer.decode(tok, skip_special_tokens=True)

            # correctness
            correctness, pred, gt = self.metric(dec_tok, gt)

            output.append(int(correctness))
            latency.append(lat)

            acc = sum(output) / len(output)
            pbar.set_description(f"Accuracy: {acc:.4f} | latency = {lat:.4f}")

            out_str = f"Test ID = [{idx}] | Correctness = {correctness} |  GT=[{gt}] | Extracted ANS = {pred} \n\n Predicted = {dec_tok}"
            self.logger.info(out_str)

        avg = sum(output) / len(self.testset["dataset"])
        avg_lat = sum(latency) / len(latency)

        self.logger.info(f"Average Score (exact match) = {avg:.4f} | average latency = {avg_lat:.2f}")



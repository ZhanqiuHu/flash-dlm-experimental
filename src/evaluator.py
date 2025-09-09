import os
import re
import json
import torch
import wandb

from tqdm import tqdm
from src.stage.base import Execute
from src.utils.utils import stop_sequences_criteria
from src.dataset.mmlu import MMLU
from src.dataset.gsm8k import GSM8K
from src.dataset.math500 import MATH500, MATH500Step
from src.utils.math.grader import MistralEqualityChecker, extract_final_answer
from src.utils.math.format import _strip_string
import math


import json
import os
from datetime import datetime

class EvaluationLogger:
    def __init__(self, output_path):
        self.output_path = output_path
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

        self.entries.append(entry)

    def save(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.entries, f, indent=4)


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
        
        return gt_str == pred

    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        output = []
        latency = []
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

            # label context
            gt = self.testset["label"][idx]

            max_gen_length = None

            # generate
            if variable_gen_lengths:

                variable_gen_length = variable_gen_lengths[idx]

                steps = self.config.get("llada", {}).get("steps", None)
                block_length = self.config.get("llada", {}).get("block_length", None)
                tokens_per_step = self.config.get("llada", {}).get("tokens_per_step", None)

                if not steps and tokens_per_step:
                    # Compute a target step count based on tokens_per_step (using round for closeness)
                    target_steps = variable_gen_length / tokens_per_step
                    target_steps = round(target_steps)
                    
                    # Adjust variable_gen_length to the nearest multiple of block_length.
                    candidate_down = (variable_gen_length // block_length) * block_length
                    candidate_up = math.ceil(variable_gen_length / block_length) * block_length
                    if abs(variable_gen_length - candidate_down) <= abs(candidate_up - variable_gen_length):
                        adjusted_gen_length = candidate_down
                    else:
                        adjusted_gen_length = candidate_up

                    # Determine the number of blocks from the adjusted generation length
                    num_blocks = adjusted_gen_length // block_length

                    # Now, adjust steps so that it’s a multiple of num_blocks and as close as possible to the target
                    steps = round(target_steps / num_blocks) * num_blocks
                    if steps < num_blocks:
                        steps = num_blocks  # Ensure at least one step per block

                    print(f"Variable Gen Length: {variable_gen_length} | Adjusted Gen Length: {adjusted_gen_length} | Steps: {steps}")
                else:
                    if steps is not None and block_length is not None:
                        adjusted_gen_length = variable_gen_length
                        while True:
                            if adjusted_gen_length % block_length == 0:
                                num_blocks = adjusted_gen_length // block_length
                                if steps % num_blocks == 0:
                                    break
                            adjusted_gen_length += 1
                    else:
                        adjusted_gen_length = variable_gen_length

                max_gen_length = adjusted_gen_length

                print(f"Variable Gen Length: {variable_gen_length} | Adjusted Gen Length: {adjusted_gen_length}")

                tok, lat = self.generate(input_ids, attn_mask, variable_gen_length=adjusted_gen_length)
                
            else:
                tok, lat = self.generate(input_ids, attn_mask)
            tok_list = tok.tolist()

            # decoded tokens
            tok = tok_list[0][input_ids.shape[1] :]

            # decode tokens
            dec_tok = self.tokenizer.decode(tok, skip_special_tokens=True)

            #### New Code:
            # Count input token length
            input_token_len = input_ids.shape[1]

            # Count generated token length from tokenizer
            generated_token_len = len(self.tokenizer.encode(dec_tok, add_special_tokens=False))

            # print(self.config["eval"])
            # print(self.config)
            
            print(dec_tok, flush=True)
            print(gt, flush=True)

            correctness = self.metric(dec_tok, gt)
            output.append(int(correctness))
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
            pbar.set_description(f"Accuracy: {acc:.2f} | latency = {lat:.2f}")

            # average latency
            avg_lat = sum(latency) / len(latency)

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

        self.logger.info(f"Average Score (exact match) = {avg:.2f} | average latency = {avg_lat:.2f}")
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
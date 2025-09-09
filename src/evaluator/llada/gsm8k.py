"""
GSM8K evaluator for Llada model
"""

import re
import torch
from src.evaluator.llada.base import BaseLladaEvaluator
from src.dataset.gsm8k import GSM8K
from src.evaluator.llada.logger import EvaluationLogger

class GSM8KLlada(BaseLladaEvaluator):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None):
        super().__init__(config_dir, model, tokenizer, draft_model, draft_tokenizer)
        
        # Load GSM8K dataset
        self.datastage = GSM8K(config_dir, tokenizer)
        self.trainset, self.testset = self.datastage.run()
        
        # Initialize logger
        self.logger = EvaluationLogger(self.config["eval"]["output_path"])

    def __name__(self):
        return "GSM8KLlada"

    def metric(self, model_pred, gt):
        """Evaluate the model's prediction against ground truth"""
        # Extract answer from ground truth using regex
        ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
        match = ANS_RE.search(gt)
        
        if not match:
            return False
            
        gt_str = match.group(1).strip()
        gt_str = gt_str.replace(",", "")

        # Extract numerical answer from model prediction
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
        
        # Try to compare as floats for numerical accuracy
        try:
            pred_float = float(pred)
            gt_float = float(gt_str)
            return pred_float == gt_float
        except:
            # Fall back to string comparison
            return gt_str == pred

    @torch.no_grad()
    def run(self):
        """Run the evaluation"""
        self.logger.info(f"Starting GSM8K evaluation with {len(self.testset)} examples")
        
        correct = 0
        total = len(self.testset)
        
        for idx, example in enumerate(self.testset):
            prompt = example["prompt"]
            answer = example["answer"]
            
            # Tokenize input
            input_ids, attention_mask = self.tokenize(prompt)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Generate response
            output_ids, latency = self.generate(input_ids, attention_mask)
            
            # Decode response
            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Calculate metrics
            is_correct = self.metric(response, answer)
            correct += int(is_correct)
            
            # Log results
            self.logger.log(
                prompt_index=idx,
                batch_size=1,
                steps=self.config["eval"].get("steps", 1),
                gen_length=self.max_gen_toks,
                block_length=self.config["eval"].get("block_length", -1),
                latency_ms=latency,
                generated_token_len=len(output_ids[0]) - len(input_ids[0]),
                input_token_len=len(input_ids[0]),
                prompt=prompt,
                response=response,
                answer=answer,
                is_correct=is_correct
            )
            
            # Print progress
            if (idx + 1) % 10 == 0:
                accuracy = (correct / (idx + 1)) * 100
                self.logger.info(f"Processed {idx + 1}/{total} examples. Current accuracy: {accuracy:.2f}%")
        
        # Calculate final accuracy
        final_accuracy = (correct / total) * 100
        self.logger.info(f"Evaluation complete. Final accuracy: {final_accuracy:.2f}%")
        
        return {"accuracy": final_accuracy} 
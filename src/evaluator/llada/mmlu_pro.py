"""
MMLU Pro evaluator for Llada model
"""

import re
import torch
from src.evaluator.llada.base import BaseLladaEvaluator
from src.dataset.mmlu_pro import MMLUPro, extract_answer
from src.evaluator.llada.logger import EvaluationLogger

class MMLUProLlada(BaseLladaEvaluator):
    def __init__(self, config_dir, model, tokenizer, draft_model=None, draft_tokenizer=None, subject_filter=None):
        super().__init__(config_dir, model, tokenizer, draft_model, draft_tokenizer)
        
        # Load MMLU Pro dataset
        self.datastage = MMLUPro(config_dir, tokenizer)
        self.trainset, self.testset = self.datastage.run()
        
        # Filter subjects if specified
        if subject_filter:
            self.testset = [item for item in self.testset if item["subject"] in subject_filter]
        
        # Initialize logger
        self.logger = EvaluationLogger(self.config["eval"]["output_path"])

    def __name__(self):
        return "MMLUProLlada"

    def metric(self, model_pred, gt):
        """Evaluate the model's prediction against ground truth"""
        # Extract answer from ground truth
        gt_answer = extract_answer(gt)
        
        # Extract answer from model prediction
        pred_answer = extract_answer(model_pred)
        
        # Compare answers
        return gt_answer == pred_answer

    @torch.no_grad()
    def run(self):
        """Run the evaluation"""
        self.logger.info(f"Starting MMLU Pro evaluation with {len(self.testset)} examples")
        
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
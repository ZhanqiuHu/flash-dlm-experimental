"""
PiQA dataset
"""

import json
from tqdm import tqdm
from typing import Dict, List
from src.stage.data import DataStage
from datasets import load_dataset

class PiQA(DataStage):
    def __init__(self, tokenizer, config_dir):
        super().__init__(config_dir)
        self.trainset_path = self.config["dataset"]["train"]
        self.validset_path = self.config["dataset"]["test"]
        self.apply_chat_template = self.config["dataset"].get("apply_chat_template", True)
        self.enable_reasoning = self.config["dataset"].get("enable_reasoning", False)

        self.tokenizer = tokenizer

        # prompt template
        self.prompt_head = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n"
        
        if self.enable_reasoning:
            self.prompt_tail = "Please briefly explain your reasoning for your choice. Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
        else:
            self.prompt_tail = "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
            
        self.ans_trigger = "The final answer is"
        # self.prompt_tail = "\n### Response:"

        # self.prompt_head = "Question: "
        # self.prompt_tail = "\nAnswer:"

    def load_json(self, split):
        if split == "train":
            file_path = self.trainset_path
        elif split == "test":
            file_path = self.validset_path
        else:
            raise ValueError("[PiQA] Unknown dataset split")
        
        json_data = json.load(open(file_path, 'r'))
        return json_data
    
    def load_dataset(self):
        trainset = self.load_json(split="train")
        testset = self.load_json(split="test")

        return trainset, testset
    
    def wrap_text(self, inputs:Dict):
        if self.apply_chat_template:
            inst = self.prompt_head + inputs["instruction"] + self.prompt_tail
        else:
            inst = inputs["instruction"]

        instruction = {"role": "user", "content": inst}
        return [instruction]
    
    def prepare_dataset(self, dataset:List):
        pbar = tqdm(dataset)
        inputs = []
        targets = []
        
        for sample in pbar:
            prepare_text = self.wrap_text(sample)

            if self.apply_chat_template:
                prepare_text = self.tokenizer.apply_chat_template(prepare_text, tokenize=False, add_generation_prompt=True)

            inputs.append(prepare_text)
            targets.append(sample["answer"])

        return {
            "dataset": inputs,
            "label": targets
        }
        
    
    def run(self):
        trainset, testset = self.load_dataset()

        trainset = self.prepare_dataset(trainset)
        testset = self.prepare_dataset(testset)
        return trainset, testset

"""
Math QA
"""

import json

from typing import Dict
from tqdm import tqdm
from typing import List
from src.stage.data import DataStage

class OpenBookQA(DataStage):
    def __init__(self, config_dir, tokenizer):
        super().__init__(config_dir)

        self.trainset_path = self.config["dataset"]["train"]
        self.validset_path = self.config["dataset"]["test"]
        self.apply_chat_template = self.config["dataset"].get("apply_chat_template", True)

        self.tokenizer = tokenizer

        # prompt template
        self.prompt_head = (
            "You are a careful reasoning assistant.\n"
            "A commonsense sentence is shown with a blank and **four** Answers that "
            "could fill the blank.\n"
            "Think step by step about which option makes the sentence most sensible, "
            "then decide.\n\n"
            "### Sentence:\n"
        )
        self.prompt_tail = (
            "\n\n### What you should do:\n"
            "1. Briefly explain your reasoning.\n"
            "2. On a **new line**, write exactly:\n"
            "   \"The final answer is [answer]\"\n"
            "   where **[answer]** is either **Answer1**, **Answer2**, **Answer3*, or **Answer4**.\n"
            "Do **not** output anything after that line."
        )

        self.ans_trigger = "The final answer is"

    def load_json(self, split):
        if split == "train":
            file_path = self.trainset_path
        elif split == "test":
            file_path = self.validset_path
        else:
            raise ValueError("[WinoGrande] Unknown dataset split")
        
        json_data = json.load(open(file_path, 'r'))
        return json_data
    
    def load_dataset(self):
        testset = self.load_json(split="test")

        return testset

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
        testset = self.load_dataset()
        testset = self.prepare_dataset(testset)
        return testset

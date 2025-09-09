import os
import pandas as pd
from tqdm import tqdm
from src.stage.data import DataStage

class MMLU(DataStage):
    """
    MMLU benchmark with 5-shot chain-of-thoughts

    Basic benchmark setup is adopted from: https://github.com/FranxYao/chain-of-thought-hub
    """
    def __init__(self, config_dir, tokenizer):
        super().__init__(config_dir)

        self.trainset_path = self.config["dataset"]["train"]
        self.validset_path = self.config["dataset"]["test"]
        self.tokenizer = tokenizer

        # prompt template
        self.cot_path = self.config["dataset"]["cot"]
        self.nshot = self.config["dataset"]["nshot"]

        # multi-choice
        self.choices = ["A", "B", "C", "D"]
        
    def prepare_cot(self, task:str):
        """
        5-shot CoT
        """
        filename = task + "_dev.csv"
        cot_task_path = os.path.join(self.cot_path, filename)

        chain_of_thoughts = pd.read_csv(cot_task_path, header=None)[:self.nshot]
        
        return chain_of_thoughts
    
    def load_dataset(self, task:str):
        filename = task + "_test.csv"
        valid_task_path = os.path.join(self.validset_path, filename)
        
        validset = pd.read_csv(valid_task_path, header=None)
        return validset
    
    def wrap_cot(self, dataset, idx, include_answer=True):
        prompt = dataset.iloc[idx, 0]
        k = dataset.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(self.choices[j], dataset.iloc[idx, j+1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(dataset.iloc[idx, k + 1])
        return prompt

    def format_subject(self, subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def prompt_template(self, cot, subject, k=-1):
        subject_txt = self.format_subject(subject)
        prompt = f"The following are multiple choice questions (with answers) about {subject_txt}.\n\n"

        if k == -1:
            k = cot.shape[0]
        for i in range(k):
            prompt += self.wrap_cot(cot, i)
        return prompt

    def few_shot_dataset(self, task:str):
        cot = self.prepare_cot(task)
        validset = self.load_dataset(task)
        
        cot_datasets = []
        cot_targets = []

        num_samples = validset.shape[0]
        for i in tqdm(range(num_samples)):
            prompt_end = self.wrap_cot(validset, i, include_answer=False)
            train_prompt = self.prompt_template(cot, task, self.nshot)

            prompt = train_prompt + prompt_end
            while len(self.tokenizer.tokenize(prompt)) + 1> 2048: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            label = validset.iloc[i, validset.shape[1]-1]

            cot_datasets.append(prompt)
            cot_targets.append(label)

        return {
            "dataset": cot_datasets,
            "label": cot_targets
        }
    
    def run(self, task):
        validset = self.few_shot_dataset(task)
        return validset
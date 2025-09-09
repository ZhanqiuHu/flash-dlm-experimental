"""
Dataset Stage of MATH500
"""

from typing import Dict
from tqdm import tqdm
from torch.utils.data import Dataset
from src.stage.data import DataStage
from datasets import load_dataset

class MATH500(DataStage):
    def __init__(self, config_dir):
        super().__init__(config_dir)
        self.dataset_name = "HuggingFaceH4/MATH-500"
        self.dataset = load_dataset(self.dataset_name, split="test")

    def wrap_text(self, inputs):
        instruction = [
            {"role": "system", "content": "Let's think step by step. And put your answer after \"Answer: \", and you need to use a \\boxed{} command to wrap your answer."},
            {"role": "user", "content": f"{inputs} Remember to put your final answer after \"Answer: \""}
        ]
        return [instruction]

    def run(self):
        print(f"Check dataset...")
        problems = self.dataset["problem"]
        solution = self.dataset["solution"]
        answers = self.dataset["answer"]

        # dataset placeholder
        dataset, target = [], []

        pbar = tqdm(problems)
        for i, prob in enumerate(pbar):
            prepared_text = self.wrap_text(prob)
            dataset.append(prepared_text)

        return {
            "dataset": dataset,
            "solution": solution,
            "target": answers
        }


class MATH500Dataset(Dataset):
    def __init__(self, dataset: Dict):
        super().__init__()

        self.data = dataset["dataset"]
        self.label = dataset["target"]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.label[index]

        return {"input_ids": sample, "label": label}

class MATH500Step(MATH500):
    def __init__(self, config_dir):
        super().__init__(config_dir)
        # system prompt template
        self.prompt_head: str = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always put your answer after \"Answer: \", and you need to use a \\boxed{} command to wrap your answer."

    def wrap_text(self, inputs):
        inst = self.prompt_head + inputs
        instruction = {"role": "user", "content": inst}
        return [instruction]
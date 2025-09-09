"""
PiQA dataset
"""

from itertools import islice
from src.stage.data import DataStage
from datasets import load_dataset

class PiQA(DataStage):
    def __init__(self, tokenizer, config_dir):
        super().__init__(config_dir)

        self.tokenizer = tokenizer

        # prompt template
        self.prompt_head = "Question: "
        self.prompt_tail = "\nAnswer:"

        # prepare dataset
        self.prepare_dataset()

        self.choices = ["sol1", "sol2"]

    def prepare_dataset(self):
        self.dataset = load_dataset(
            path="piqa",
            name=None,
            trust_remote_code=True
        )

    @property
    def trainset(self):
        if "train" in self.dataset.keys():
            # TODO: enable multi-gpu iterator
            trainset = list(islice(self.dataset["train"], 0, None, 1))
            return trainset
        else:
            print(f"The loaded dataset has no partition of 'train', available partitions: {self.dataset.keys()}")
            return None
    
    @property
    def validset(self):
        if "validation" in self.dataset.keys():
            # TODO: enable multi-gpu iterator
            validset = list(islice(self.dataset["validation"], 0, None, 1))
            return validset
        else:
            print(f"The loaded dataset has no partition of 'validation', available partitions: {self.dataset.keys()}")
            return None

    @property
    def testset(self):
        if "test" in self.dataset.keys():
            # TODO: enable multi-gpu iterator
            validset = list(islice(self.dataset["test"], 0, None, 1))
            return validset
        else:
            print(f"The loaded dataset has no partition of 'test', available partitions: {self.dataset.keys()}")
            return None
    
    def wrap_text(self, inputs):
        inst = self.prompt_head + inputs["goal"] + self.prompt_tail
        return inst
    
    def text2target(self, inputs):
        return inputs[self.choices]
    
    def prepare_prompt(self, inputs, target):
        arguments = [(inputs, f"{cont}") for cont in target]
        return
    
    def prepare_dataloader(self, dataset):
        pass

    def get_choice(self, doc):
        pass

    def run(self):
        trainset = self.trainset
        print(trainset[0])

class OpenBookQA(DataStage):
    def __init__(self, config_dir):
        super().__init__(config_dir)
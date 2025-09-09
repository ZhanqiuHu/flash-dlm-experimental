"""
MMLU Pro benchmark

Data preprocessing is adopted from: https://github.com/TIGER-AI-Lab/MMLU-Pro
"""
import re
from tqdm import tqdm
from datasets import load_dataset
from src.stage.data import DataStage

def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df

def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res

def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

class MMLUPro(DataStage):
    def __init__(self, config_dir):
        super().__init__(config_dir)
        self.choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
        self.nshot = self.config["dataset"].get("nshot", 5)

    def load_dataset(self):
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        test_df, val_df = dataset["test"], dataset["validation"]
        
        test_df = preprocess(test_df)
        val_df = preprocess(val_df)
        return test_df, val_df
    
    def prepare_subjects(self):
        all_subjects = []
        testset, validset = self.load_dataset()

        for each in testset:
            if each["category"] not in all_subjects:
                all_subjects.append(each["category"])

        return sorted(all_subjects), testset, validset

    def select_by_category(self, df, subject):
        res = []
        for each in df:
            if each["category"] == subject:
                res.append(each)
        return res

    def format_cot_example(self, example, including_answer=True):
        prompt = "Question:\n"
        question = example["question"]
        options = example["options"]

        prompt += question + "\n"
        prompt += "Options:\n"
        for i, opt in enumerate(options):
            prompt += "{}. {}\n".format(self.choices[i], opt)
        if including_answer:
            cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                        "Answer: Let's think step by step.")
            prompt += cot_content + "\n\n"
        else:
            prompt += "Answer: Let's think step by step."
            # prompt += "Please briefly explain your reasoning for your choice, and then provide the final answer."
        return prompt

    def wrap_text(self, validset, curr):
        subject = curr["category"]
        validset = select_by_category(validset, subject)
        validset = validset[: self.nshot]

        prompt = ""
        # with open(f"/home/jm2787/MMLU-Pro/cot_prompt_lib/initial_prompt.txt", "r") as fi:
        #     for line in fi.readlines():
        #         prompt += line

        with open(f"dataset/mmlu-pro/initial_prompt.txt", "r") as fi:
            for line in fi.readlines():
                prompt += line

        prompt = prompt.replace("{$}", subject) + "\n"
        
        for example in validset:
            prompt += self.format_cot_example(example, including_answer=True)
        
        prompt += self.format_cot_example(curr, including_answer=False)
        answer = curr["answer"]
        return prompt, answer
    
    def prepare_cot(self, testset, validset, subject:str):
        batches = []
        answers = []
        
        pbar = tqdm(range(len(testset)))
        for i in pbar:
            curr = testset[i]
            prompt, answer = self.wrap_text(validset, curr)
            batches.append(prompt)
            answers.append(answer)

            pbar.set_description(subject)

        return batches, answers
    
    def run(self):
        all_subjects, testset, validset = self.prepare_subjects()
        whole_set = {}

        for subject in all_subjects:
            test_df = self.select_by_category(testset, subject)
            val_df = self.select_by_category(validset, subject)

            # fetch the questions and answers
            batch, answers = self.prepare_cot(test_df, val_df, subject)

            assert len(batch) == len(answers), f"Missing answers for dataset {subject}"
            whole_set[subject] = {"sample": batch, "label": answers}

        return whole_set

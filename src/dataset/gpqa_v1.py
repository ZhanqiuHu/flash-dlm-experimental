"""
GPQA dataset (5-shot using moved prompt file)
"""

import json
import os
from tqdm import tqdm
from typing import Dict, List
from datasets import load_dataset
from src.stage.data import DataStage

# Standard fg colors
BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"
PURPLE = "\033[35m"
ORANGE = "\033[33m"
PINK = "\033[31m"

# Bright / "light" variants
BRIGHT_BLACK   = "\033[90m"  # often used for DEBUG
BRIGHT_RED     = "\033[91m"
BRIGHT_GREEN   = "\033[92m"
BRIGHT_YELLOW  = "\033[93m"
BRIGHT_BLUE    = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN    = "\033[96m"
BRIGHT_WHITE   = "\033[97m"

# Styles
BOLD      = "\033[1m"
DIM       = "\033[2m"
UNDERLINE = "\033[4m"
RESET     = "\033[0m"



class GPQA(DataStage):
    def __init__(self, config_dir, tokenizer):
        super().__init__(config_dir)

        self.tokenizer = tokenizer
        self.apply_chat_template = self.config["dataset"].get("apply_chat_template", True)

        # path to your 5-shot prompt JSON
        prompt_path = self.config["dataset"].get(
            "five_shot_prompt_path",
            os.path.join("dataset", "gpqa", "prompts", "chain_of_thought_examples.json")
        )
        # load raw JSON
        with open(prompt_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # ensure we have a list of message dicts
        if isinstance(raw, dict):
            # find first value that is a list
            list_vals = [v for v in raw.values() if isinstance(v, list)]
            if not list_vals:
                raise ValueError(f"No list found in JSON at {prompt_path}")
            self.five_shot_msgs: List[Dict[str, str]] = list_vals[0]
        elif isinstance(raw, list):
            self.five_shot_msgs: List[Dict[str, str]] = raw
        else:
            raise ValueError(f"Unsupported JSON structure for five-shot prompts: {type(raw)}")

        # prompt head/tail for each example
        self.prompt_head = (
            "You are a careful reasoning assistant.\n"
            "A question is shown with **four** possible answers.\n"
            "Think step by step about which option is correct, then decide.\n\n"
        )
        self.prompt_tail_option1 = (
            "\n\n### What you should do:\n"
            "1. Briefly explain your reasoning.\n"
            "2. On a **new line**, write exactly:\n"
            "   \"The final answer is [answer]\"\n"
            "   where **[answer]** is either **Answer1**, **Answer2**, **Answer3**, or **Answer4**.\n"
            "Do **not** output anything after that line."
        )
        self.prompt_tail_option2 = (
            "\n\n### Instructions:\n"
            "First, briefly explain your reasoning.\n"
            "Then on its own line, write exactly one of: Answer1, Answer2, Answer3, or Answer4\n"
            "—and nothing else.\n"
            "Do not include any additional text or punctuation."
        )
        self.prompt_tail_option3 = (
            "\n\nPlease think through your answer step by step.\n"
            "When youre done, on a new line write exactly one of: Answer1, Answer2, Answer3, Answer4.\n"
            "Do not add any other words or symbols."
        )
        self.prompt_tail_option4 = (
            "\n\n### In in your response:\n"
            "1. First reason and explain your reasoning briefly before providing the final answer to the question.\n"
            "2. Then at the end of your explanation, on a **new line**, write exactly:\n"
            "   \"The final answer is [answer]\"\n"
            "   where **[answer]** is one of **Answer1**, **Answer2**, **Answer3**, or **Answer4**.\n"
            "Do **not** output anything after that line."
        )
        self.prompt_tail_no_explanation = (
            "\n\nNow give only the final answer as one of: Answer1, Answer2, Answer3, Answer4\n"
            "with no explanation or extra text."
        ) 

        self.prompt_tail = self.prompt_tail_option4

        # Logging the promt head and tail used
        print(f"{BLUE}{BOLD}GPQA 5-shot prompt head: {self.prompt_head}{RESET}")
        print(f"{BLUE}{BOLD}GPQA 5-shot prompt tail: {self.prompt_tail}{RESET}")





        


        # self.ans_trigger = "The final answer is"
        self.ans_trigger = "final answer is"

    def load_dataset(self):
        """Load the GPQA dataset—use only the 'train' split."""
        ds = load_dataset("idavidrein/gpqa", "gpqa_main")
        return ds["train"]

    def wrap_single(self, sample: Dict) -> str:
        """Format one Q&A block (question + four AnswerX: options)."""
        question = sample["Question"]
        opts = [
            sample["Correct Answer"],
            sample["Incorrect Answer 1"],
            sample["Incorrect Answer 2"],
            sample["Incorrect Answer 3"],
        ]
        s = question + "\n\n"
        for i, o in enumerate(opts, 1):
            s += f"Answer{i}: {o}\n"
        return s

    def prepare_dataset(self, dataset):
        """
        Build 5-shot prompts by prefixing each test sample with the fixed
        five-shot messages loaded from JSON.
        """
        inputs, targets = [], []
        for sample in tqdm(dataset, desc="GPQA 5-shot prep"):
            # test block
            test_block = self.prompt_head + self.wrap_single(sample) + self.prompt_tail
            test_msgs = [{"role": "user", "content": test_block}]

            # combine demos + test
            all_msgs = self.five_shot_msgs + test_msgs

            # apply chat template & add generation prompt if needed
            if self.apply_chat_template:
                try:
                    all_msgs = self.tokenizer.apply_chat_template(
                        all_msgs,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    # Convert five-shot examples to proper chat format
                    formatted_five_shot = []
                    for example in self.five_shot_msgs:
                        # Format question and choices
                        question_text = example["question"]
                        choices_text = "\n".join([f"Answer{i+1}: {choice}" for i, choice in enumerate(example["choices"].values())])
                        full_question = f"{question_text}\n\n{choices_text}"
                        
                        # Add user message
                        formatted_five_shot.append({
                            "role": "user",
                            "content": self.prompt_head + full_question + self.prompt_tail
                        })
                        
                        # Add assistant message
                        formatted_five_shot.append({
                            "role": "assistant",
                            "content": f"{example['explanation']}\n\nThe final answer is Answer{ord(example['correct_answer']) - ord('A') + 1}"
                        })
                    
                    # Combine with test message and try again
                    all_msgs = formatted_five_shot + test_msgs
                    all_msgs = self.tokenizer.apply_chat_template(
                        all_msgs,
                        tokenize=False,
                        add_generation_prompt=True
                    )

            inputs.append(all_msgs)
            
            # Find the index of the correct answer
            correct_answer = sample["Correct Answer"]
            opts = [
                sample["Correct Answer"],
                sample["Incorrect Answer 1"],
                sample["Incorrect Answer 2"],
                sample["Incorrect Answer 3"],
            ]
            answer_idx = opts.index(correct_answer) + 1
            targets.append(f"Answer{answer_idx}")

        return {
            "dataset": inputs,
            "label": targets
        }

    def run(self):
        ds = self.load_dataset()
        testset = self.prepare_dataset(ds)
        return None, testset

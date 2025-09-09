"""
GSM8K
"""

import copy
import json
import gzip
import random

from typing import Dict, Union
from tqdm import tqdm
from torch.utils.data import Dataset
from src.stage.data import DataStage

class GSM8K(DataStage):
    def __init__(self, config_dir: Union[str, Dict], tokenizer):
        super().__init__(config_dir)
        if isinstance(config_dir, dict):
            self.config = config_dir
        self.trainset_path = self.config["dataset"]["train"]
        self.validset_path = self.config["dataset"]["test"]
        self.prm = self.config["dataset"].get("prm", False)
        self.nshot = self.config["dataset"].get("nshot", 8)
        self.apply_chat_template = self.config["dataset"].get("apply_chat_template", True)
        self.tokenizer = tokenizer

        # prompt template
        if not self.prm:
            self.prompt_head = "Given the following problem, reason and give a final answer to the problem. \nProblem:"
            # self.prompt_tail = "\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n"
            self.prompt_tail = "\nYour response should end with \"The final answer is [answer]\" where [answer] is the Where [answer] is just the final number to the problem.\n"
            
            self.ans_trigger = "The final answer is"
        else:
            self.logger.info(f"\nStep-by-step Eval with prm flag = {self.prm}\n")
            self.prompt_head: str = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: [answer]. Where [answer] is just the final number or expression that solves the problem. "
            self.prompt_tail = ""
            self.ans_trigger = "The final answer is"

            chat_template: str = '{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'
            self.tokenizer.chat_template = chat_template


        self.cot_flag = self.config["eval"]["cot"]
        self.chain_of_thoughts = self.prepare_cot()

    def wrap_text(self, inputs, is_cot:bool=False):
        if self.apply_chat_template:
            inst = self.prompt_head + inputs["instruction"] + self.prompt_tail
        else:
            inst = inputs["instruction"]
        instruction = {"role": "user", "content": inst}

        if is_cot:
            assert "chain" in inputs.keys()
            chain = inputs["chain"]

            cot = {"role": "assistant", "content": chain}
            
            return [instruction, cot]
        else:
            return [instruction]

    def prepare_cot(self):
        """
        8-shot COT.
        """
        chain_of_thoughts = []

        chain_of_thoughts.append({
            "instruction": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "chain": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6"
        })
        
        chain_of_thoughts.append({
            "instruction": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "chain": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5"
        })

        chain_of_thoughts.append({
            "instruction": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "chain": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39"
        })

        chain_of_thoughts.append({
            "instruction": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
            "chain": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8"
        })

        chain_of_thoughts.append({
            "instruction": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
            "chain": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9"
        })

        chain_of_thoughts.append({
            "instruction": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
            "chain": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29"
        })

        chain_of_thoughts.append({
            "instruction": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
            "chain": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33"
        })

        chain_of_thoughts.append({
            "instruction": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
            "chain": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8"
        })

        return chain_of_thoughts

    def load_jsonl(self, split):
        is_gzip = ".zip" in self.trainset_path
        
        if split == "train":
            file_path = self.trainset_path
        elif split == "test":
            file_path = self.validset_path
        else:
            raise ValueError("[GSM8K] Unknown dataset split")

        # data list
        collected_data = []

        open_func = open if not is_gzip else gzip.open

        with open_func(file_path, "r") as f:
            for line in f:
                item = json.loads(line)

                # access the data item
                new_item = dict(
                    instruction = item["question"] if "question" in item else None,
                    output = item["answer"] if "answer" in item else None, 
                )

                collected_data.append(new_item)
    
        return collected_data

    def load_dataset(self):
        trainset = self.load_jsonl(split="train")
        validset = self.load_jsonl(split="test")

        return trainset, validset
    
    def wrap_cot(self):
        shuffled_cot = list(range(len(self.chain_of_thoughts)))
        random.shuffle(shuffled_cot)

        instruction = []

        for cot in self.chain_of_thoughts:
            wrapped_cot = self.wrap_text(cot, is_cot=True)
            instruction += wrapped_cot
        
        return instruction
    
    def few_shot_dataset(self, dataset):
        cot_datasets = []
        cot_targets = []
        
        pbar = tqdm(dataset)
        for sample in pbar:
            # chain of thoughts
            cot = self.wrap_cot()

            # context of question
            prepare_text = self.wrap_text(sample, is_cot=False)

            cot += prepare_text

            if self.apply_chat_template:
                cot = self.tokenizer.apply_chat_template(cot, tokenize=False, add_generation_prompt=True)
            
            # fetch labels and datasets
            cot_datasets.append(cot)
            cot_targets.append(sample["output"])

        return {
            "dataset": cot_datasets,
            "label": cot_targets
        }
    
    def zero_shot_dataset(self, dataset):
        inputs = []
        targets = []
        
        pbar = tqdm(dataset)
        for sample in pbar:
            
            # context of question
            prepare_text = self.wrap_text(sample, is_cot=False)
            
            if self.apply_chat_template:
                prepare_text = self.tokenizer.apply_chat_template(prepare_text, tokenize=False, add_generation_prompt=True)

            # fetch labels and datasets
            inputs.append(prepare_text)
            targets.append(sample["output"])

        return {
            "dataset": inputs,
            "label": targets
        }

    def run(self):
        trainset, validset = self.load_dataset()

        trainset = self.zero_shot_dataset(trainset)
        
        if self.nshot == 0:
            validset = self.zero_shot_dataset(validset)
        else:
            validset = self.few_shot_dataset(validset)
        return trainset, validset


class GSM8KDataset(Dataset):
    def __init__(self, dataset:Dict):
        super().__init__()
    
        self.data = dataset["input_ids"]
        self.label = dataset["labels"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.label[index]

        return {"input_ids": sample, "label": label}


"""
Grade School Math Fine-tuning with CoT
"""
import json
import torch

from torch.utils.data.dataloader import DataLoader
from functools import partial
from collections import defaultdict
from datasets import DatasetDict, Dataset
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from src.stage.data import DataStage

class GSM8KSFT(DataStage):
    """
    Dataset stage for supervised fine-tuning (SFT).

    The GSM8K fine-tuning datasets are adopted from: https://arxiv.org/abs/2401.08967
    """
    def __init__(self, config_dir, tokenizer, accelerator):
        super().__init__(config_dir)

        self.tokenizer = tokenizer
        self.trainset_path = self.config["dataset"]["train"]
        self.validset_path = self.config["dataset"]["test"]

        # accelerator
        self.accelerator = accelerator

    def prepare_cot(self):
        pass

    def prepare_dataloader(self):
        with self.accelerator.main_process_first():
            raw_dataset = DatasetDict({
                'train': Dataset.from_list(json.load(open(self.trainset_path,'r'))),
                'test': Dataset.from_list(json.load(open(self.validset_path,'r'))),
            })

            def tokenize_fn(batch, tokenizer):
                assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
                new_batch = defaultdict(list)
                all_keys = list(batch.keys())
                
                instruction = "Question:\n"
                cot_trigger = "\nAnswer reasoning:\n"
                answer_trigger = "\nTherefore, the answer is: "

                for item_values in zip(*(batch[k] for k in all_keys)):
                    item = {k: item_values[i] for i, k in enumerate(all_keys)}

                    item_id, question, answer_value, answer_cot = \
                        item['item_id'], \
                        item['question'], \
                        item['answer_value'], \
                        item.get('answer_cot', None), \
                        
                    question = question.strip()
                
                    if answer_value is not None:
                        answer_value = answer_value.strip()

                    if answer_cot is not None:
                        answer_cot = answer_cot.strip()
                        answer_cot += f'{answer_trigger}{answer_value}'
                    
                    # get the full prompt with CoT
                    input = f"{instruction}{question}{cot_trigger}"
                    output = f"{answer_cot}"
                    prefix_text = f'{instruction}{question}{cot_trigger}'

                    # encode the input data with tokenizer
                    input_encode = tokenizer(input, add_special_tokens=False)
                    output_encode = tokenizer(output, add_special_tokens=False)
                    prefix_encode = tokenizer(prefix_text, add_special_tokens=False)

                    # fuse the cot input and output, followed by eos
                    input_ids = input_encode['input_ids'] + output_encode['input_ids'] + [tokenizer.eos_token_id]
                    labels = [-100]*len(input_encode['input_ids']) + output_encode['input_ids'] + [tokenizer.eos_token_id]

                    # all one values in attention mask => all tokens are real tokens
                    attention_mask = [1]* len(input_ids)

                    prefix = prefix_encode['input_ids']
                    prefix_attention_mask = prefix_encode['attention_mask']

                    # truncation
                    input_ids_max_length = len(input_ids)

                    max_input_length = self.config["train"]["max_input_length"]
                    input_ids = input_ids[:max_input_length]
                    labels = labels[:max_input_length]
                    attention_mask = attention_mask[:max_input_length]
                    prefix = prefix[:max_input_length]
                    prefix_attention_mask = prefix_attention_mask[:max_input_length]

                    # now formulate the final batch 
                    new_batch['input_ids'].append(input_ids)
                    new_batch['labels'].append(labels)
                    new_batch['attention_mask'].append(attention_mask)
                    new_batch['prefix'].append(prefix)
                    new_batch['prefix_attention_mask'].append(prefix_attention_mask)

                    new_batch['item_id'].append(item_id)
                    new_batch['question'].append(question)
                    new_batch['answer_cot'].append(answer_cot)
                    new_batch['answer_value'].append(answer_value)
                    new_batch['input_ids_max_length'].append(input_ids_max_length)
            
                return new_batch
            
            tokenized_dataset = DatasetDict({
                mode: dataset.map(
                    tokenize_fn, 
                    fn_kwargs={'tokenizer': self.tokenizer}, 
                    batched=True, 
                    remove_columns=dataset.column_names,
                    num_proc=8, 
                    load_from_cache_file=False
                ) for mode, dataset in raw_dataset.items()
            })

            self.accelerator.print('Processed data:', tokenized_dataset)

        def collate_fn(batch, tokenizer):
            max_input_length = max([len(item['input_ids']) for item in batch])
            max_target_length = max([len(item['labels']) for item in batch])
            max_prefix_length = max([len(item['prefix']) for item in batch])

            input_ids  = []
            attention_mask  = []
            labels, labels_left_padded  = [], []
            prefix_left_padded  = []
            prefix_attention_mask_left_padded  = []

            # unify the size of the input via padding
            for item in batch:
                input_ids.append(item['input_ids'] + [tokenizer.pad_token_id]*(max_input_length - len(item['input_ids'])))
                attention_mask.append(item['attention_mask'] + [0]*(max_input_length - len(item['attention_mask'])))
                labels.append(item['labels'] + [-100]*(max_target_length - len(item['labels'])))

                labels_left_padded.append([-100]*(max_target_length - len(item['labels'])) + item['labels'])
                prefix_left_padded.append([tokenizer.pad_token_id]*(max_prefix_length - len(item['prefix'])) + item['prefix'])
                prefix_attention_mask_left_padded.append([0]*(max_prefix_length - len(item['prefix_attention_mask'])) + item['prefix_attention_mask'])
            
            forward_kwargs = {
                'input_ids': torch.LongTensor(input_ids),
                'attention_mask': torch.BoolTensor(attention_mask),
                'labels': torch.LongTensor(labels)
            }
            
            generate_prefix_kwargs = {
                'input_ids': torch.LongTensor(prefix_left_padded),
                'attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
                'labels': torch.LongTensor(labels_left_padded)
            }
            
            return {'forward_kwargs': forward_kwargs, 'generate_prefix_kwargs': generate_prefix_kwargs}
        
        train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=self.config['train']['batch_size'], num_workers=self.config['train']['num_workers'], pin_memory=True, 
                            collate_fn=partial(collate_fn, tokenizer=self.tokenizer))
                            
        test_dataloader = DataLoader(tokenized_dataset['test'], shuffle=False, batch_size=self.config['train']['eval_batch_size'], num_workers=self.config['train']['num_workers'], pin_memory=True, 
                            collate_fn=partial(collate_fn, tokenizer=self.tokenizer))
        
        return (tokenized_dataset['train'], train_dataloader), (tokenized_dataset['test'], test_dataloader)
    
    def run(self):
        (train_dataset, train_dataloader), (test_dataset, test_dataloader) = self.prepare_dataloader()
        return train_dataset, train_dataloader, test_dataset, test_dataloader


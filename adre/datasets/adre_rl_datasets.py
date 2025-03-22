from torch.utils.data import Dataset
from collections import defaultdict
import torch

def adre_collate_fn(batch, pad_id, padding_side='left',num_experts=1):
    result = {}
    keys = batch[0].keys()
    for key in keys:
        if key == 'input_ids':
            max_length = max(len(item[key]) for item in batch)
            padded_inputs = []
            attention_mask = []
            for item in batch:
                input_ids = item[key]
                padding_length = max_length - len(input_ids)
                
                padded_input = input_ids + [pad_id] * padding_length
                padded_inputs.append(padded_input)
                if padding_side == 'left':
                    mask = [0] * padding_length + [1] * len(input_ids)
                    attention_mask.append(mask)
                else:
                    mask = [1] * len(input_ids) + [0] * padding_length
                    attention_mask.append(mask)
            
            result[key] = torch.tensor(padded_inputs)
            result['attention_mask'] = torch.tensor(attention_mask)
        elif key=='rank':
            values = [item[key] if item[key] == -100 else int(item[key] * num_experts) for item in batch]
            result[key] = torch.tensor(values)
        elif key in ['indices', 'rank']:
            values = [item[key] for item in batch]
            result[key] = torch.tensor(values) if None not in values else values
        else:
            values = [item[key] for item in batch]
            if isinstance(values[0], (int, float)) and None not in values:
                result[key] = torch.tensor(values)
            else:
                result[key] = values
    return result

class PromptDataset(Dataset):
    def __init__(self, dataset, tokenizer,prompt_key='prompt', 
                 label_key=None, max_prompt_length =1024,type_key=None, post_fix=None,
                 ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompt_key = prompt_key
        self.label_key = label_key
        self.post_fix = post_fix
        self.type_key = type_key

        self.tokenized_prompts = None
        self._pre_process()
        self._filter_prompt()

        self.accumulated_rewards = {i: 0.0 for i in range(len(dataset))}
        self.reward_ranks = {i: -100 for i in range(len(dataset))}

    def _pre_process(self):
        """
        tokenize the prompts
        """
        self.tokenized_prompts = []
        for prompt in self.dataset[self.prompt_key]:
            if isinstance(prompt, str):
                # if the prompt is a string, we take it as pre-processed prompt
                self.tokenized_prompts.append(self.tokenizer(prompt,add_special_tokens=False).input_ids)

            elif isinstance(prompt, list):
                # if the prompt is a list of messages, we need to apply the chat template
                input_text = self.tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)

                if self.post_fix:
                    # for deepseek-r1 style, prompt always ends with <think>\n
                    if not input_text.endswith(self.post_fix):
                        input_text = input_text + self.post_fix
                
                self.tokenized_prompts.append(self.tokenizer(input_text,add_special_tokens=False).input_ids)
            else:
                raise ValueError(f"prompt must be a string or a list of messages, but got {type(prompt)}")
    
    def _filter_prompt(self):
        """
        filter the prompt by the max length
        """
        filtered_indices = [i for i, prompt in enumerate(self.tokenized_prompts) if len(prompt) <= self.max_prompt_length]
        raw_length = len(self.dataset)
        self.tokenized_prompts = [self.tokenized_prompts[i] for i in filtered_indices]
        self.dataset = self.dataset.select(filtered_indices)

        print(f"filtered {raw_length - len(self.dataset)} prompts from {raw_length}")
        self.accumulated_rewards = {i: 0.0 for i in range(len(self.dataset))}
        
    def __len__(self):
        return len(self.tokenized_prompts)
        
    def __getitem__(self, index):
        input_ids = self.tokenized_prompts[index]
        res={'input_ids': input_ids, 'indices': index}  
        if self.label_key:
            res['label'] = self.dataset[self.label_key][index]
        if self.type_key:
            res['data_type'] = self.dataset[self.type_key][index]
        res['rank'] = self.reward_ranks.get(index, -100)
        return res
    
    def update_reward(self, index, reward):
        self.accumulated_rewards[index] += reward
    
    def get_rewards(self):
        return self.accumulated_rewards
    
    def calculate_reward_ranks(self):
        rewards_items = list(self.accumulated_rewards.items())
        sorted_rewards = sorted(rewards_items, key=lambda x: x[1], reverse=True)
        for rank, (index, _) in enumerate(sorted_rewards):
            self.reward_ranks[index] = rank / (len(self.accumulated_rewards))

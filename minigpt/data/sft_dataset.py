import json
import time
import torch
import linecache
from functools import partial
from torch.utils.data import Dataset, DataLoader

class InstructionDataset(Dataset):
    def __init__(self, jsonl_file_path, tokenizer, max_len=1024, max_lines=0):
        self.jsonl_file_path = jsonl_file_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        start_time = time.time()
        # 如果入参没有指定最大条数，则自动计算文件中的总行数作为总条数，反之则以入参指定的总条数为准
        if max_lines <= 0:
            with open(self.jsonl_file_path, 'r', encoding='utf-8') as f:
                self.total_lines = sum(1 for _ in f)
        else:
            self.total_lines = max_lines
        print(f"calculate lines[{self.total_lines}] use time: {time.time()-start_time:.3f}s")

    def __len__(self):
        return self.total_lines
    
    def process(self, item):
        messages = []
        for history_item in item.get("history", []):
            if len(history_item) < 2:
                continue
            messages.append({"role": "user", "content": history_item[0][:self.max_len//2]})
            messages.append({"role": "assistant", "content": history_item[1][:self.max_len//2]})
        
        user_content = item['instruction'] + '\n' + item['input']
        assistant_content = item['output']
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if not isinstance(input_ids, list):  # transformers>=5.0 返回 BatchEncoding，解包为 list[int]
            input_ids = input_ids["input_ids"]
        return input_ids[:self.max_len]

    def __getitem__(self, idx):
        indexes = [idx] if isinstance(idx, int) else idx
        assert isinstance(indexes, list)
    
        # 使用 linecache 读取指定行
        lines = [linecache.getline(self.jsonl_file_path, i + 1).strip() for i in indexes]
        datas = [json.loads(line) for line in lines]
        inputs =  [self.process(data) for data in datas]
        return inputs[0] if isinstance(idx, int) else inputs

def find_sublist_index(main_list, sub_list):
    for i in range(len(main_list) - len(sub_list) + 1):
        if main_list[i:i + len(sub_list)] == sub_list:
            return i
    return -1

def calc_label(input_ids, pad_token_id, tokenizer):
    target_ids = input_ids[1:] + [pad_token_id]
    output_seperator = tokenizer("<|im_start|>assistant\n")['input_ids']
    output_start_index = find_sublist_index(target_ids, output_seperator)
    if output_start_index == -1:
        # 找不到 assistant 分隔符（例如 chat_template 不一致）时退化为不遮蔽指令部分，
        # 避免 instruction_length 变成负数导致掩码逻辑错乱。
        instruction_length = 0
    else:
        instruction_length = output_start_index + len(output_seperator)
    label_ids = [-100 if (item == pad_token_id or i < instruction_length) else item for i, item in enumerate(target_ids)]
    return label_ids

def collate(batch_inputs, pad_token_id, tokenizer, device='cpu'):
    assert isinstance(pad_token_id, int)
    # pad each sequence to max_length
    max_length = max([len(item) for item in batch_inputs])
    batch_padded = [item + [pad_token_id] * (max_length - len(item)) for item in batch_inputs]
    input_tensors = torch.tensor(batch_padded, dtype=torch.int64).to(device)

    attention_mask = torch.ones(input_tensors.shape, dtype=torch.int64).to(device)
    attention_mask = attention_mask.masked_fill(input_tensors == pad_token_id, 0)

    batch_targets = [calc_label(item, pad_token_id, tokenizer) for item in batch_padded]
    target_tensors = torch.tensor(batch_targets, dtype=torch.int64).to(device)
    return input_tensors, target_tensors, attention_mask

def split_dataset(data, train_ratio, eval_ratio):
    train_len = int(len(data) * train_ratio)
    eval_len = int(len(data) * eval_ratio)
    test_len = len(data) - train_len - eval_len
    return torch.utils.data.random_split(data, [train_len, eval_len, test_len])

def create_batch_collator(tokenizer):
    return partial(
        collate,
        pad_token_id = tokenizer.unk_token_id,
        tokenizer = tokenizer, 
    )

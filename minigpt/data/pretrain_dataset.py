import os
import json
import numpy as np
import torch
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader,  random_split, Subset


def read_text_dataset(data_path, max_size=100*1024*1024, content_key='text'):
    with open(data_path, 'r', encoding='utf-8') as f:
        current_size = 0
        current_texts = []
        while True:
            line = f.readline()
            if not line:
                if current_texts:
                    yield current_texts
                break

            data = json.loads(line)
            current_texts.append(data[content_key])
            current_size += len(data[content_key])
            if current_size >= max_size:
                yield current_texts
                current_texts = []
                current_size = 0

class PretrainTextDataset(Dataset):
    def __init__(self, texts: list, tokenizer, max_length, stride=1):
        self.max_length = max_length
        self.stride = stride
        self.tokenizer: AutoTokenizer = tokenizer
        separator = self.tokenizer.unk_token
        token_ids = tokenizer.encode(separator.join(texts)+separator)
        self.input_set = []
        self.target_set = []
        for j in range(0, len(token_ids) - self.max_length, self.stride):
            input_ids = token_ids[j: j + self.max_length]
            target_ids = token_ids[j+1: j + self.max_length + 1]
            self.input_set.append(torch.tensor(input_ids))
            self.target_set.append(torch.tensor(target_ids))
        
    def __len__(self):
        return len(self.input_set)
        
    def __getitem__(self, i):
        return self.input_set[i], self.target_set[i]

def create_dataloaders_from_texts(texts_data, train_ratio, tokenizer, max_tokens, batch_size):
    train_size = int(len(texts_data) * train_ratio)
    train_texts = texts_data[:train_size]
    eval_texts = texts_data[train_size:]

    train_dataset = PretrainTextDataset(train_texts, tokenizer, max_tokens, max_tokens)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    eval_dataset = PretrainTextDataset(eval_texts, tokenizer, max_tokens, max_tokens)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, eval_loader

def texts_to_bin(input_path, output_path, tokenizer, content_key="content"):
    bos_token = tokenizer.special_tokens_map['bos_token']
    eos_token = tokenizer.special_tokens_map['eos_token']
    max_buffered_length = 1 * 1024 * 1024
    with open(input_path, "r", encoding="utf-8") as reader:
        with open(output_path, "wb") as writer:
            buffered_ids = []
            i = 0
            while True:
                line = reader.readline()
                if not line:
                    break
                content = json.loads(line).get(content_key, "")
                if not content:
                    continue
                
                # 将数据序列化为二进制格式
                tokenized = tokenizer(bos_token + content + eos_token)
                buffered_ids += tokenized["input_ids"]
                if len(buffered_ids) >= max_buffered_length:
                    arr = np.array(buffered_ids, dtype=np.uint16)
                    writer.write(arr.tobytes())
                    buffered_ids.clear()
                    i += 1
                    print(f"write {i}m bytes") if i % 100 == 0 else None
            # 处理最后一段不满max_buffer_length的token序列
            if len(buffered_ids) > 0:
                arr = np.array(buffered_ids, dtype=np.uint16)
                writer.write(arr.tobytes())
                print(f"write arr: {len(arr)}")

class PretrainBinaryDataset(Dataset):
    def __init__(self, data_path, max_tokens):
        # 二进制文件必须以 rb 打开，文本模式下 seek/tell 计算字节数不可靠
        with open(data_path, 'rb') as f:
            f.seek(0, 2)
            self.total_tokens = f.tell() // np.dtype("uint16").itemsize
            print(f"total_tokens: {self.total_tokens}")
        
        self.data = np.memmap(data_path, dtype=np.uint16, shape=(self.total_tokens//max_tokens, max_tokens))

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_single_item(index)
        elif isinstance(index, slice):
            return self._get_slice_items(index)
        elif isinstance(index, list):
            return self._get_list_items(index)
        else:
            raise TypeError(f"unknown param type of {type(index)}")

    def _get_single_item(self, index):
        assert isinstance(index, int)
        item = self.data[index]
        input = item[:-1].astype(np.int64)
        target = item[1:].astype(np.int64)  # 在计算交叉熵损失时要求目标输出为长整型
        # attention_mask = torch.ones(len(input), dtype=torch.int64)  # 长度与 input 相同
        return torch.from_numpy(input), torch.from_numpy(target)
    
    def _get_list_items(self, indexes):
        assert isinstance(indexes, list)

        items = [self.data[index] for index in indexes]
        inputs = [item[:-1] for item in items]
        targets = [item[1:] for item in items]  
        # 在计算交叉熵损失时要求目标输出为长整型
        input_tensors = torch.tensor(inputs, dtype=torch.int64)
        target_tensors = torch.tensor(targets, dtype=torch.int64)
        # 创建全1的 attention_mask，形状为 (batch_size, max_seq_length)  
        # batch_size = len(inputs)  
        # attention_mask = torch.ones(batch_size, input_tensors.size(1), dtype=torch.int64)  # 假设所有输入长度一致  

        return input_tensors, target_tensors
    
    def _get_slice_items(self, index):
        # slice为内置切片对象，indices方法返回start, stop, step三个元素的元组, range(start, stop, step)则返回一个正常的索引迭代器
        return Subset(self, range(*index.indices(len(self))))

def split_dataset(data, train_ratio):
    train_len = int(len(data) * train_ratio)
    eval_len = len(data) - train_len
    return random_split(data, [train_len, eval_len])

def create_dataloaders(ds, batch_size, train_ratio, local_rank=-1):
    train_set, eval_set = split_dataset(ds, train_ratio)
    sampler = DistributedSampler(train_set, rank=local_rank) if local_rank >= 0 else None
    shuffle = True if sampler == None else False
    # num_workers用于epoch结束后不关闭workers，但实际测试，我们这个场景下用多进程num_workers加载，并不能提高速度。
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True, sampler=sampler)
    # 陷阱：这里的eval_set已经是被split_dataset切过的SubSet类型，它无法再次切割。
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    return train_loader, eval_loader


# ---------------------------------------------------------------------------
# 生产级数据管线（v2）：jsonl -> uint16/uint32 .bin + .meta.json 元数据
# ---------------------------------------------------------------------------
import json as _json

def _resolve_meta(bin_path, meta_path):
    if meta_path:
        return meta_path
    return os.path.splitext(bin_path)[0] + ".meta.json"

def tokenize_jsonl_to_bin(input_path, output_path, tokenizer, content_key="text",
                          max_lines=0, dtype=None, meta_path="", write_every=200000):
    """把 jsonl 语料（按行 {content_key: ...}）序列化为二进制 token 流，并写出元数据。

    - 每条文本后追加 tokenizer.eos_token_id 作为文档结束符（无 BOS 的 tokenizer 不加前缀）；
    - 词表 >65535 时自动使用 uint32，否则默认 uint16（可用 dtype 显式指定）；
    - 返回 (lines, tokens, dtype)；同目录生成 .meta.json 供训练侧自动识别。
    """
    vocab_size = len(tokenizer)
    dtype = dtype or ("uint32" if vocab_size > 65535 else "uint16")
    eos_id = tokenizer.eos_token_id
    n_lines, n_tokens, n_chunks = 0, 0, 0
    buf = []
    out_meta = _resolve_meta(output_path, meta_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as reader, open(output_path, "wb") as writer:
        while True:
            line = reader.readline()
            if not line:
                break
            if max_lines and n_lines >= max_lines:
                break
            content = _json.loads(line).get(content_key)
            if not content:
                n_lines += 1
                continue
            ids = tokenizer(content)["input_ids"]
            if eos_id is not None:
                ids = list(ids) + [eos_id]
            buf.extend(ids)
            n_tokens += len(ids)
            n_lines += 1
            if len(buf) >= (1 << 20):
                _dump_uints(writer, buf, dtype)
                buf.clear()
                n_chunks += 1
        if buf:
            _dump_uints(writer, buf, dtype)
    meta = {
        "dtype": dtype, "lines": n_lines, "tokens": n_tokens,
        "vocab_size": vocab_size, "eos_id": eos_id,
        "content_key": content_key, "tokenizer": getattr(tokenizer, "name_or_path", ""),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        _json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[data] lines={n_lines} tokens={n_tokens} dtype={dtype} -> {output_path}")
    print(f"[data] meta -> {out_meta}")
    return n_lines, n_tokens, dtype


def _dump_uints(writer, buf, dtype):
    import numpy as _np
    writer.write(_np.asarray(buf, dtype=dtype).tobytes())


def load_bin_meta(bin_path, meta_path=""):
    meta_file = _resolve_meta(bin_path, meta_path)
    if os.path.exists(meta_file):
        with open(meta_file, encoding="utf-8") as f:
            return _json.load(f), meta_file
    return None, meta_file


class TokenBinDataset(Dataset):
    """窗口化 token 流数据集（uint16/uint32 自适应）。

    - 读取 .meta.json 判断 dtype；无 meta 时按 uint16 兼容旧产物。
    - 返回 (input, target)，长度为 max_len-1 / 由窗口滑出（与 PretrainBinaryDataset 一致：
      total_tokens//max_len 行，每行 [0,max_len)，输入取 [0,max_len)，目标取 [1,max_len])。
    """

    def __init__(self, bin_path, max_len, meta_path=""):
        import numpy as _np
        self.max_len = max_len
        self.meta, self.meta_file = load_bin_meta(bin_path, meta_path)
        self.dtype = (self.meta or {}).get("dtype", "uint16")
        arr = _np.fromfile(bin_path, dtype=self.dtype)
        rows = arr.size // max_len
        self.data = arr[: rows * max_len].reshape(rows, max_len)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        item = self.data[index]
        input_ids = item[:-1].astype(np.int64)
        target_ids = item[1:].astype(np.int64)
        return torch.from_numpy(input_ids), torch.from_numpy(target_ids)


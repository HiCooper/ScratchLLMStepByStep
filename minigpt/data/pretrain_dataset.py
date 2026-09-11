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
    """【教学用】按大小分块读取 jsonl 文本（notebook 08 演示；生产走 tokenize_jsonl_to_bin）。"""
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
    """【教学用】把文本在内存里切成窗口（notebook 08 演示）。

    生产训练不要用它：需要先把全部文本 tokenize 进内存，语料一大就 OOM；
    生产走 `tokenize_jsonl_to_bin` + `TokenBinDataset`（memmap）。
    """
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

def texts_to_bin(input_path, output_path, tokenizer, content_key="content"):
    """【教学用·已过时】把 jsonl 拼成 uint16 token 流（notebook 08 演示）。

    生产请用 `tokenize_jsonl_to_bin`：它按词表自动选 uint16/uint32、写 `.meta.json`、
    只追加 eos 而不强拼 BOS 字符串。本函数硬编码 uint16，词表 >65535 会溢出。
    """
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
    """【教学/链路验证用】按窗口切分的二进制数据集（uint16/uint32 自适应）。

    真实事故：旧实现把 dtype 硬编码成 uint16 且不看 `.meta.json`——词表 >65535 的
    uint32 产物会被读成"总 token 数翻倍、窗口全错位"的数据，`max_tokens` 大于总 token
    时还会静默得到空数据集（len=0）后在别处报出难懂的错。现在：
      - 读 meta 判断 dtype（与生产用的 `TokenBinDataset` 同一口径）；
      - 文件字节数不是 itemsize 整数倍、或切不出一个完整窗口时，直接给出可读报错。
    生产训练请用 `TokenBinDataset`（memmap，不整份读入内存）。
    """

    def __init__(self, data_path, max_tokens, meta_path=""):
        self.data_path = data_path
        self.max_tokens = int(max_tokens)
        self.meta, self.meta_file = load_bin_meta(data_path, meta_path)
        self.dtype = (self.meta or {}).get("dtype", "uint16")
        if self.meta is None:
            print(f"[data] 注意：{data_path} 没有 .meta.json，按 {self.dtype} 读取；"
                  f"词表 >65535 的产物必须显式提供 meta（否则会读错）")
        itemsize = np.dtype(self.dtype).itemsize
        with open(data_path, "rb") as f:      # 必须以 rb 打开，文本模式下 tell 不可靠
            f.seek(0, 2)
            nbytes = f.tell()
        if nbytes % itemsize:
            raise ValueError(
                f"{data_path} 大小 {nbytes}B 不是 {self.dtype}({itemsize}B) 的整数倍："
                f"dtype 与文件不匹配（meta.dtype={(self.meta or {}).get('dtype')!r}）")
        self.total_tokens = nbytes // itemsize
        rows = self.total_tokens // self.max_tokens
        if rows == 0:
            raise ValueError(
                f"{data_path} 只有 {self.total_tokens} 个 token，不足一个窗口 "
                f"(max_tokens={self.max_tokens})：请换更小的 --context-length 或更大的语料")
        self.data = np.memmap(data_path, dtype=self.dtype, shape=(rows, self.max_tokens))
        print(f"total_tokens: {self.total_tokens} (dtype={self.dtype}, rows={rows})")

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
        target = item[1:].astype(np.int64)  # 交叉熵要求目标为长整型
        return torch.from_numpy(input), torch.from_numpy(target)

    def _get_list_items(self, indexes):
        assert isinstance(indexes, list)
        items = [self.data[index] for index in indexes]
        inputs = [item[:-1] for item in items]
        targets = [item[1:] for item in items]
        return (torch.tensor(inputs, dtype=torch.int64),
                torch.tensor(targets, dtype=torch.int64))

    def _get_slice_items(self, index):
        # slice.indices 返回 (start, stop, step)，range 得到正常索引迭代器
        return Subset(self, range(*index.indices(len(self))))


def split_train_eval_random(data, train_ratio):
    """【随机窗口切分，非生产口径】按比例随机切成 (train, eval)。

    ⚠️ 与 `sft_dataset.split_dataset`（3 段、按样本随机）语义不同，故改名消歧义。
    生产预训练请用 `split_train_eval_blocks`：随机切会让同一文档的前半进训练集、
    后半进验证集（文档中位仅 ~151 token，窗口 512 平均横跨 2.6 篇），eval_loss 偏乐观。
    本函数保留给 notebook 09/10/11 与 validate_*.py 的教学/链路验证使用。
    """
    train_len = int(len(data) * train_ratio)
    eval_len = len(data) - train_len
    return random_split(data, [train_len, eval_len])


# 兼容别名（notebook 通过 %run 使用旧名；validate_*.py 也按此名导入）
split_dataset = split_train_eval_random

def create_dataloaders(ds, batch_size, train_ratio, local_rank=-1):
    """【教学用】构造 train/eval DataLoader（notebook 10 演示；生产在 Trainer 内部构造）。"""
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
    - 用 `np.memmap` 惰性映射，不把整个 bin 读进内存（v3 全量 uint16 约 495MB，
      uint32 约 1GB；旧实现用 np.fromfile，每个 DataLoader worker 各持一份，
      域外语料放大后极易 OOM）。
    - 返回 (input, target)，长度为 max_len-1：total_tokens//max_len 行，
      每行 [0,max_len)，输入取 [0,max_len)，目标取 [1,max_len)。
    """

    def __init__(self, bin_path, max_len, meta_path=""):
        import numpy as _np
        self.bin_path = bin_path
        self.max_len = max_len
        self.meta, self.meta_file = load_bin_meta(bin_path, meta_path)
        self.dtype = (self.meta or {}).get("dtype", "uint16")
        self.flat = _np.memmap(bin_path, dtype=self.dtype, mode="r")
        self.total_tokens = int(self.flat.size)
        rows = self.total_tokens // max_len
        self.data = self.flat[: rows * max_len].reshape(rows, max_len)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        item = self.data[index]
        input_ids = item[:-1].astype(np.int64)
        target_ids = item[1:].astype(np.int64)
        return torch.from_numpy(input_ids), torch.from_numpy(target_ids)


def validate_bin_tokenizer(bin_path, tokenizer, meta_path="") -> dict:
    """校验 .bin 与 tokenizer 是否配套（词表大小必须一致）。

    真实隐患：`.meta.json` 里已经写了 vocab_size，但训练入口以前只按 `len(tokenizer)`
    建模型，不比对 meta——用默认的 qwen2 词表(151643) 去训 v3 的 bin(32000) 会静默
    错配（embedding 白建 5 倍参数）。加载侧 `checkpoint.py` 已做严格校验，训练侧补齐。
    """
    meta, meta_file = load_bin_meta(bin_path, meta_path)
    tok_vocab = len(tokenizer)
    if meta is None:
        return {"checked": False, "reason": f"未找到 {meta_file}", "tokenizer_vocab": tok_vocab}
    bin_vocab = meta.get("vocab_size")
    if bin_vocab is not None and int(bin_vocab) != tok_vocab:
        raise ValueError(
            f"分词器与 .bin 不配套：{bin_path} 的 meta.vocab_size={bin_vocab}，"
            f"而 tokenizer 词表={tok_vocab}（meta 记录的分词器：{meta.get('tokenizer')!r}）。\n"
            f"请确认 --data_tokenizer_dir 与构建该 bin 时使用的一致。")
    return {"checked": True, "bin_vocab": bin_vocab, "tokenizer_vocab": tok_vocab,
            "bin_tokens": meta.get("tokens"), "meta_file": meta_file}


def _scan_back_to_doc_start(ds, token_pos, eos_id, max_scan=65536):
    """从 token_pos 向前找到最近的 eos，返回其后一个位置（即文档起点）。"""
    if eos_id is None or token_pos <= 0:
        return int(token_pos)
    import numpy as _np
    lo = max(0, int(token_pos) - max_scan)
    seg = _np.asarray(ds.flat[lo:int(token_pos)])
    hits = _np.nonzero(seg == eos_id)[0]
    if hits.size == 0:
        return int(token_pos)
    return lo + int(hits[-1]) + 1


def _scan_forward_to_doc_end(ds, token_pos, eos_id, max_scan=65536):
    """从 token_pos 向后找到最近的 eos，返回其后一个位置（即文档结束边界）。"""
    if eos_id is None:
        return int(token_pos)
    import numpy as _np
    hi = min(int(ds.flat.size), int(token_pos) + max_scan)
    if token_pos >= hi:
        return int(token_pos)
    seg = _np.asarray(ds.flat[int(token_pos):hi])
    hits = _np.nonzero(seg == eos_id)[0]
    if hits.size == 0:
        return int(token_pos)
    return int(token_pos) + int(hits[0]) + 1


def split_train_eval_blocks(ds, eval_ratio=0.002, max_eval=512, eos_id=None, n_blocks=32):
    """把验证集切成 n_blocks 个**均匀分布**的块，每块双端对齐文档边界。

    为什么不能随机抽窗口（旧实现 random_split）：文档中位长度只有 ~151 token，
    而窗口 512，一个窗口平均横跨 ~2.6 篇文档。窗口边界几乎必然切断某篇文档，
    随机切就会让"文档前半在训练集、后半在验证集"。实测该切分下有相当比例的验证
    token 属于训练时见过的文档，eval_loss 偏乐观。

    为什么也不能只取尾部连续块：本语料 `pretrain_t2t_mini.jsonl` 是**按领域排序**的，
    尾部整段是英文选择题、头部是中文创作/指令。实测同一 checkpoint
      head 前 64 窗 ppl=23.40  vs  尾部 64 窗 ppl=6.78
    取尾部会把"验证集"变成单一领域，测出来的不是整体质量，而且与 trainer 内部
    随机切分的 eval_loss（同一 ckpt 为 16.8）完全不是一个口径。

    做法：每块先向前吸附到文档起点、再向后吸附到文档结束边界，取 token 区间
    [doc_start, doc_end)。训练集排除所有**与该区间重叠**的窗口（保证没有文档被
    train/eval 劈开），验证集只取**完全落在区间内**的窗口。

    返回 (train_subset, eval_subset, info)。
    """
    import numpy as _np
    from torch.utils.data import Subset
    n_rows = len(ds)
    L = ds.max_len
    if n_rows < 4 * n_blocks:
        raise ValueError(f"数据集窗口数过少({n_rows})，无法切出 {n_blocks} 个验证块")
    n_eval = max(n_blocks, min(int(n_rows * eval_ratio), int(max_eval), n_rows // 4))
    per_block = max(1, n_eval // n_blocks)
    stride = n_rows // n_blocks

    val_tok_ranges, blocks = [], []
    for b in range(n_blocks):
        nominal_tok = b * stride * L
        doc_start = _scan_back_to_doc_start(ds, nominal_tok, eos_id)
        doc_end = _scan_forward_to_doc_end(ds, doc_start + per_block * L, eos_id)
        if doc_end <= doc_start:
            doc_end = doc_start + per_block * L
        val_tok_ranges.append((doc_start, doc_end))
        blocks.append({"block": b, "doc_start_token": doc_start, "doc_end_token": doc_end})

    excluded = _np.zeros(n_rows, dtype=bool)   # 与验证区间重叠的窗口（训练集要排除）
    val_mask = _np.zeros(n_rows, dtype=bool)   # 完全落在验证区间内的窗口
    for s, e in val_tok_ranges:
        w_lo, w_hi = s // L, min(n_rows, -(-e // L))
        excluded[w_lo:w_hi] = True
        v_lo, v_hi = -(-s // L), e // L
        if v_hi > v_lo:
            val_mask[v_lo:v_hi] = True
            excluded[v_lo:v_hi] = True
    train_rows = int((~excluded).sum())
    eval_rows = int(val_mask.sum())
    if train_rows < 1 or eval_rows < 1:
        raise ValueError(f"切分失败：train={train_rows} eval={eval_rows}（n_rows={n_rows}）")

    info = {
        "split": "blocked-document-aligned",
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "excluded_rows": int(excluded.sum()),
        "n_blocks": n_blocks,
        "windows_per_block": per_block,
        "requested_eval_ratio": eval_ratio,
        "actual_eval_ratio": round(eval_rows / n_rows, 6),
        "eos_id": eos_id,
        "blocks": blocks,
    }
    return (Subset(ds, _np.nonzero(~excluded)[0].tolist()),
            Subset(ds, _np.nonzero(val_mask)[0].tolist()),
            info)



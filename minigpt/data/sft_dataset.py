import json
import time
import torch
import linecache
from functools import partial
from torch.utils.data import Dataset, DataLoader

# Qwen2 系 chat template 的 assistant 起始标记（可用 create_batch_collator 的
# assistant_marker 参数覆盖；换模板时应当显式改，而不是静默退化）
DEFAULT_ASSISTANT_MARKER = "<|im_start|>assistant\n"
DEFAULT_TURN_END_MARKER = "<|im_end|>"

class InstructionDataset(Dataset):
    def __init__(self, jsonl_file_path, tokenizer, max_len=1024, max_lines=0):
        self.jsonl_file_path = jsonl_file_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        start_time = time.time()
        # 无论是否指定 max_lines，都必须回夹到文件真实行数：
        # 否则 max_lines 偏大时 __getitem__ 会去读不存在的行（实测抛 JSONDecodeError）。
        # 指定 max_lines 时只扫描前 max_lines 行，避免大文件全量计数。
        with open(self.jsonl_file_path, 'r', encoding='utf-8') as f:
            if max_lines and max_lines > 0:
                n = 0
                for n, _ in enumerate(f, start=1):
                    if n >= max_lines:
                        break
            else:
                n = sum(1 for _ in f)
        self.total_lines = min(max_lines, n) if (max_lines and max_lines > 0) else n
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

        # 超长时的截断策略：SFT 的 loss 只落在 assistant 回复上，**右截断会把整段回复
        # 连同 <|im_start|>assistant 标记一起切掉**，得到一条毫无监督信号的样本
        # （真实数据里长 prompt 很容易触发，冒烟即命中）。因此：
        #   1) 优先截掉前面的 prompt，保留最后一轮回复（含标记）完整；
        #   2) 若连回复本身都超过 max_len，则退化为"监督尾部"（等价于续写目标），
        #      此时标记确实会丢，用 supervise_all=True 告知 collate。
        # 返回 (input_ids, supervise_all)。
        if len(input_ids) <= self.max_len:
            return input_ids, False
        spans = assistant_spans(input_ids, self.tokenizer)
        if spans:
            marker_len = len(self.tokenizer(DEFAULT_ASSISTANT_MARKER)["input_ids"])
            turn_ids = input_ids[spans[-1][0] - marker_len:]
            if len(turn_ids) <= self.max_len:
                return input_ids[: self.max_len - len(turn_ids)] + turn_ids, False
        return input_ids[-self.max_len:], True

    def __getitem__(self, idx):
        indexes = [idx] if isinstance(idx, int) else idx
        assert isinstance(indexes, list)
    
        # 使用 linecache 读取指定行
        datas = []
        for i in indexes:
            line = linecache.getline(self.jsonl_file_path, i + 1).strip()
            if not line:
                raise ValueError(
                    f"{self.jsonl_file_path} 第 {i + 1} 行为空或不存在（数据集共 {self.total_lines} 行）；"
                    f"请检查 jsonl 是否存在空行/尾部多余换行")
            datas.append(json.loads(line))
        inputs = [self.process(data) for data in datas]
        return inputs[0] if isinstance(idx, int) else inputs

def find_sublist_index(main_list, sub_list):
    for i in range(len(main_list) - len(sub_list) + 1):
        if main_list[i:i + len(sub_list)] == sub_list:
            return i
    return -1


def find_all_sublist_index(main_list, sub_list):
    """返回所有匹配起点（多轮对话需要定位每一处 assistant 标记）。"""
    if not sub_list:
        return []
    out, m = [], len(sub_list)
    for i in range(len(main_list) - m + 1):
        if main_list[i:i + m] == sub_list:
            out.append(i)
    return out


def assistant_spans(input_ids, tokenizer,
                    assistant_marker=DEFAULT_ASSISTANT_MARKER,
                    turn_end_marker=DEFAULT_TURN_END_MARKER):
    """定位所有"assistant 回复"区间，返回 [(start, end)]（半开区间，基于 input_ids 下标）。

    区间含义：从 assistant 内容的首 token 起，到该轮的结束标记（含 `<|im_end|>`）为止。
    多轮 history 会得到多个区间——这正是"只监督模型该说的话"所必需的。

    真实事故：旧实现只找**第一个** assistant 标记，于是多轮样本里前面所有 history
    （包括用户提问）全部参与 loss（实测一条 3 轮样本 55 token 里 41 个被监督），
    模型被训练去生成用户提问，指令跟随能力退化。
    """
    marker = tokenizer(assistant_marker)["input_ids"]
    end_marker = tokenizer(turn_end_marker)["input_ids"]
    spans = []
    for start in find_all_sublist_index(input_ids, marker):
        content_start = start + len(marker)
        end = len(input_ids)
        if end_marker:
            for j in range(content_start, len(input_ids) - len(end_marker) + 1):
                if input_ids[j:j + len(end_marker)] == end_marker:
                    end = j + len(end_marker)
                    break
        if content_start < end:
            spans.append((content_start, end))
    return spans


def calc_label(input_ids, pad_token_id=None, tokenizer=None,
               assistant_marker=DEFAULT_ASSISTANT_MARKER,
               turn_end_marker=DEFAULT_TURN_END_MARKER):
    """构造 labels：只监督 assistant 回复区间，其余为 -100（交叉熵忽略）。

    与旧实现的区别：
    1. 逐轮定位 assistant 区间，多轮 history 的用户/旧回复不再进 loss；
    2. **不再按 `token == pad_token_id` 屏蔽**——那会把真实的 eos 一并屏蔽掉。
       qwen2 词表下 pad==unk==eos(151643)，旧实现让模型永远学不会停止；
    3. padding 位置由 `collate` 按真实长度补 -100，与 token id 无关。
    """
    assert tokenizer is not None, "calc_label 需要 tokenizer 来定位 assistant 标记"
    n = len(input_ids)
    supervised = set()
    for s, e in assistant_spans(input_ids, tokenizer, assistant_marker, turn_end_marker):
        supervised.update(range(s, e))
    labels = []
    for i in range(n):
        pos = i + 1                      # labels[i] 对应 input_ids[i+1]
        labels.append(input_ids[pos] if (pos < n and pos in supervised) else -100)
    return labels, bool(supervised)


def collate(batch_inputs, pad_token_id, tokenizer, device='cpu',
            assistant_marker=DEFAULT_ASSISTANT_MARKER,
            turn_end_marker=DEFAULT_TURN_END_MARKER):
    assert isinstance(pad_token_id, int)
    # 数据集可能返回 (input_ids, supervise_all) —— supervise_all=True 表示
    # 回复本身超过 max_len、assistant 标记已被截掉，此时监督整段尾部（续写目标）
    items, supervise_all = [], []
    for it in batch_inputs:
        if isinstance(it, tuple):
            items.append(it[0])
            supervise_all.append(bool(it[1]))
        else:
            items.append(it)
            supervise_all.append(False)

    lengths = [len(item) for item in items]
    max_length = max(lengths)
    batch_padded = [item + [pad_token_id] * (max_length - len(item)) for item in items]
    input_tensors = torch.tensor(batch_padded, dtype=torch.int64).to(device)

    # attention_mask 按**真实长度**构造，而不是按 `token == pad_token_id`：
    # 后者在 pad_token_id 恰好等于真实特殊 token（qwen2 下 pad==eos）时会把真实
    # 的 <|im_end|> 也屏蔽掉。
    ar = torch.arange(max_length).unsqueeze(0)
    attention_mask = (ar < torch.tensor(lengths).unsqueeze(1)).to(torch.int64).to(device)

    batch_targets, n_no_assistant = [], 0
    for item, length, sup_all in zip(items, lengths, supervise_all):
        if sup_all:
            # 续写目标：除最后一个位置（没有下一个 token）外全部监督
            labels = [item[i + 1] if i + 1 < length else -100 for i in range(length)]
            found = True
        else:
            labels, found = calc_label(item, pad_token_id, tokenizer,
                                       assistant_marker, turn_end_marker)
        if not found:
            n_no_assistant += 1
        # 用 -100 补齐到 max_length（padding 永不参与 loss）
        labels = labels + [-100] * (max_length - length)
        batch_targets.append(labels)
    if n_no_assistant:
        # 显式失败：模板不一致是会静默毁掉整个 SFT 的系统性错误，不能退化处理。
        # 正常情况下 apply_chat_template 一定会渲染出 assistant 标记，且数据集已做
        # prompt-aware 截断保留该标记；标记彻底消失只可能是 chat template 不匹配。
        raise RuntimeError(
            f"批次中有 {n_no_assistant}/{len(items)} 条样本找不到 assistant 标记 "
            f"{assistant_marker!r}。这通常意味着 chat template 与 assistant_marker 不匹配"
            f"（请确认 tokenizer 是 Qwen2 系模板，或显式传入 assistant_marker）。\n"
            f"样例 token 文本：{tokenizer.decode(items[0][:60])!r}")
    target_tensors = torch.tensor(batch_targets, dtype=torch.int64).to(device)
    return input_tensors, target_tensors, attention_mask
    target_tensors = torch.tensor(batch_targets, dtype=torch.int64).to(device)
    return input_tensors, target_tensors, attention_mask

def resolve_stop_token_ids(tokenizer,
                           turn_end_marker=DEFAULT_TURN_END_MARKER) -> list:
    """生成时应当停止的 token id 列表。

    真实事故：qwen2 词表下 `<|im_end|>`=151645 而 `eos_token_id`=151643，SFT 数据里
    模型学到的回合结束符是前者。若生成时只传 `tokenizer.eos_token_id`，模型已经学会
    输出 `<|im_end|>` 却永远不触发停止条件，只能靠 max_new_tokens 截断。
    因此优先返回 turn_end_marker 的 id，并保留 eos 作为兜底。
    """
    ids = []
    try:
        mark_id = tokenizer.convert_tokens_to_ids(turn_end_marker)
    except Exception:  # noqa: BLE001
        mark_id = None
    unk = getattr(tokenizer, "unk_token_id", None)
    if mark_id is not None and mark_id >= 0 and mark_id != unk:
        ids.append(int(mark_id))
    if tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) not in ids:
        ids.append(int(tokenizer.eos_token_id))
    return ids


def split_train_eval_test(data, train_ratio, eval_ratio):
    """按样本随机切成 (train, eval, test) 三段。

    ⚠️ 与 `pretrain_dataset.split_dataset`（2 段、按窗口）语义不同，故改名消歧义。
    `test` 段当前未被 `sft_trainer` 使用（只用了 train/eval），保留以兼容 notebook 13/14。
    """
    train_len = int(len(data) * train_ratio)
    eval_len = int(len(data) * eval_ratio)
    test_len = len(data) - train_len - eval_len
    return torch.utils.data.random_split(data, [train_len, eval_len, test_len])


# 兼容别名（notebook 13/14 与 sft_trainer 按旧名导入）
split_dataset = split_train_eval_test

def create_batch_collator(tokenizer, assistant_marker=DEFAULT_ASSISTANT_MARKER,
                          turn_end_marker=DEFAULT_TURN_END_MARKER):
    """构造 collator。

    pad 使用 tokenizer 自己声明的 pad_token_id（缺省回落到 eos 并给出提示），
    不再用 unk_token_id——那会让 padding 落在 `<unk>` 上，且 qwen2 下 unk==eos
    导致真实 eos 被当作 padding 屏蔽。
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
        print(f"[sft] tokenizer 未定义 pad_token，回退用 eos({pad_id}) 作 padding；"
              f"padding 位置由真实长度标记，不会影响 loss")
    if pad_id is None:
        raise ValueError("tokenizer 既没有 pad_token_id 也没有 eos_token_id，无法构造 collator")
    return partial(
        collate,
        pad_token_id=pad_id,
        tokenizer=tokenizer,
        assistant_marker=assistant_marker,
        turn_end_marker=turn_end_marker,
    )


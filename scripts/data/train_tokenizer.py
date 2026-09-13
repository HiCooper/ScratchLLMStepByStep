"""训练 BPE 分词器（对应 notebook 01_分词器训练.ipynb）。

把 notebook 里的训练逻辑抽成可复用脚本，便于命令行复现：
    python scripts/data/train_tokenizer.py \
        --data dataset/pretrain_t2t.jsonl \
        --output models/tokenizer_v3 \
        --vocab-size 32000

数据格式：jsonl，每行 {"text": "..."}（与 notebook 01 的 read_texts_from_jsonl 一致）。
"""
import argparse
import json
import os

from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

# 特殊 token：填充/开始/结束标记（与语言模型约定一致）
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def read_texts_from_jsonl(file_path, max_lines=0):
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            data = json.loads(line)
            yield data["text"]


def build_tokenizer_config(vocab_size: int) -> dict:
    """构造 tokenizer_config.json，使 AutoTokenizer.from_pretrained 能正确加载。"""
    added = {
        str(i): {
            "content": tok,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
        for i, tok in enumerate(SPECIAL_TOKENS)
    }
    return {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "added_tokens_decoder": added,
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": None,
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        "use_default_system_prompt": False,
        "chat_template": (
            "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}"
            "{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}"
            "{% for message in messages %}{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}"
            "{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
        ),
    }


def train_tokenizer(data_path: str, output_dir: str, vocab_size: int, max_lines: int = 0):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # 分词器训练只需有代表性的语料子集即可，无需全量语料（尤其字节级 BPE 对长中文段落内存开销大）
    texts = read_texts_from_jsonl(data_path, max_lines=max_lines)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 编码用 ByteLevel，解码也必须用 ByteLevel，否则中文等多语言字符无法正常还原
    tokenizer.decoder = decoders.ByteLevel()

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    tokenizer.model.save(output_dir)

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(build_tokenizer_config(vocab_size), f, ensure_ascii=False, indent=4)

    print(f"Tokenizer training completed and saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="训练 BPE 分词器")
    parser.add_argument("--data", required=True, help="jsonl 语料路径（每行 {\"text\": \"...\"}）")
    parser.add_argument("--output", required=True, help="分词器输出目录")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-lines", type=int, default=0, help="只读取前 N 行（0=全量）；长中文段落建议用子集避免内存爆炸")
    args = parser.parse_args()

    train_tokenizer(args.data, args.output, args.vocab_size, args.max_lines)


if __name__ == "__main__":
    main()

import os
import torch
import time
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from config import paths, TrainConfig
from pretrain_dataset import PretrainBinaryDataset, split_dataset
from transformer import GPTConfig, MiniGPT
from trainer import Trainer

def main():
    tc = TrainConfig()

    # 模型分布式, autocast 会自动将 float32 转为 float16；使用 bf16 请见 trainer 的 dtype 配置
    config = GPTConfig(flash_attn=False)
    model = MiniGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay)

    # 数据加载要设置分布式采样器
    ds = PretrainBinaryDataset(paths.pretrain_dataset, config.context_length)
    train_set, eval_set = split_dataset(ds[:], tc.train_ratio)

    train_args = {
        "train_batch_size": tc.batch_size,
        "eval_strategy": "step",
        "eval_steps": 1000,
        "warmup_steps": 1000,
        "save_strategy": "step",
        "save_steps": 30000,
        "num_train_epochs": tc.epochs,
        "output_dir": paths.output_dir,
        "last_checkpoint_path": paths.last_checkpoint_path,
        "use_mixed_precision": True,
    }
    start_time = time.time()
    trainer = Trainer(model, optimizer, train_args, verbose=True)
    trainer.set_seed(123)
    trainer.set_dataset(train_set, eval_set)
    trainer.train()
    print(f"train use time: {(time.time()-start_time)/60:.2f}min") if trainer.verbose else None
    test_generate_text(trainer)


def test_generate_text(trainer):
    if not trainer.is_main_process:
        print(f"{trainer.device}:return for not main process.")
        return
    # input_text = "小丽: 你好，我是文毅斌，很高兴认识你。\n小美: 你好"
    input_text = "库里在第三节上篮时被防守球员犯规，但裁判并未理会"
    tokenizer = AutoTokenizer.from_pretrained(paths.tokenizer_dir, use_fast=False)
    generated_text = trainer.predict(tokenizer, input_text, 100)
    print(f"{trainer.device}:generate text test result:{generated_text}")

# I/O
if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    # generate()
    main()


"""单卡版训练器（**教学用**，对应 notebook 10「运算加速」/ 11「多卡并行」）。

⚠️ 本模块的 `Trainer`（正式名 `SingleCardTrainer`）与 `minigpt/train/trainer.py` 的
`Trainer` **不是同一个类**：这里是教学用的最小实现，没有 AMP/梯度累积/原子 checkpoint/
数据顺序可控/指标钩子等生产特性。**生产入口请一律用 `minigpt.train.trainer.Trainer`**
（`python -m minigpt.train.pretrainer` / `sft_trainer` 已封装）。

为避免与生产类同名造成误用，类已重命名为 `SingleCardTrainer`；同时保留
`Trainer = SingleCardTrainer` 别名 —— notebook 通过 `%run` 本文件后仍按 `Trainer(...)` 调用。
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn.functional as f
from contextlib import nullcontext


class SingleCardTrainer:
    def __init__(self, model, optimizer, train_args:dict, device='cpu', verbose=False):
        self.cur_time = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.verbose = verbose
        self.target_lr = float(optimizer.defaults['lr'])
        self.num_epochs = train_args.get("num_train_epochs", 0)
        self.batch_size = train_args.get("train_batch_size", 8)
        self.eval_steps = train_args.get("eval_steps", 1000)
        self.save_steps = train_args.get("save_steps", 1000)
        self.warmup_steps = train_args.get("warmup_steps", 1000)
        self.output_dir = train_args.get("output_dir")
        self.last_checkpoint_path = train_args.get("last_checkpoint_path")
        self.train_loader = None
        self.eval_loader = None
        self.scaler = None
        self.steps_per_epoch = 0   # 每个epoch的训练步骤数
        self.step = 0              # 正在训练的step编号
        self.total_steps = 0       # 总的步骤数
        self.train_loss_acc = 0    # 训练损失累计

    def set_loader(self, train_loader, eval_loader):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.steps_per_epoch = len(train_loader)
        self.total_steps = self.num_epochs * self.steps_per_epoch
        print(f'set train_loader steps: {len(train_loader)}, eval_loader: {len(eval_loader)}') if self.verbose else None

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  
        print(f"set seed to {seed}") if self.verbose else None

    def set_grad_scaler(self, enabled=True):
        if enabled:
            self.scaler = torch.amp.GradScaler('cuda', enabled=True)
        else:
            self.scaler = None
        print("enable grad scaler: ", enabled)

    @staticmethod
    def get_dynamic_lr(target_lr, cur_step, warmup_steps, decay_steps):
        min_lr = target_lr/10
        if cur_step < warmup_steps:
            return target_lr * (cur_step / warmup_steps)
        if cur_step > decay_steps:
            return min_lr
        
        step_ratio = (cur_step - warmup_steps)/(decay_steps-warmup_steps)
        cos_scope = 0.5 * (1 + math.cos(math.pi * step_ratio))
        return min_lr + (target_lr - min_lr) * cos_scope

    def adjust_lr(self):
        lr = self.get_dynamic_lr(self.target_lr, self.step, self.warmup_steps, self.total_steps)
        if lr <= 0: return self.target_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
        
    def train_step(self, X, Y): 
        enable_mixed_precision = X.device.type == "cuda" and self.scaler is not None
        ctx = (torch.amp.autocast('cuda') if enable_mixed_precision else nullcontext())  
        
        self.optimizer.zero_grad(set_to_none=True)  
        with ctx:  
            logits = self.model(X)  
            loss = f.cross_entropy(logits.flatten(0, 1), Y.flatten())
    
        if enable_mixed_precision:  # 检查是否使用混合精度  
            self.scaler.scale(loss).backward()  
            self.scaler.unscale_(self.optimizer)  
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  
            self.scaler.step(self.optimizer)  
            self.scaler.update()  
        else:  
            loss.backward()  # 普通精度的反向传播  
            self.optimizer.step()  # 更新参数  
    
        return loss 

    def evaluate(self):
        self.model.eval()
        num_batches = len(self.eval_loader)
        total_loss = 0
        for (X, Y) in self.eval_loader:
            with torch.no_grad():
                logits = self.model(X.to(self.device))
            loss = f.cross_entropy(logits.flatten(0, 1), Y.to(self.device).flatten())
            total_loss += loss.item()
        self.model.train()
        return total_loss/num_batches  

    def train_epoch(self, epoch):
        assert self.train_loader and self.eval_loader, f"train_loader and eval_loader can't be empty."
        
        for i, (X, Y) in enumerate(self.train_loader):     
            loss = self.train_step(X.to(self.device), Y.to(self.device))
            self.train_loss_acc += loss.item()
            self.step += 1
    
            if self.step % self.eval_steps == 0:
                train_loss = self.train_loss_acc/self.eval_steps
                eval_loss = self.evaluate()
                print(f"{self.cur_time()} train_loss: {train_loss:.4f}, eval_loss: {eval_loss:.4f}, "
                    + f"step: {self.step}/{self.total_steps}")
                self.train_loss_acc = 0

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)
        
    
    def save_model(self, checkpoint_path, epoch):
        local_model, optimizer, step = self.model, self.optimizer, self.step
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            "model_state": local_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        }, checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path):
        local_model, optimizer, device = self.model, self.optimizer, self.device
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        local_model.load_state_dict(checkpoint['model_state'])
        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        self.step = checkpoint.get('step', 0)
        return checkpoint.get('epoch', 0), 
   
    def predict(self, tokenizer, input_text, max_length=100):
        inputs = torch.tensor([tokenizer.encode(input_text)]).to(self.device)
        response_ids = self.model.generate(inputs, max_length=max_length, eos_token_id=tokenizer.eos_token_id, use_kv_cache=False)
        return tokenizer.decode(response_ids.squeeze(0))


# 向后兼容别名：notebook 10/11 通过 `%run minigpt/train/pretrainer_single.py` 引入本文件后
# 直接按 `Trainer(...)` 调用。新代码请显式使用 `SingleCardTrainer`，生产用
# `minigpt.train.trainer.Trainer`。
Trainer = SingleCardTrainer

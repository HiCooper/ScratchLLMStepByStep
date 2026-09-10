import os
import math
import random as _random
import time
import numpy as np
import torch
import torch.nn.functional as f
from contextlib import nullcontext
import torch.distributed as dist
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.distributed import init_process_group, destroy_process_group
from minigpt.model.transformer import GPTConfig, MiniGPT


class Trainer:
    def __init__(self, model, optimizer, train_args:dict, device='cpu', verbose=False):
        self.cur_time = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        self.model = model
        self.optimizer = optimizer
        self.train_args = train_args
        self.device = device
        self.verbose = verbose
        self.target_lr = float(optimizer.defaults['lr'])
        self.num_epochs = train_args.get("num_train_epochs", 0)
        self.batch_size = train_args.get("train_batch_size", 8)
        self.eval_steps = train_args.get("eval_steps", 1000)
        self.save_strategy = train_args.get("save_strategy", "step")
        self.save_steps = train_args.get("save_steps", 10000)
        self.warmup_steps = train_args.get("warmup_steps", 1000)
        self.use_mixed_precision = train_args.get("use_mixed_precision", False)
        # 混合精度类型：float16（需 GradScaler）或 bfloat16（指数位与 fp32 相同，无需 scaling）
        amp_dtype_name = train_args.get("mixed_precision_dtype", "float16")
        self.amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(amp_dtype_name, torch.float16) \
            if self.use_mixed_precision else None
        self.gradient_accumulation_steps = max(1, int(train_args.get("gradient_accumulation_steps", 1)))
        self.micro_step = 0
        self.output_dir = train_args.get("output_dir")
        self.last_checkpoint_path = train_args.get("last_checkpoint_path")
        self.train_set = None
        self.eval_set = None
        self.train_loader = None
        self.eval_loader = None
        self.steps_per_epoch = 0
        self.step = 0
        self.cur_epoch = 0
        self.total_steps = 0
        self.train_loss_acc = 0
        self.last_grad_norm = 0.0
        self.ddp = False
        self.rank = -1
        self.local_rank = -1
        self.is_main_process = True
        self.scaler = None
        self.batch_collator = None
        # ---- 生产化扩展 ----
        self.writer = None                       # tensorboard SummaryWriter（可选）
        self.grad_clip = float(train_args.get("grad_clip", 1.0))
        self.max_updates = int(train_args.get("max_steps", 0) or 0)
        # 增量续训：reset_step 让调度/skip/max_steps 全部相对新 run 起算（见 train()）
        self.reset_step = bool(train_args.get("reset_step", False))
        self.extra_steps = int(train_args.get("extra_steps", 0) or 0)
        self.effective_max = 0                   # train() 中根据 loader 确定
        self.best_eval_loss = float("inf")
        self.best_step = None
        self.save_best = bool(train_args.get("save_best", True))
        self.last_eval_loss = None
        self.final_metrics = {}
        self.extra_ckpt = None   # 附加到每个 checkpoint 的字典（如 config）
        self.torch_compile = bool(train_args.get("torch_compile", False))
        self.num_workers = int(train_args.get("num_workers", 0))
        self.compile_mode = train_args.get("compile_mode", "default")
        self.metrics = None      # 可选的 MetricsLogger（tensorboard 直方图/图像/投影/模型图）
        self.tokenizer = None    # 可选：供 metrics 生成样本文本与嵌入 metadata

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  
        print(f"set seed to {seed}") if self.verbose else None

    def set_dataset(self, train_set, eval_set, batch_collator=None):
        self.train_set = train_set
        self.eval_set = eval_set
        self.batch_collator = batch_collator
        print(f'set trainset: {len(train_set)}, evalset: {len(eval_set)}') if self.verbose else None
    


    def _init_dataloader(self):
        assert self.train_set and self.eval_set, f"train_set and eval_set can't be empty."
        train_set, eval_set, batch_size = self.train_set, self.eval_set, self.batch_size
        sampler = DistributedSampler(train_set) if self.ddp else None
        self.train_loader = DataLoader(train_set, 
                                       batch_size=batch_size, 
                                       shuffle=(sampler==None), 
                                       num_workers=self.num_workers, 
                                       drop_last=True, 
                                       collate_fn=self.batch_collator,
                                       sampler=sampler)
        self.eval_loader = DataLoader(eval_set, 
                                      batch_size=batch_size, 
                                      shuffle=True, 
                                      num_workers=self.num_workers, 
                                      drop_last=False,
                                      collate_fn=self.batch_collator)
        self.steps_per_epoch = len(self.train_loader)
        # 梯度累积下，优化器更新次数 = 微批次数 / 累积步数（用于 LR 调度与总步数）
        self.updates_per_epoch = self.steps_per_epoch // self.gradient_accumulation_steps
        self.total_steps = self.num_epochs * self.updates_per_epoch
        print(f'init train_loader steps: {len(self.train_loader)}, eval_loader: {len(self.eval_loader)}') if self.verbose else None

    def _unwrap(self):
        """取出真实模型：兼容 DistributedDataParallel 与 torch.compile(OptimizedModule)。"""
        model = self.model
        model = model.module if isinstance(model, DistributedDataParallel) else model
        return getattr(model, "_orig_mod", model)

    def _save_model(self, checkpoint_path, epoch):
        model, optimizer, step = self.model, self.optimizer, self.step
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        local_model = self._unwrap()
        payload = {
            "model_state": local_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_eval_loss": self.best_eval_loss,
            "best_step": self.best_step,
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "random": _random.getstate(),
            },
        }
        if self.extra_ckpt is not None:
            payload["config"] = self.extra_ckpt
        torch.save(payload, checkpoint_path)

    def _load_from_checkpoint(self):
        # 在分布式训练的多GPU环境中，map_location可以确保模型的参数和优化器的状态被加载到正确的GPU上，避免出现设备不匹配而报错。
        model, optimizer, device = self.model, self.optimizer, self.device
        local_model = model.module if isinstance(model, DistributedDataParallel) else model
        checkpoint = torch.load(self.last_checkpoint_path, map_location=device, weights_only=False)
        local_model.load_state_dict(checkpoint['model_state'])
        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        self.step = checkpoint.get('step', 0)
        last_epoch = checkpoint.get('epoch', 0)
        # 续训时恢复"历史最优"记录，避免 best.pt 覆盖后指标从 inf 重新起算
        if checkpoint.get("best_eval_loss") is not None:
            self.best_eval_loss = float(checkpoint["best_eval_loss"])
            self.best_step = checkpoint.get("best_step")
        # 恢复混合精度缩放器与随机数状态，保证续训可复现
        if self.scaler is not None and checkpoint.get("scaler_state") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        rng = checkpoint.get("rng_state")
        if rng is not None:
            # 注意：torch.load(map_location=device) 会把 RNG 的 ByteTensor 也搬到 GPU，
            # 而 set_rng_state 要求 CPU ByteTensor，这里统一 .cpu() 修正
            torch_state = rng["torch"]
            if torch.is_tensor(torch_state):
                torch_state = torch_state.cpu()
            torch.set_rng_state(torch_state)
            if rng.get("cuda") and torch.cuda.is_available():
                cuda_states = [st.cpu() if torch.is_tensor(st) else st for st in rng["cuda"]]
                torch.cuda.set_rng_state_all(cuda_states)
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("random") is not None:
                import random as _r
                _r.setstate(tuple(rng["random"]))
        print(f"load from checkpoint: {self.last_checkpoint_path}, last_epoch:{last_epoch}, last_step: {self.step}")
        return last_epoch
    
    def _init_distributed_mode(self):
        rank = int(os.environ.get("RANK", -1))
        if rank == -1: 
            self.is_main_process = True
            return
        
        os.environ['NCCL_DEBUG'] = 'WARN'
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=120))
        
        self.ddp = True
        self.rank = rank
        self.is_main_process = self.rank == 0
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.device = f'cuda:{self.local_rank}'
        self.verbose = self.verbose and self.is_main_process
        torch.cuda.set_device(self.device)
        
    def _cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group() 
            print("clean multi process.")        
        print(f"train over, steps: {self.step}") if self.verbose else None

    @staticmethod
    def _wrap_model_with_ddp(model, local_rank):
        # 位置编码用的是复数，而nccl不支持复数形式，此变量并不要求在多进程中保持一致，所以暂时屏蔽对此变量的同步
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
        print(f"packaged model with DDP in cuda:{local_rank}")
        return model
    
    def _init_grad_scaler(self):
        # bfloat16 指数位与 fp32 相同、数值范围一致，无需 loss scaling；仅 float16 需要 GradScaler
        if self.use_mixed_precision and self.amp_dtype == torch.float16:
            self.scaler = torch.amp.GradScaler('cuda', enabled=True)
        else:
            self.scaler = None
        print(f"init grad scaler: {self.scaler is not None}, amp_dtype: {self.amp_dtype}") if self.verbose else None

    def _calc_grad_norm(self):
        # 计算梯度范数，梯度范数为所有参数平方和的平方根。先为每一层计算梯度范数，再计算所有层合在一起的梯度范数。
        total_norm = 0.0
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _accumulate_training_loss(self, loss):
        if not self.ddp:
            self.train_loss_acc += loss.item()
            return
        
        # 分布式训练下，需要计算所有进程的平均损失来作为训练损失，比单独主进程的损失数据要更平滑
        dist.reduce(loss, dst=0)
        self.train_loss_acc += loss.item()/dist.get_world_size()
    
    def _record_metrics(self, train_loss, eval_loss, grad_norm, lr):
        # 展示用总步数用 effective_max：领域增量续训时 epochs*每epoch步数 会远大于实际目标
        print(f"{self.cur_time()} lr={lr:.5f}, train_loss: {train_loss:.4f}, "
            + f"eval_loss: {eval_loss:.4f}, grad_norm={grad_norm:.5f}, "
            + f"steps: {self.step}/{self.effective_max or self.total_steps}"
        )
        self.last_eval_loss = eval_loss
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_step = self.step
            # 保存"历史最优"权重：小模型训练后期常过拟合，用 best.pt 做下游/评测通常优于 final.pt
            if self.save_best and self.is_main_process and self.output_dir:
                self._save_model(os.path.join(self.output_dir, "best.pt"), self.cur_epoch)
        if self.writer is not None and self.metrics is None:
            self.writer.add_scalar("train/loss", train_loss, self.step)
            self.writer.add_scalar("eval/loss", eval_loss, self.step)
            self.writer.add_scalar("eval/perplexity", math.exp(min(eval_loss, 80.0)), self.step)
            self.writer.add_scalar("train/lr", lr, self.step)
            self.writer.add_scalar("train/grad_norm", grad_norm, self.step)
        if self.metrics is not None:
            self.metrics.on_eval(self.step, train_loss, eval_loss, lr, grad_norm)

    @staticmethod
    def _get_dynamic_lr(target_lr, cur_step, warmup_steps, decay_steps):
        min_lr = target_lr/10
        if cur_step < warmup_steps:
            return target_lr * (cur_step / warmup_steps)
        if cur_step > decay_steps:
            return min_lr
        
        step_ratio = (cur_step - warmup_steps)/(decay_steps-warmup_steps)
        cos_scope = 0.5 * (1 + math.cos(math.pi * step_ratio))
        return min_lr + (target_lr - min_lr) * cos_scope

    def _adjust_lr(self):
        lr = self._get_dynamic_lr(self.target_lr, self.step, self.warmup_steps, self.total_steps )
        if lr <= 0: return self.target_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
            

    def _check_and_evaluate(self, lr):
        if (self.step + 1) % self.eval_steps != 0:
            return

        if self.is_main_process:
            # 梯度累积下，每个 step 之间会累积多个 micro-batch 的 loss，故分母要乘累积步数
            train_loss = self.train_loss_acc / (self.eval_steps * self.gradient_accumulation_steps)
            eval_loss = self._evaluate()
            grad_norm = self.last_grad_norm
            self._record_metrics(train_loss, eval_loss, grad_norm, lr)
            self.train_loss_acc = 0

        dist.barrier() if self.ddp else None

    def _evaluate(self):
        # 这里不能多进程同步，必须用原始Model
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        model.eval()
        num_batches = len(self.eval_loader)
        total_loss = 0
        
        for batch in self.eval_loader:
            X, Y = batch[0].to(self.device), batch[1].to(self.device)
            attnmask = batch[2].to(self.device) if len(batch) == 3 else None
            with torch.no_grad():
                logits = model(X, attention_mask=attnmask)
            loss = f.cross_entropy(logits.flatten(0, 1), Y.flatten())
            total_loss += loss.item()

        model.train()
        return total_loss/num_batches

    def test(self, dataset):
        """用于对训练的模型进行评估测试"""
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        model.eval()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.batch_collator)
        num_batches = len(dataloader)
        total_loss = 0
        
        for batch in dataloader:
            X, Y = batch[0].to(self.device), batch[1].to(self.device)
            attnmask = batch[2].to(self.device) if len(batch) == 3 else None
            with torch.no_grad():
                logits = model(X, attention_mask=attnmask)
            loss = f.cross_entropy(logits.flatten(0, 1), Y.flatten())
            total_loss += loss.item()
        
        model.train()
        return total_loss/num_batches  

    def _check_and_save_checkpoint(self, cur_epoch):
        if self.save_strategy != "step" or self.step % self.save_steps != 0:
            return
        
        if self.is_main_process:
            checkpoint_path = f"{self.output_dir}/checkpoint-{self.step}.pth"
            self._save_model(checkpoint_path, cur_epoch)
            print(f"{self.cur_time()} device:{self.device}-save checkpoint: {checkpoint_path}")
            
        # 设置屏障, 让所有进程等待主进程的checkpoint操作
        dist.barrier() if self.ddp else None  
        print(f"{self.cur_time()} barrier wait over of device:{self.device} at step: {self.step}.")
    
    def _train_step(self, X, Y, attnmask):
        use_amp = X.device.type == "cuda" and self.amp_dtype is not None
        ctx = torch.amp.autocast('cuda', dtype=self.amp_dtype) if use_amp else nullcontext()

        with ctx:
            logits = self.model(X, attention_mask=attnmask)
            loss = f.cross_entropy(logits.flatten(0, 1), Y.flatten())

        # 梯度累积：对 loss 按累积步数缩放，多次 backward 后等价于大 batch 的梯度
        scale = 1.0 / self.gradient_accumulation_steps
        if use_amp and self.scaler is not None:
            self.scaler.scale(loss * scale).backward()
        else:
            (loss * scale).backward()

        self.micro_step += 1
        if self.micro_step < self.gradient_accumulation_steps:
            return loss, False

        # 累积满 gradient_accumulation_steps 次后，执行一次参数更新
        # 梯度范数必须在 optimizer.step / zero_grad 之前计算，否则梯度已被清零、恒为 0
        if use_amp and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            self.last_grad_norm = self._calc_grad_norm()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 梯度裁剪在混合精度与全精度下保持一致，防止梯度爆炸
            self.last_grad_norm = self._calc_grad_norm()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()  # 更新参数
        self.optimizer.zero_grad(set_to_none=True)
        self.micro_step = 0
        return loss, True

    def _train_epoch(self, cur_epoch):
        assert self.train_loader and self.eval_loader, f"train_loader and eval_loader can't be empty."
        self.cur_epoch = cur_epoch
        # 从中断的优化器步位置换算回微批索引继续训练（梯度累积下两者相差 accumulation 倍）
        skip_updates = self.step - cur_epoch * self.updates_per_epoch
        skip_micro = skip_updates * self.gradient_accumulation_steps

        # 每个epoch开始时都重新打乱数据
        self.train_loader.sampler.set_epoch(cur_epoch) if self.ddp else None
        print(f"{self.cur_time()} start epoch:{cur_epoch} from step:{self.step}") if self.verbose else None

        for i, batch in enumerate(self.train_loader):
            if self.effective_max and self.step >= self.effective_max:
                return
            if i < skip_micro: continue
            X, Y = batch[0].to(self.device), batch[1].to(self.device)
            attnmask = batch[2].to(self.device) if len(batch) == 3 else None
            lr = self._adjust_lr()
            train_loss, did_update = self._train_step(X, Y, attnmask)
            self._accumulate_training_loss(train_loss)
            # 只有真正完成一次参数更新才推进 step 计数并触发 eval/save
            if did_update:
                self.step += 1
                if self.metrics is not None:
                    self.metrics.on_train_step(self.step, train_loss.item(), lr,
                                               self.last_grad_norm, batch)
                self._check_and_evaluate(lr)
                self._check_and_save_checkpoint(cur_epoch)
            

    def train(self):
        last_epoch = 0
        # 初始化分布式环境
        self._init_distributed_mode()
        # 初始化数据加载器
        self._init_dataloader()
        # 初始化梯度缩放器
        self._init_grad_scaler()
        # 将模型移动到指定设备上
        self.model.to(self.device)
        # 从指定的checkpoint恢复训练状态
        if self.last_checkpoint_path:
            last_epoch = self._load_from_checkpoint()
        # ---- 增量续训（领域自适应）----
        # 默认沿用 checkpoint 里的绝对步数；reset_step（或 extra_steps，见下）时把步数归零，
        # 否则"从 211000 步的基座再训 26847 步"会被解读成 max_steps=26847 而直接判定已训完，
        # 且 cosine 调度/epoch skip 也会立刻失效（这是踩过的真实坑）。
        if self.reset_step or self.extra_steps > 0:
            print(f"[trainer] 步数归零（reset_step={self.reset_step}, extra_steps={self.extra_steps}）："
                  f"{self.step} -> 0（模型/优化器权重保留）") if self.verbose else None
            self.step = 0
            last_epoch = 0
            self.best_eval_loss = float("inf")
            self.best_step = None
        # 总更新步数：extra_steps 语义 = "从现在起再训 N 步"
        if self.extra_steps > 0:
            self.effective_max = self.step + self.extra_steps
        else:
            self.effective_max = self.max_updates if self.max_updates > 0 else self.total_steps
        # 分布式训练需要使用ddp同步模型状态
        if self.ddp:
            self.model = self._wrap_model_with_ddp(self.model, self.local_rank)
        if self.torch_compile:
            print(f"[trainer] torch.compile(mode={self.compile_mode}) 已启用（固定形状下提速显著）") \
                if self.verbose else None
            self.model = torch.compile(self.model, mode=self.compile_mode)
        self.step = min(self.step, self.effective_max) if self.effective_max else self.step

        for epoch in range(last_epoch, self.num_epochs):
            if self.effective_max and self.step >= self.effective_max:
                break
            self._train_epoch(epoch)

        # 结束：主进程做最终评估并保存 final.pt（含完整 RNG/缩放器状态）
        if self.is_main_process:
            if self.total_steps > 0:
                final_eval = self._evaluate()
                self.final_metrics = {
                    "step": self.step,
                    "epoch": last_epoch + max(0, self.num_epochs - last_epoch - 1)
                             if self.step else last_epoch,
                    "train_loss": self.train_loss_acc / max(1, self.eval_steps),
                    "eval_loss": final_eval,
                    "perplexity": float(math.exp(min(final_eval, 80.0))),
                    "best_eval_loss": self.best_eval_loss,
                    "best_step": self.best_step,
                }
                if self.writer is not None:
                    self.writer.add_scalar("eval/final_loss", final_eval, self.step)
            os.makedirs(self.output_dir, exist_ok=True) if self.output_dir else None
            if self.output_dir:
                self._save_model(os.path.join(self.output_dir, "final.pt"), self.num_epochs - 1)
                print(f"{self.cur_time()} final checkpoint saved: "
                      f"{os.path.join(self.output_dir, 'final.pt')}")
            if self.metrics is not None:
                self.metrics.on_final(self.step, final_eval if self.total_steps > 0 else None)
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
        self._cleanup()

    def set_writer(self, writer):
        """注入 tensorboard SummaryWriter（仅主进程）。"""
        self.writer = writer
    
    def predict(self, tokenizer, input_text, max_length=100):
        inputs = torch.tensor([tokenizer.encode(input_text)]).to(self.device)
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        response_ids = model.generate(inputs, max_length=max_length, eos_token_id=tokenizer.eos_token_id, use_kv_cache=True)
        new_tokens = response_ids[0][inputs.shape[1]:]
        return tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True).strip()


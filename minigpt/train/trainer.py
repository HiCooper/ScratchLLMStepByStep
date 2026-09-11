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
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.distributed import init_process_group, destroy_process_group
from minigpt.model.transformer import GPTConfig, MiniGPT
from minigpt.train import checkpoint_io
from minigpt.train.ddp_utils import ddp_kwargs, distributed_scalar_mean, unwrap_model, wrap_ddp
from minigpt.train.schedule import LossAccumulator, get_dynamic_lr


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
        # 训练 loss 记账（(sum,count) 口径，见 schedule.LossAccumulator）
        self.loss_acc = LossAccumulator()
        self.last_grad_norm = 0.0
        self.ddp = False
        self.seed = int(train_args.get("seed", 123))
        self.ddp_timeout_seconds = int(train_args.get("ddp_timeout_seconds", 1800) or 1800)
        self.train_sampler = None
        self._train_generator = None
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
        self.deterministic_cudnn = bool(train_args.get("deterministic_cudnn", False))
        self.compile_mode = train_args.get("compile_mode", "default")
        self.metrics = None      # 可选的 MetricsLogger（tensorboard 直方图/图像/投影/模型图）
        self.tokenizer = None    # 可选：供 metrics 生成样本文本与嵌入 metadata

    def set_seed(self, seed):
        """设置全局随机种子。

        注意：数据顺序**不依赖**这里的全局 RNG（训练 sampler 用独立 generator，
        按 seed+epoch 播种），因此续训时恢复 RNG 状态不会再影响"跳过哪些样本"。

        `cudnn.deterministic/benchmark` 由 `deterministic_cudnn` 控制：默认关闭以保留吞吐
        （本模型全是 matmul/attention，cudnn benchmark 收益本就有限），需要严格复现时打开。
        """
        self.seed = int(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = bool(self.deterministic_cudnn)
        torch.backends.cudnn.benchmark = not bool(self.deterministic_cudnn)
        print(f"set seed to {seed} (cudnn.deterministic={self.deterministic_cudnn})") \
            if self.verbose else None

    def set_train_epoch(self, epoch):
        """固定本 epoch 的数据顺序。

        - DDP：`DistributedSampler.set_epoch(epoch)`，各 rank 用 (seed, epoch) 派生同一划分；
        - 单卡：把我们自己持有的 generator 重新播种为 `seed + epoch`，
          于是"第 N 个 epoch 的第 i 个 batch"在任何时候（含断点续训）都是同一批数据。
        """
        if self.ddp and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)
        elif self._train_generator is not None:
            self._train_generator.manual_seed(self.seed + int(epoch))

    def set_dataset(self, train_set, eval_set, batch_collator=None):
        self.train_set = train_set
        self.eval_set = eval_set
        self.batch_collator = batch_collator
        print(f'set trainset: {len(train_set)}, evalset: {len(eval_set)}') if self.verbose else None
    


    def _init_dataloader(self):
        assert self.train_set and self.eval_set, f"train_set and eval_set can't be empty."
        train_set, eval_set, batch_size = self.train_set, self.eval_set, self.batch_size
        # 训练顺序完全由**我们自己的 sampler/generator** 决定，不依赖全局 RNG：
        # 旧实现用 shuffle=True，RandomSampler 每次 __iter__ 都从全局 RNG 取种子，而 checkpoint
        # 保存的 RNG 状态位于 epoch 中段 —— 续训恢复后拿到的是**另一个排列**，
        # `skip_micro` 跳过的不再是同一批样本（会造成数据重复/漏训）。
        #
        # 注意 generator 还必须同时传给 DataLoader：`iter(loader)` 会抽一个 `_base_seed`
        # 用于 worker 播种，而这一步**默认走全局 RNG**（与 sampler 类型无关，实测连
        # SequentialSampler 也会消耗）。只有 generator= 指向我们自己的生成器，整条数据管线
        # 才与全局 RNG 彻底解耦，否则 eval 的疏密仍会改变后续训练的随机性。
        self._train_generator = torch.Generator()
        self._train_generator.manual_seed(self.seed)
        self._eval_generator = torch.Generator()
        self._eval_generator.manual_seed(self.seed + 1_000_003)   # 与训练错开，互不干扰
        if self.ddp:
            self.train_sampler = DistributedSampler(train_set)
        else:
            # 固定 generator；每个 epoch 用 seed+epoch 重新播种（见 set_train_epoch）
            self.train_sampler = RandomSampler(train_set, generator=self._train_generator)
        self.train_loader = DataLoader(train_set,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=self.num_workers,
                                       drop_last=True,
                                       collate_fn=self.batch_collator,
                                       sampler=self.train_sampler,
                                       generator=self._train_generator)
        # 验证集必须 shuffle=False：
        #   1) shuffle 会让每次 eval 的样本顺序不同，指标不可复现；
        #   2) 旧实现在这里 shuffle，会**消耗训练用的全局 RNG**，于是 eval 的疏密会改变
        #      后续训练的数据顺序，同一 run 的两次续训结果因此不一致。
        self.eval_loader = DataLoader(eval_set,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=self.num_workers,
                                      drop_last=False,
                                      collate_fn=self.batch_collator,
                                      generator=self._eval_generator)
        self.steps_per_epoch = len(self.train_loader)
        # 梯度累积下，优化器更新次数 = 微批次数 / 累积步数（用于 LR 调度与总步数）
        self.updates_per_epoch = self.steps_per_epoch // self.gradient_accumulation_steps
        if self.num_epochs <= 0 and self.max_updates <= 0:
            # 真实事故：num_train_epochs 缺省为 0 时 train() 静默跑 0 步，而且不报错
            raise ValueError(
                "num_train_epochs<=0 且未指定 max_steps：训练会一步都不跑。"
                "请设置 --train_epochs>=1 或 --train_max_steps>0")
        if self.updates_per_epoch <= 0:
            print(f"[trainer] 警告：每个 epoch 只有 {self.steps_per_epoch} 个 batch，"
                  f"不足 gradient_accumulation_steps={self.gradient_accumulation_steps}，"
                  f"需要靠 max_steps 驱动；请同时设置 --train_max_steps")
        self.total_steps = self.num_epochs * self.updates_per_epoch
        print(f'init train_loader steps: {len(self.train_loader)}, eval_loader: {len(self.eval_loader)}') if self.verbose else None

    def _unwrap(self):
        """取出真实模型（委托给 ddp_utils.unwrap_model，顺序敏感，见其文档）。"""
        return unwrap_model(self.model)

    def _save_model(self, checkpoint_path, epoch):
        checkpoint_io.save_training_checkpoint(
            checkpoint_path, self.model, self.optimizer, epoch=epoch, step=self.step,
            best_eval_loss=self.best_eval_loss, best_step=self.best_step,
            scaler=self.scaler, config=self.extra_ckpt)

    def _load_from_checkpoint(self):
        info = checkpoint_io.load_training_checkpoint(
            self.last_checkpoint_path, self.model, self.optimizer, self.scaler,
            device=self.device, verbose=self.verbose)
        self.step = info["step"]
        if info["best_eval_loss"] is not None:
            self.best_eval_loss = info["best_eval_loss"]
            self.best_step = info["best_step"]
        return info["epoch"]

    def _init_distributed_mode(self):
        rank = int(os.environ.get("RANK", -1))
        if rank == -1: 
            self.is_main_process = True
            return
        
        os.environ['NCCL_DEBUG'] = 'WARN'
        world_size = int(os.environ["WORLD_SIZE"])
        # 超时必须覆盖 rank0 独占的最慢操作（完整 eval + 数百 MB checkpoint 同步落盘，
        # 二者都在 barrier 保护内）。旧值 120s 会在慢盘/大验证集时以 NCCL 超时打挂整轮训练。
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size,
                                timeout=timedelta(seconds=self.ddp_timeout_seconds))
        
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
    def _ddp_kwargs(local_rank=None):
        """DDP 构造参数（委托给 ddp_utils，便于单测）。"""
        return ddp_kwargs(local_rank)

    @staticmethod
    def _wrap_model_with_ddp(model, local_rank):
        return wrap_ddp(model, local_rank)

    def _init_grad_scaler(self):
        # bfloat16 指数位与 fp32 相同、数值范围一致，无需 loss scaling；仅 float16 需要 GradScaler
        if self.use_mixed_precision and self.amp_dtype == torch.float16:
            self.scaler = torch.amp.GradScaler('cuda', enabled=True)
        else:
            self.scaler = None
        print(f"init grad scaler: {self.scaler is not None}, amp_dtype: {self.amp_dtype}") if self.verbose else None

    def _calc_grad_norm(self):
        """全局梯度 L2 范数。

        旧实现逐参数 `p.grad.norm(2).item()`，每个参数一次 host-device 同步
        （~100 个参数即 ~100 次 .item()），在 torch.compile 下还会反复打断图。
        这里用 `torch._foreach_norm` 批量算各参数范数，只在最后做一次同步。
        """
        grads = [p.grad.detach() for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        norms = torch._foreach_norm(grads, 2)
        return float(torch.linalg.vector_norm(torch.stack(norms)))

    # ---- loss 记账（对外暴露的属性名保持稳定，便于测试与看板读取）----
    @property
    def micro_loss_sum(self):
        return self.loss_acc.total

    @property
    def micro_count(self):
        return self.loss_acc.count

    @property
    def last_train_loss(self):
        return self.loss_acc.last_mean

    @property
    def last_train_loss_count(self):
        return self.loss_acc.last_count

    def _accumulate_training_loss(self, loss):
        """累计一个 micro-batch 的 loss（本地累加，跨 rank 平均推迟到需要时）。

        旧实现每个 micro-batch 都做一次 `dist.reduce`（还直接原地写在带梯度的张量上），
        即每步一个集合通信同步点；现在只本地累加，跨卡平均由 `_global_mean_loss()`
        在 eval/结束时一次性完成。
        """
        self.loss_acc.add(loss)

    def _global_mean_loss(self):
        """跨 rank 的 micro-batch 平均 loss（**所有 rank 都必须调用**，内部含集合通信）。"""
        return distributed_scalar_mean(self.loss_acc.total, self.loss_acc.count,
                                      self.device, self.ddp)

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
        """委托给 schedule.get_dynamic_lr（纯函数，单测见 tests/test_schedule.py）。"""
        return get_dynamic_lr(target_lr, cur_step, warmup_steps, decay_steps)

    def _adjust_lr(self):
        """按"即将执行的第 self.step+1 次更新"设置 LR 并返回。

        horizon 用 `effective_max`（本次 run 真正要跑的更新数），而不是 `total_steps`
        （epochs × 每 epoch 更新数）：增量续训 / `--train_max_steps` 下后者可能大一个
        数量级，余弦永远走不完，末端 LR 停在接近峰值处。
        """
        horizon = self.effective_max or self.total_steps
        lr = get_dynamic_lr(self.target_lr, self.step + 1, self.warmup_steps, horizon)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _check_and_evaluate(self, lr):
        # self.step 是在调用本方法**之前**自增的，所以这里直接按已完成步数取模。
        # （旧写法 `(self.step + 1) % eval_steps` 使首次 eval 只积累了 eval_steps-1
        #   次更新，train_loss 分母却按 eval_steps 算，首个窗口系统性偏小。）
        if self.eval_steps <= 0 or self.step % self.eval_steps != 0:
            return

        # 所有 rank 都要参与 loss 的 all_reduce；只有 rank0 真正跑 eval
        train_loss = self._global_mean_loss()
        self.loss_acc.snapshot_and_reset()
        if self.is_main_process:
            eval_loss = self._evaluate()
            grad_norm = self.last_grad_norm
            self._record_metrics(train_loss, eval_loss, grad_norm, lr)

        dist.barrier() if self.ddp else None

    def _reset_loss_acc(self):
        self.loss_acc.total = 0.0
        self.loss_acc.count = 0

    def _evaluate(self):
        # 这里不能多进程同步，必须用原始Model
        model = self._unwrap()
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
        model = self._unwrap()
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

        # 梯度累积：只有最后一个 micro-batch 才需要跨卡同步梯度，前面几个用
        # DDP.no_sync() 跳过 all-reduce（否则累积 N 步就做 N 次通信，梯度通信量 ×N）
        is_last_micro = (self.micro_step + 1) >= self.gradient_accumulation_steps
        no_sync = self.model.no_sync() if (self.ddp and not is_last_micro) else nullcontext()

        with no_sync:
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
        # 指标钩子在 zero_grad **之前**：直方图需要读到 p.grad（否则 grads/* 永远为空）
        if self.metrics is not None:
            self.metrics.before_zero_grad(self.step + 1, tokens=int(X.numel()),
                                          batch=(X, Y))
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
        self.set_train_epoch(cur_epoch)
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

        # epoch 循环上界：真实事故是 `--train_max_steps 200000` 配 `--train_epochs 1`
        # 时，epoch 循环先结束，训练在 updates_per_epoch 步就静默停下（远少于目标步数）。
        # 这里按 max_steps 反推所需的 epoch 数，保证"要多少步就跑到多少步"。
        loop_epochs = self.num_epochs
        if self.max_updates > 0 and self.updates_per_epoch > 0 \
                and self.max_updates > self.total_steps:
            loop_epochs = max(loop_epochs, math.ceil(self.max_updates / self.updates_per_epoch))
            if self.verbose:
                print(f"[trainer] max_steps={self.max_updates} 超过 {self.num_epochs} 个 epoch 的 "
                      f"{self.total_steps} 次更新，自动把 epoch 上界提到 {loop_epochs}")

        for epoch in range(last_epoch, loop_epochs):
            if self.effective_max and self.step >= self.effective_max:
                break
            self._train_epoch(epoch)

        # 结束：先在所有 rank 上完成 loss 的集合通信，主进程再做最终评估与落盘
        if self.step <= 0:
            final_train_loss = None
        elif self.loss_acc.count > 0:
            final_train_loss = self._global_mean_loss()
        else:
            # 最后一个 eval 窗口恰好落在终点：累加器已被清零，用该窗口的快照值
            final_train_loss = self.loss_acc.last_mean
        if self.is_main_process:
            final_eval = None
            if self.step > 0:
                final_eval = self._evaluate()
                tail_train_loss = final_train_loss
                self.final_metrics = {
                    "step": self.step,
                    "epoch": last_epoch + max(0, self.num_epochs - last_epoch - 1)
                             if self.step else last_epoch,
                    "train_loss": tail_train_loss,
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
                self.metrics.on_final(self.step, final_eval)
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
        self._cleanup()

    def set_writer(self, writer):
        """注入 tensorboard SummaryWriter（仅主进程）。"""
        self.writer = writer
    
    def predict(self, tokenizer, input_text, max_length=100):
        inputs = torch.tensor([tokenizer.encode(input_text)]).to(self.device)
        model = self._unwrap()
        response_ids = model.generate(inputs, max_length=max_length, eos_token_id=tokenizer.eos_token_id, use_kv_cache=True)
        new_tokens = response_ids[0][inputs.shape[1]:]
        return tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True).strip()


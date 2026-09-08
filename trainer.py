import os
import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as f
from contextlib import nullcontext
import torch.distributed as dist
from datetime import timedelta
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.distributed import init_process_group, destroy_process_group 
from transformer import GPTConfig, MiniGPT


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
        self.output_dir = train_args.get("output_dir")
        self.last_checkpoint_path = train_args.get("last_checkpoint_path")
        self.train_set = None
        self.eval_set = None
        self.train_loader = None
        self.eval_loader = None
        self.steps_per_epoch = 0
        self.step = 0
        self.total_steps = 0
        self.train_loss_acc = 0
        self.ddp = False
        self.rank = -1
        self.local_rank = -1
        self.is_main_process = True
        self.scaler = None
        self.batch_collator = None

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
                                       num_workers=0, 
                                       drop_last=True, 
                                       collate_fn=self.batch_collator,
                                       sampler=sampler)
        self.eval_loader = DataLoader(eval_set, 
                                      batch_size=batch_size, 
                                      shuffle=True, 
                                      num_workers=0, 
                                      drop_last=False,
                                      collate_fn=self.batch_collator)
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.num_epochs * self.steps_per_epoch
        print(f'init train_loader steps: {len(self.train_loader)}, eval_loader: {len(self.eval_loader)}') if self.verbose else None

    def _save_model(self, checkpoint_path, epoch):
        model, optimizer, step = self.model, self.optimizer, self.step
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        local_model = model.module if isinstance(model, DistributedDataParallel) else model
        torch.save({
            "model_state": local_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        }, checkpoint_path)

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
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda', enabled=True)
        else:
            self.scaler = None
        print("init grad scaler: ", self.use_mixed_precision) if self.verbose else None

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
        print(f"{self.cur_time()} lr={lr:.5f}, train_loss: {train_loss:.4f}, "
            + f"eval_loss: {eval_loss:.4f}, grad_norm={grad_norm:.5f}, "
            + f"steps: {self.step}/{self.total_steps}"
        )

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
            train_loss = self.train_loss_acc/self.eval_steps
            eval_loss = self._evaluate()
            grad_norm = self._calc_grad_norm()
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
        enable_mixed_precision = X.device.type == "cuda" and self.scaler is not None
        ctx = (torch.amp.autocast('cuda') if enable_mixed_precision else nullcontext())  
        
        self.optimizer.zero_grad(set_to_none=True)
        with ctx:
            logits = self.model(X, attention_mask=attnmask)
            loss = f.cross_entropy(logits.flatten(0, 1), Y.flatten())
    
        if enable_mixed_precision:  # 检查是否使用混合精度
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()  # 普通精度的反向传播
            # 梯度裁剪在混合精度与全精度下保持一致，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()  # 更新参数

        return loss

    def _train_epoch(self, cur_epoch):
        assert self.train_loader and self.eval_loader, f"train_loader and eval_loader can't be empty."
        # 从中断的数据索引位置继续训练
        skip_steps = self.step - cur_epoch * self.steps_per_epoch
        
        # 每个epoch开始时都重新打乱数据
        self.train_loader.sampler.set_epoch(cur_epoch) if self.ddp else None  
        print(f"{self.cur_time()} start epoch:{cur_epoch} from step:{self.step}") if self.verbose else None
        
        for i, batch in enumerate(self.train_loader):  
            if i < skip_steps: continue
            X, Y = batch[0].to(self.device), batch[1].to(self.device)
            attnmask = batch[2].to(self.device) if len(batch) == 3 else None     
            lr = self._adjust_lr()
            train_loss = self._train_step(X, Y, attnmask)
            self._accumulate_training_loss(train_loss)
            self.step += 1
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
        # 分布式训练需要使用ddp同步模型状态
        if self.ddp:
            self.model = self._wrap_model_with_ddp(self.model, self.local_rank)
        
        for epoch in range(last_epoch, self.num_epochs):
            self._train_epoch(epoch)

        self._cleanup()
    
    def predict(self, tokenizer, input_text, max_length=100):
        inputs = torch.tensor([tokenizer.encode(input_text)]).to(self.device)
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        response_ids = model.generate(inputs, max_length=max_length, eos_token_id=tokenizer.eos_token_id, use_kv_cache=True)
        return tokenizer.decode(response_ids.squeeze(0), skip_special_tokens=False)


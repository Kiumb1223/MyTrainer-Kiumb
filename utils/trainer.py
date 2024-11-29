#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :trainer.py
:Description:
:EditTime   :2024/11/25 14:34:14
:Author     :Kiumb
'''

import os
import time
import torch
import datetime
import torch.nn as nn
from loguru import logger
from typing import  Optional,Union
from utils.misc import get_model_info
from torch.utils.data import DataLoader
from utils.lr_scheduler import LRScheduler
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
from utils.metric import MeterBuffer, gpu_mem_usage
from torch.nn.parallel import DistributedDataParallel
from utils.distributed import get_rank, get_local_rank,synchronize


__all__ = ['GraphTrainer']

class GraphTrainer:
    def __init__(self,
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 lr_scheduler:Union[LRScheduler,_LRScheduler], 
                 loss_func: nn.Module,
                 max_epoch:int,
                 train_loader:DataLoader, 
                 val_loader:DataLoader=None, 
                 enable_amp:bool=False,
                 clip_grad_norm:float=0.0, 
                 work_dir:str='work_dir',
                 log_period:int=10,        # iteration interval
                 checkpoint_period:int=10, # epoch interval
                 device:str = None,
                 ):
        model.train()
        self.model   = model
        self.optimizer  = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epoch  = max_epoch
        self._clip_grad_norm = clip_grad_norm
        self._enable_amp  = enable_amp

        self.work_dir   = work_dir
        self.log_period = log_period
        self.checkpoint_period = checkpoint_period
        

        self._train_iter = iter(self.train_loader)
        self.meter  = MeterBuffer(window_size=10)
        self.epoch_len  = len(self.train_loader)
        self.ckpt_dir   = os.path.join(self.work_dir, 'checkpoints')
        self.tb_log_dir = os.path.join(self.work_dir, 'tb_logs')

        self.device     = device if device is not None else 'cpu'
        self.rank       = get_rank()
        if self.rank == 0:
            os.makedirs(self.work_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.tb_log_dir, exist_ok=True)
            self.tbWritter = SummaryWriter(self.tb_log_dir)



    @property
    def cur_total_iter(self) -> int:
        '''The total number of current iterations'''
        return self.cur_epoch * self.epoch_len + self.cur_iter

    @property
    def model_or_module(self) -> nn.Module:
        """The model not wrapped by :class:`DistributedDataParallel`."""
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

    def train(self,ckpt_path:Optional[str]=None,auto_resume:Optional[bool]=True):
        self.before_train(ckpt_path,auto_resume)
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.cur_epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.cur_iter in range(self.epoch_len):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def before_train(self,ckpt_path:Optional[str]=None,auto_resume:Optional[bool]=True):
        '''

        '''
        self._start_train_time = time.perf_counter()
        logger.info("Model info:\n" + get_model_info(self.model))

        split_line = "-" * 50
        logger.info(f"\n{split_line}\n"
                    f"Work directory: {self.work_dir}\n"
                    f"Checkpoint directory: {self.ckpt_dir}\n"
                    f"Tensorboard directory: {self.tb_log_dir}\n"
                    f"{split_line}")
        self._grad_scaler = GradScaler(enabled=self._enable_amp)
        if self._enable_amp:
            logger.info("Automatic Mixed Precision (AMP) training is on.")
        
        
        self.load_checkpoint(ckpt_path,auto_resume)

    def after_train(self):
        if self.rank == 0 :
            self.tbWritter.close()
        logger.info(f"Total training time: {datetime.timedelta(seconds=time.perf_counter() - self._start_train_time)}")
    
    def before_epoch(self):
        '''distributed data parallel setting'''
        if hasattr(self.train_loader.sampler,"set_epoch"):
            self.train_loader.sampler.set_epoch(self.cur_epoch)
        elif hasattr(self.train_loader.batch_sampler.sampler,"set_epoch"):
            # batch sampler in Pytorch warps the sampler as its attributes
            self.train_loader.batch_sampler.sampler.set_epoch(self.cur_epoch)

    def after_epoch(self):
        '''save latest checkpoint and eval model'''

        self.lr_scheduler.step()

        self.save_checkpoint("latest.pth")
        if (self.cur_epoch + 1) % self.checkpoint_period == 0:
            if self.val_loader is not None:
                self.eval_save_model()
            else:
                self.save_checkpoint(f"epoch_{self.cur_epoch}.pth")
    def before_iter(self):
        pass
    def train_one_iter(self):

        iter_start_time = time.perf_counter()
        #---------------------------------#
        #  load batch data 
        #---------------------------------#
        try:
            batch = next(self._train_iter)
        except StopIteration:
            logger.warning("StopIteration Occur.")
            self._train_iter = iter(self.train_loader)
            batch = next(self._train_iter)
        data_end_time = time.perf_counter()

        #---------------------------------#
        # Calculate loss
        #---------------------------------#
        with autocast(enabled=self._enable_amp):
            tra,det,gt_mtx_list = batch
            det,tra = det.to(self.device),tra.to(self.device)
            gt_mtx_list   = [i.to(self.device) for i in gt_mtx_list]
            pred_mtx_list = self.model(det,tra)
        # RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.
        # Tackle this error by disabling the  Mixed Precision Training 
        losses = self.loss_func(pred_mtx_list,gt_mtx_list)
        
        #---------------------------------#
        # Calculate gradients
        #---------------------------------#
        self.optimizer.zero_grad()
        self._grad_scaler.scale(losses).backward()
        if self._clip_grad_norm > 0:
            self._grad_scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self._clip_grad_norm)

        #---------------------------------#
        # Update model parameters
        #---------------------------------#
        self._grad_scaler.step(self.optimizer)
        self._grad_scaler.update()

        lr = self.optimizer.param_groups[0]['lr']
        # lr = self.lr_scheduler.update_lr(self.cur_total_iter + 1)
        # for param_group in self.optimizer.param_groups:
        #     param_group["lr"] = lr
        
        iter_end_time = time.perf_counter()

        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            loss=losses,
        )
    def after_iter(self):
        """ log information """
        if (self.cur_iter + 1) % self.log_period == 0:
            # write to console 
            left_iters = self.epoch_len * self.max_epoch - (self.cur_total_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}"

            progress_str = f"epoch: {self.cur_epoch + 1}/{self.max_epoch}, iter: {self.cur_iter + 1}/{self.epoch_len}"
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.3f}".format(k, v.latest.item()) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                f"{progress_str}, mem: {gpu_mem_usage():.0f}MB, {time_str}, " +
                f"{loss_str}, lr: {self.meter['lr'].latest:.3e}, {eta_str}"
            )
            if self.rank == 0:
                # write to tensorboard
                self.tbWritter.add_scalar('Train/loss',list(loss_meter.values())[-1].latest, self.cur_epoch + 1)
            # empty the meters
            self.meter.clear_meters()
        
    def load_checkpoint(self, path: Optional[str] = None, auto_resume: bool = True):
        """Load the given checkpoint or resume from the latest checkpoint.

        Args:
            path (str): Path to the checkpoint to load.
            auto_resume (bool): If True, automatically resume from the latest checkpoint.
        """
        if path is None and auto_resume:
            latest_ckpt = os.path.join(self.ckpt_dir, "latest.pth")
            if not os.path.exists(latest_ckpt):
                logger.warning("You specify auto_resume=True, but we fail to find "
                               f"{latest_ckpt} to auto resume from.")
            else:
                logger.info(f"Found {latest_ckpt} to auto resume from.")
                path = latest_ckpt
        if path:
            logger.info(f"Loading checkpoint from {path} ...")
            checkpoint = torch.load(path, map_location="cpu")
        else:
            logger.info("Skip loading checkpoint.")
            self.start_epoch = 0
            return

        # # check if the number of GPUs is consistent with the checkpoint
        # num_gpus = get_world_size()
        # ckpt_num_gpus = checkpoint["num_gpus"]
        # assert num_gpus == ckpt_num_gpus, (
        #     f"You are trying to load a checkpoint trained with {ckpt_num_gpus} GPUs, "
        #     f"but currently only have {num_gpus} GPUs.")

        # 1. load epoch / iteration
        self.start_epoch = checkpoint["start_epoch"]
        

        # 2. load model
        incompatible = self.model_or_module.load_state_dict(checkpoint["model"], strict=False)
        if incompatible.missing_keys:
            logger.warning("Encounter missing keys when loading model weights:\n"
                           f"{incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logger.warning("Encounter unexpected keys when loading model weights:\n"
                           f"{incompatible.unexpected_keys}")

        # # 3. load metric_storage
        # self.metric_storage = checkpoint["metric_storage"]

        # 4. load optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # # 5. load lr_scheduler
        # self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # 6. load grad scaler
        consistent_amp = not (self._enable_amp ^ ("grad_scaler" in checkpoint))
        assert consistent_amp, "Found inconsistent AMP training setting when loading checkpoint."
        if self._enable_amp:
            self._grad_scaler.load_state_dict(checkpoint["grad_scaler"])

    def save_checkpoint(self, file_name:str):
        '''
        Save training state: ``start_epoch``, ``model``,
            ``optimizer``, ``grad_scaler`` (optional).
        '''
        if self.rank == 0:
            data = {
                "start_epoch": self.cur_epoch + 1,
                "model": self.model_or_module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            if self._enable_amp:
                data["grad_scaler"] = self._grad_scaler.state_dict()

            file_path = os.path.join(self.ckpt_dir, file_name)
            logger.info(f"Saving checkpoint to {file_path}")
            torch.save(data, file_path)

    def eval_save_model(self):
        if self.rank == 0:

            self.model.eval()
            #---------------------------------#
            # eval and save the best model 
            # wait to implement
            #---------------------------------#
            self.model.train()
            self.tbWritter('wait to write')
            
        synchronize()
        
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :train.py
:Description:
:EditTime   :2024/11/25 14:52:12
:Author     :Kiumb
'''
import torch
import cProfile
from loguru import logger
from torch.optim import AdamW
from configs.config import get_config
from utils.logger import setup_logger
from models.lossFunc import GraphLoss
from utils.distributed import get_rank
from torch.utils.data import DataLoader
from models.graphModel import GraphModel
from utils.graphTrainer import GraphTrainer
from torch.optim.lr_scheduler import MultiStepLR
from utils.graphDataset import GraphDataset, graph_collate_fn
from utils.misc import collect_env,get_exp_info,set_random_seed

@logger.catch
def main():

    cfg = get_config()
    #---------------------------------#
    #  print some necessary infomation
    #---------------------------------#
    setup_logger(cfg.WORK_DIR,get_rank(),f'log_rank{get_rank()}.txt')
    logger.info("Environment info:\n" + collect_env())
    logger.info("Config info:\n" + get_exp_info(cfg))

    #---------------------------------#
    #  prepare training
    #---------------------------------#
    set_random_seed(cfg.RANDOM_SEED)
    train_dataset = GraphDataset(cfg,'Train',True)  # Move tensor to the device specified in cfg.DEVICE
    test_dataset  = GraphDataset(cfg,'Validation')

    train_loader  = DataLoader(train_dataset,batch_size=cfg.BATCH_SIZE,shuffle=True,pin_memory=True,
                               num_workers=cfg.NUM_WORKS,collate_fn=graph_collate_fn,drop_last=True)

    valid_loader   = DataLoader(test_dataset,batch_size=cfg.BATCH_SIZE,shuffle=True,pin_memory=True,
                               num_workers=cfg.NUM_WORKS,collate_fn=graph_collate_fn,drop_last=True)
    
    model = GraphModel(cfg).to(cfg.DEVICE)
    optimizer = AdamW(model.parameters(), lr=cfg.LR,weight_decay=cfg.WEIGHT_DECAY)
    lr_scheduler = MultiStepLR(optimizer,milestones=cfg.MILLESTONES)
    loss_func = GraphLoss()

    graphTrainer = GraphTrainer(
        model=model,optimizer=optimizer,lr_scheduler=lr_scheduler,loss_func=loss_func,
        max_epoch=cfg.MAXEPOCH,train_loader=train_loader,val_loader=valid_loader,enable_amp=cfg.EMABLE_AMP,
        work_dir=cfg.WORK_DIR,log_period=cfg.LOG_PERIOD,checkpoint_period=cfg.CHECKPOINT_PERIOD,device = cfg.DEVICE,
        # warmup settings
        by_epoch = cfg.BY_EPOCH,warmup_t = cfg.WARMUP_T,warmup_by_epoch = cfg.WARMUP_BY_EPOCH,
        warmup_mode = cfg.WARMUP_MODE,warmup_init_lr = cfg.WARMUP_INIT_LR,warmup_factor = cfg.WARMUP_FACTOR,
    
    )
    #---------------------------------#
    #  start Training
    #---------------------------------#
    graphTrainer.train()

if __name__ == '__main__':
    main()
    # cProfile.run('main()',filename='TimeAnalysis.out')
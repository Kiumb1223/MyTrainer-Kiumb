#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :train.py
:Description:
:EditTime   :2024/11/25 14:52:12
:Author     :Kiumb
'''

import torch
from loguru import logger
from torch.optim import AdamW
from utils.trainer import MotTrainer
from models.Tracker import Tracker
from models.Lossfunc import GraphLoss
from torch.utils.data import DataLoader
from utils.misc import set_random_seed,load_config
from utils.lr_scheduler import LRScheduler
from utils.mot_dataset import MOTGraph,graph_collate_fn

@logger.catch
def main():

    cfg_exp = load_config('configs/train.yaml')
    cfg_data = load_config('configs/data.yaml')
    cfg_model = load_config('configs/model.yaml')
    cfg = {**cfg_exp,**cfg_data,**cfg_model}
    train_dataset = MOTGraph(cfg,'Train_split')
    test_dataset  = MOTGraph(cfg,'Validation')
    train_loader  = DataLoader(train_dataset,batch_size=cfg['BATCH_SIZE'],shuffle=True,pin_memory=True,
                              num_workers=0,collate_fn=graph_collate_fn,drop_last=True)
    test_loader   = DataLoader(test_dataset,batch_size=cfg['BATCH_SIZE'],shuffle=True,pin_memory=True,
                              num_workers=0,collate_fn=graph_collate_fn,drop_last=True)
    # cfg['NUM_WORKS']
    model = Tracker(cfg)
    model.to(cfg['DEVICE'])
    optimizer = AdamW(model.parameters(), lr=cfg['LR'],weight_decay=cfg['WEIGHT_DECAY'])
    lr_scheduler = LRScheduler(name=cfg['SCHEDULER'],lr = cfg['LR'],
            iters_per_epoch = len(train_loader),total_epochs =cfg['MAXEPOCH'],
            warmup_epochs=cfg['WARMUP_EPOCHS'],warmup_lr_start=cfg['WARM_LR'],
            no_aug_epochs=cfg['NO_AUG_EPOCHS'],min_lr_ratio=cfg['MIN_LR_RATIO'],)
    lossfunc = GraphLoss()
    
    trainer = MotTrainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,loss_func=lossfunc,
                         train_loader=train_loader, max_epoch = cfg['MAXEPOCH'],
                         work_dir=cfg['WORK_DIR'], log_period=cfg['LOG_PERIOD'],enable_amp=cfg['EMABLE_AMP'],
                         seed=cfg['RANDOM_SEED'],device=cfg['DEVICE'])
    trainer.train()
    
if __name__ == '__main__':
    main()
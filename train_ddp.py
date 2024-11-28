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
from argparse import Namespace
from utils.logger import setup_logger
from models.lossFunc import GraphLoss
from utils.trainer import GraphTrainer
from torch.utils.data import DataLoader
from utils.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import StepLR
from models.graphTracker import GraphTracker
from torch.nn.parallel import DistributedDataParallel
from utils.distributed import get_rank,init_distributed
from torch.utils.data.distributed import DistributedSampler
from utils.graphDataset import GraphDataset, graph_collate_fn
from utils.misc import collect_env,get_exp_info,set_random_seed
@logger.catch
def main():

    cfg = Namespace(
        #---------------------------------#
        #  Experimental setting
        #---------------------------------#
        RANDOM_SEED       = 1,
        LOG_PERIOD        = 10,       # Iteration 
        CHECKPOINT_PERIOD = 5,        # Epoch
        DEVICE            = 'cuda',
        NUM_WORKS         = 0,
        EMABLE_AMP        = True,
        WORK_DIR          = "experiments",

        LR                = 3e-4,
        WEIGHT_DECAY      = 1e-4,
        BATCH_SIZE        = 2,
        MAXEPOCH          = 50,
        
        # StepLR
        LR_DROP           = 40,
        # Lr scheduler (self)
        WARMUP_EPOCHS     = 5,
        NO_AUG_EPOCHS     = 0,
        MIN_LR_RATIO      = 0.05,
        SCHEDULER         = 'yoloxwarmcos',
        WARM_LR           = 0,

        #---------------------------------#
        #  Model related
        #---------------------------------#
        MAXAGE            = 100,  # Maximum age for tracking an object
        K_NEIGHBOR        = 2,    # Excluding self-loop
        RESIZE_TO_CNN     = [224, 224],
        NODE_EMBED_SIZE   = 32,
        EDGE_EMBED_SIZE   = 18,
        SINKHORN_ITERS    = 8,

        #---------------------------------#
        #  Dataset related
        #---------------------------------#
        DETECTOR          = 'FRCNN',
        DATA_DIR          = 'datasets/MOT17/train',
        ACCEPTABLE_OBJ_TYPE=[1, 2, 7],
        MOT17_TRAIN_NAME  = [
            'MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
            'MOT17-10', 'MOT17-11', 'MOT17-13'
        ],
        MOT17_TRAIN_START = [2, 2, 2, 2, 2, 2, 2],
        MOT17_VAL_NAME    = [
            'MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
            'MOT17-10', 'MOT17-11', 'MOT17-13'
        ],
        MOT17_VAL_START   = [501, 951, 738, 426, 555, 801, 651],
        MOT17ALLFRAMENUM  = {
            'MOT17-01': 450, 'MOT17-02': 600, 'MOT17-03': 1500,
            'MOT17-04': 1050, 'MOT17-05': 837, 'MOT17-06': 1194,
            'MOT17-07': 500, 'MOT17-08': 625, 'MOT17-09': 525,
            'MOT17-10': 654, 'MOT17-11': 900, 'MOT17-12': 900,
            'MOT17-13': 750, 'MOT17-14': 750
        },
        TRACKBACK_WINDOW  = 10,
    )

    rank , local_rank ,world_size = init_distributed()
    is_distributed = world_size > 1 
    #---------------------------------#
    #  print some necessary infomation
    #---------------------------------#
    setup_logger(cfg.WORK_DIR,rank,f'log_rank{get_rank()}.txt')
    logger.info("Environment info:\n" + collect_env())
    logger.info("Config info:\n" + get_exp_info(cfg))

    #---------------------------------#
    #  prepare training
    #---------------------------------#
    set_random_seed(None if cfg.RANDOM_SEED < 0 else cfg.RANDOM_SEED + rank)
    train_dataset = GraphDataset(cfg,'Train_split')  # Move tensor to the device specified in cfg.DEVICE
    test_dataset  = GraphDataset(cfg,'Validation')

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None

    train_loader  = DataLoader(train_dataset,batch_size=cfg.BATCH_SIZE,shuffle=False,sampler=train_sampler,
                               pin_memory=False if cfg.DEVICE.startswith('cuda') and torch.cuda.is_available() else True,
                               num_workers=cfg.NUM_WORKS,collate_fn=graph_collate_fn,drop_last=True)

    test_loader   = DataLoader(test_dataset,batch_size=cfg.BATCH_SIZE,shuffle=False,
                               pin_memory=False if cfg.DEVICE.startswith('cuda') and torch.cuda.is_available() else True,
                               num_workers=cfg.NUM_WORKS,collate_fn=graph_collate_fn,drop_last=True)
    
    model = GraphTracker(cfg).to(cfg.DEVICE)
    if is_distributed:
        model = DistributedDataParallel(model,device_ids=[local_rank])
        
    optimizer = AdamW(model.parameters(), lr=cfg.LR,weight_decay=cfg.WEIGHT_DECAY)
    # lr_scheduler = LRScheduler(name=cfg.SCHEDULER,lr = cfg.LR,
    #             iters_per_epoch = len(train_loader),total_epochs =cfg.MAXEPOCH,
    #             warmup_epochs=cfg.WARMUP_EPOCHS,warmup_lr_start=cfg.WARM_LR,
    #             no_aug_epochs=cfg.NO_AUG_EPOCHS,min_lr_ratio=cfg.MIN_LR_RATIO,)
    
    lr_scheduler = StepLR(optimizer,cfg.LR_DROP)
    loss_func = GraphLoss()

    graphTrainer = GraphTrainer(
        model=model,optimizer=optimizer,lr_scheduler=lr_scheduler,loss_func=loss_func,
        max_epoch=cfg.MAXEPOCH,train_loader=train_loader,enable_amp=cfg.EMABLE_AMP,
        work_dir=cfg.WORK_DIR,log_period=cfg.LOG_PERIOD,checkpoint_period=cfg.CHECKPOINT_PERIOD,device = cfg.DEVICE
    )
    #---------------------------------#
    #  Training
    #---------------------------------#
    graphTrainer.train()
if __name__ == '__main__':
    main()
    # cProfile.run('main()',filename='TimeAnalysis.out')
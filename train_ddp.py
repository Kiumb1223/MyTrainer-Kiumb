#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :train_ddp.py
:Description:
:EditTime   :2024/11/25 14:53:12
:Author     :Kiumb

OMP_NUM_THREADS=2 python -m torch.distributed.launch  --nproc_per_node 2 train_ddp.py
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 train_ddp.py

'''

import torch
from loguru import logger
from torch.optim import AdamW
from configs.config import get_config
from utils.logger import setup_logger
from models.lossFunc import GraphLoss
from torch.utils.data import DataLoader
from utils.graphTrainer import GraphTrainer
from models.graphModel import TrainingGraphModel
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel import DistributedDataParallel
from utils.distributed import get_rank,init_distributed
from torch.utils.data.distributed import DistributedSampler
from utils.graphDataset import GraphDataset, graph_collate_fn
from utils.misc import collect_env,get_exp_info,set_random_seed,get_model_configuration


@logger.catch
def main():

    cfg = get_config()

    rank , local_rank ,world_size = init_distributed()
    is_distributed = world_size > 1 
    #---------------------------------#
    #  print some necessary infomation
    #---------------------------------#
    setup_logger(cfg.WORK_DIR,rank,f'log_rank{get_rank()}.txt')
    logger.info("Environment info:\n" + collect_env())
    logger.info("Config info:\n" + get_exp_info(cfg))
    logger.info("Model Config:\n" + get_model_configuration(cfg.MODEL_YAML_PATH))

    #---------------------------------#
    #  prepare training
    #---------------------------------#
    set_random_seed(None if cfg.RANDOM_SEED < 0 else cfg.RANDOM_SEED + rank)
    train_dataset = GraphDataset(cfg,'Train',False)  # Move tensor to the device specified in cfg.DEVICE
    test_dataset  = GraphDataset(cfg,'Validation')

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None

    train_loader  = DataLoader(train_dataset,batch_size=cfg.BATCH_SIZE,shuffle=False,sampler=train_sampler,pin_memory=True,
                               num_workers=cfg.NUM_WORKS,collate_fn=graph_collate_fn,drop_last=True)

    valid_loader   = DataLoader(test_dataset,batch_size=cfg.BATCH_SIZE,shuffle=False,pin_memory=True,
                               num_workers=cfg.NUM_WORKS,collate_fn=graph_collate_fn,drop_last=True)
    
    model = TrainingGraphModel(cfg['MODEL_YAML_PATH']).to(cfg.DEVICE)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) .to(cfg.DEVICE)
    if is_distributed:
        model = DistributedDataParallel(model,device_ids=[local_rank])
        
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

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File     :     config.py
@Time     :     2024/12/03 16:13:11
@Author   :     Louis Swift
@Desc     :     Parameter Configuration System
'''


from argparse import Namespace
def get_config():
    cfg = Namespace(
        #---------------------------------#
        #  1. Experimental setting
        #---------------------------------#
        RANDOM_SEED       = 1,
        LOG_PERIOD        = 10,       # Iteration 
        CHECKPOINT_PERIOD = 1,        # Epoch
        DEVICE            = 'cuda',
        NUM_WORKS         = 0,
        EMABLE_AMP        = True,
        WORK_DIR          = "experiments",

        LR                = 3.5e-4,
        WEIGHT_DECAY      = 1e-4,
        BATCH_SIZE        = 2,
        MAXEPOCH          = 120,
        
        # lr scheduler(MultiStepLR)
        MILLESTONES       = [40,70],
        # warmup settings
        # see: https://core-pytorch-utils.readthedocs.io/en/latest/
        BY_EPOCH          = True,
        WARMUP_T          = 5000,
        WARMUP_BY_EPOCH   = False,
        WARMUP_MODE       = "auto",
        WARMUP_INIT_LR    = 0.0,
        WARMUP_FACTOR     = 0.05,
        #---------------------------------#
        #  2. Model related
        #---------------------------------#
        FAST_REID_MODELE  = 'fastreid_market_BOT_R50_ibn',
        FAST_REID_WEIGHT  = 'model_weights/fastreid-models/market_bot_R50-ibn.pth',

        K_NEIGHBOR        = 2,    # Excluding self-loop
        RESIZE_TO_CNN     = [256, 128],
        NODE_EMBED_SIZE   = 32,
        EDGE_EMBED_SIZE   = 16,
        SINKHORN_ITERS    = 8,

        #---------------------------------#
        #  3. Dataset related
        #---------------------------------#
        DATA_DIR          = 'datasets',
        JSON_PATH         = 'configs/train_mix.json',
        TRACKBACK_WINDOW  = 10,
        ACCEPTABLE_OBJ_TYPE = [1,2,7],
        #---------------------------------#
        #  4. TrackManager related
        #---------------------------------#
        PATH_TO_WEIGHTS   = r'model_weights\bestScore(1.0)_epoch40.pth',
        MIN_DET_CONF      = 0.1,
        MATCH_THRESH      = 0.05,
        Det2Tra_CONF      = 0.7,
        CNT_TO_ACTIVE     = 1,
        CNT_TO_SLEEP      = 1,
        MAX_CNT_TO_DEAD   = 30,  # Maximum age for tracking an object
        FEATURE_LIST_SIZE = 10,

        #---------------------------------#
        # 5. Evalution related
        #---------------------------------#
        EVAL_FOLDER       = 'datasets/eval_datasets'
    )

    return cfg
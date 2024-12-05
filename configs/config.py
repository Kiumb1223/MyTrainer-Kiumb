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
        CHECKPOINT_PERIOD = 5,        # Epoch
        DEVICE            = 'cuda',
        NUM_WORKS         = 0,
        EMABLE_AMP        = True,
        WORK_DIR          = "experiments",

        LR                = 3.5e-4,
        WEIGHT_DECAY      = 1e-4,
        BATCH_SIZE        = 32,
        MAXEPOCH          = 120,
        
        # StepLR
        LR_DROP           = 40,
        # Lr scheduler (self)
        WARMUP_EPOCHS     = 5,
        NO_AUG_EPOCHS     = 0,
        MIN_LR_RATIO      = 0.05,
        SCHEDULER         = 'yoloxwarmcos',
        WARM_LR           = 0,

        #---------------------------------#
        #  2. Model related
        #---------------------------------#
        K_NEIGHBOR        = 2,    # Excluding self-loop
        RESIZE_TO_CNN     = [224, 224],
        NODE_EMBED_SIZE   = 32,
        EDGE_EMBED_SIZE   = 18,
        SINKHORN_ITERS    = 8,

        #---------------------------------#
        #  3. Dataset related
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
    
        #---------------------------------#
        #  4. TrackManager related
        #---------------------------------#
        PATH_TO_WEIGHTS   = r'experiments\checkpoints\epoch_59_marginal_1.pth',
        MIN_DET_CONF      = 0.1,
        MATCH_THRESH      = 0.05,
        Det2Tra_CONF      = 0.7,
        CNT_TO_ACTIVE     = 1,
        MAX_CNT_TO_DEAD   = 30,  # Maximum age for tracking an object
        FEATURE_LIST_SIZE = 10,
    )

    return cfg
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
        RANDOM_SEED       = 3407,
        LOG_PERIOD        = 10,       # Iteration 
        CHECKPOINT_PERIOD = 1,        # Epoch
        DEVICE            = 'cuda',
        NUM_WORKS         = 2,
        EMABLE_AMP        = True,
        WORK_DIR          = "Edge_mode_exps/FixedProgressiveEdgeDimEdgeDim/vanillaModel[IoUFamily8-convex-seq]",

        BATCH_SIZE        = 16,
        MAXEPOCH          = 120,

        LR                = 1e-2,
        # Optimizer Adamw
        # WEIGHT_DECAY      = 1e-4,
        # Optimizer SGD
        MOMENTUM          = 0.937,
        WEIGHT_DECAY      = 5e-4,
        

        # lr scheduler(MultiStepLR)
        MILLESTONES       = [50,80],
        # lr scheduler(ExponentialLR)
        GAMMA             = 0.98,
        # warmup settings
        # see: https://core-pytorch-utils.readthedocs.io/en/latest/
        BY_EPOCH          = True,
        WARMUP_T          = 800,
        WARMUP_BY_EPOCH   = False,
        WARMUP_MODE       = "auto",
        WARMUP_INIT_LR    = 0.0,
        WARMUP_FACTOR     = 0.05,
        
        #---------------------------------#
        #  2. Model related
        #---------------------------------#
        MODEL_YAML_PATH   = r'configs/Edge_mode/ProgressiveEdgeDim/VanillaModel_ProgressiveEdgeDim[IoUFamily8-convex-seq].yaml',
        
        #---------------------------------#
        #  3. Dataset related
        #---------------------------------#
        DATA_DIR          = 'datasets',
        JSON_PATH         = 'configs/train_MOT17.json',
        TRACKBACK_WINDOW  = 10,
        ACCEPTABLE_OBJ_TYPE   = [1,2,7],

        RESIZE_TO_CNN     = [256, 128],  # [height , width]
        # data augumentation
        MIN_IDS_TO_DROP_PERC  = 0,     # Minimum percentage of ids s.t. all of its detections will be dropped
        MAX_IDS_TO_DROP_PERC  = 0.1,   # Maximum percentage of ids s.t. all of its detections will be dropped
        MIN_DETS_TO_DROP_PERC = 0,     # Minimum Percentage of detections that might be randomly dropped 
        MAX_DETS_TO_DROP_PERC = 0.2,   # Maximum Percentage of detections that might be randomly dropped

        #---------------------------------#
        #  4. TrackManager related
        #---------------------------------#
        PATH_TO_WEIGHTS   = r'model_weights\ModelZoo\selfConcat.pth',
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
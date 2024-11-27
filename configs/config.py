#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Parameter Configuration System
'''
import os
from easydict import EasyDict as edict
__C = edict()
cfg = __C
__C.RANDOM_SEED = 1
__C.LOG_PERIOD = 50
__C.CHECKPOINT_PERIOD = 5
__C.WORK_DIR = "experiments"
__C.DEVICE = 'cuda'
__C.NUM_WORKS = 2
#---------------------------------#
# Parameters concerning DATA
# Several situations when TRAIN.MODE:
#   1. MODE - 'Train_full'
#      -  Training data range: MOT17_TRAIN_START to MOT17ALLFRAMENUM
#   2. MODE - 'Train_split'
#      -  Training data range: MOT17_TRAIN_START to MOT17_VAL_START
#      -  
#   3. MODE - 'Validate'
#      - Validation data range: MOT17_VAL_START to MOT17ALLFRAMENUM
#---------------------------------#
__C.DATA = edict()
__C.DATA.DETECTOR = 'FRCNN'
__C.DATA.DATA_DIR = r'Datasets\MOT17\train'
__C.DATA.ACCEPTABLE_OBJ_TYPE = [1,2,7]
__C.DATA.MOT17_TRAIN_NAME    = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
__C.DATA.MOT17_TRAIN_START   = [2, 2, 2, 2, 2, 2, 2]
__C.DATA.MOT17_VAL_NAME      = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
__C.DATA.MOT17_VAL_START     = [501, 951, 738, 426, 555, 801, 651]
__C.DATA.MOT17ALLFRAMENUM    = {
    'MOT17-01':450,'MOT17-02':600,'MOT17-03':1500,'MOT17-04':1050,
    'MOT17-05':837,'MOT17-06':1194,'MOT17-07':500,'MOT17-08':625,
    'MOT17-09':525,'MOT17-10':654,'MOT17-11':900,'MOT17-12':900,
    'MOT17-13':750,'MOT17-14':750}

__C.DATA.TRACKBACK_WINDOW = 10 # use the past TRACKBACK_WINDOW frames to construct Tracklet Graph
__C.DATA.STATIC = []
__C.DATA.MOVING = []
__C.DATA.REID_DIR = ''
__C.DATA.MAXAGE = 100
# __C.DATA.AUGMENTATION = False

__C.TRAIN = edict()
# __C.TRAIN.MODE = 'Train_split' # 'Train_full' or 'Train_split' or 'Validate' or 'Test' 
# __C.TRAIN.FRAME_LOOK_BACK = 10 
__C.TRAIN.LR = 1.25e-4
__C.TRAIN.WEIGHT_DECAY = 1e-4
__C.TRAIN.MIN_LR_RATIO = 0.05
__C.TRAIN.BATCH_SIZE = 4
__C.TRAIN.MAXEPOCH = 50
__C.TRAIN.WARMUP_EPOCHS = 5
__C.TRAIN.NO_AUG_EPOCHS = 0
__C.TRAIN.SCHEDULER = 'yoloxwarmcos'
__C.TRAIN.WARM_LR = 0

#__C.gpu_id = 1,2,3

__C.MODEL = edict()
__C.MODEL.K_NEIGHBOR  = 2 
__C.MODEL.PATH_TO_WEIGHT  = None 
__C.MODEL.CROP_RESIZE_TO_CNN  = (224,224)
__C.MODEL.NODE_EMBED_SIZE   = 32
__C.MODEL.EDGE_EMBED_SIZE   = 18
__C.MODEL.SINKHORN_ITERS    = 8

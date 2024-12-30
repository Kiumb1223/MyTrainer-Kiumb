

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     fast_reid.py
@Time     :     2024/12/11 21:36:52
@Author   :     Louis Swift
@Desc     :     

Extracted from https://github.com/dvl-tum/SUSHI , with minor modification 
and much thanks for their brilliant works~
'''


import os
import sys
import torch
from pathlib import Path
from collections import OrderedDict


# Enable imports from the fast-reid directory
root = Path(__file__).parent.parent.parent
_FASTREID_ROOT = os.path.join(root,'thirdparty','fast-reid')
sys.path.append(_FASTREID_ROOT)

from fastreid.config import get_cfg 
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import  default_argument_parser

__all__  = ['load_fastreid_model']


_FASTREID_MODEL_ZOO = { 
    'msmt_SBS_R101_ibn': 'configs/MSMT17/sbs_R101-ibn.yml',
    'msmt_SBS_S50':'configs/MSMT17/sbs_S50.yml',
    'msmt_BOT_R101_ibn': 'configs/MSMT17/bagtricks_R101-ibn.yml',
    'msmt_AGW_R101_ibn': 'configs/MSMT17/AGW_R101-ibn.yml',

    # More BOT methods on MSMT17
    'msmt_BOT_S50': 'configs/MSMT17/bagtricks_S50.yml',
    'msmt_BOT_R50_ibn': 'configs/MSMT17/bagtricks_R50-ibn.yml',
    'msmt_BOT_R50':'configs/MSMT17/bagtricks_R50.yml',

    # Some BOT methods on Market
    #https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_R101-ibn.yml
    'market_BOT_R101_ibn':'configs/Market1501/bagtricks_R101-ibn.yml',
    'market_BOT_R50_ibn':'configs/Market1501/bagtricks_R50-ibn.yml',
    
    'market_bot_S50':'configs/Market1501/bagtricks_S50.yml',

    # MGN
    'market_mgn_R50_ibn': 'configs/Market1501/mgn_R50-ibn.yml'
                        
}    


def _get_cfg(fastreid_cfg_file):
    """
    Create configs and perform basic setups.
    """    
    args = default_argument_parser().parse_args(['--config-file', os.path.join(_FASTREID_ROOT, fastreid_cfg_file)])
    #cfg = setup(args)
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False

    return cfg

def load_ckpt_FastReid(model, path_to_weight):
    # Fix keys mismatches due to using different versions!
    ckpt = torch.load(path_to_weight, map_location = torch.device('cpu'))
    state_dict = model.state_dict()
    new_ckpt_model = OrderedDict()
    resave_ckpt = False
    for key, val in ckpt['model'].items():
        if key.startswith('heads.bnneck'):
            key_ = key.replace('heads.bnneck', 'heads.bottleneck.0')
            assert key_ in state_dict
            resave_ckpt=True
            
        else:
            key_ = key
        
        new_ckpt_model[key_] = val

    ckpt['model'] = new_ckpt_model
    if resave_ckpt:
        torch.save(ckpt, path_to_weight)
    
    Checkpointer(model).load(path_to_weight)  # load trained model

    return model


def load_fastreid_model(reid_arch_name):
    model_name = reid_arch_name.split('fastreid_')[-1]
    fastreid_cfg_file = _FASTREID_MODEL_ZOO[model_name]

    # build model
    cfg = _get_cfg(fastreid_cfg_file)
    model = build_model(cfg)

    feature_embedding_model = model.eval()

    return feature_embedding_model
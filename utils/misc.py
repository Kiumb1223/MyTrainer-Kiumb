#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Extracted from https://github.com/serend1p1ty/core-pytorch-utils, with minor modification
and much thanks for their brilliant works~
'''

import os
import sys
import yaml 
import torch
import random
import logging
import numpy as np
from argparse import Namespace
from typing import Optional
from tabulate import tabulate
from collections import defaultdict
from easydict import EasyDict as edict
from torch_geometric.profile.utils import get_model_size,count_parameters


__all__ = ["collect_env", "get_model_info","get_model_configuration","get_exp_info","set_random_seed", "symlink","load_config"]

logger = logging.getLogger(__name__)


def collect_env() -> str:
    """Collect the information of the running environments.

    The following information are contained.

        - sys.platform: The value of ``sys.platform``.
        - Python: Python version.
        - Numpy: Numpy version.
        - CUDA available: Bool, indicating if CUDA is available.
        - GPU devices: Device type of each GPU.
        - PyTorch: PyTorch version.
        - TorchVision (optional): TorchVision version.
        - Pytorch-geometric (optional): Pytorch-geometric version.

    Returns:
        str: A string describing the running environment.
    """
    env_info = []
    env_info.append(("sys.platform", sys.platform))
    env_info.append(("Python", sys.version.replace("\n", "")))
    env_info.append(("Numpy", np.__version__))

    cuda_available = torch.cuda.is_available()
    env_info.append(("CUDA available", cuda_available))

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info.append(("GPU " + ",".join(device_ids), name))

    env_info.append(("PyTorch", torch.__version__))

    try:
        import torchvision
        env_info.append(("TorchVision", torchvision.__version__))
    except ModuleNotFoundError:
        pass
    try:
        import torch_geometric
        env_info.append(("Pytorch-geometric", torch_geometric.__version__))
    except ModuleNotFoundError:
        pass

    # try:
    #     import cv2
    #     env_info.append(("OpenCV", cv2.__version__))
    # except ModuleNotFoundError:
    #     pass

    return tabulate(env_info,tablefmt="grid")

def get_model_info(model:torch.nn.Module) -> str:
    '''get model information'''
    model_info = []
    model_info.append(("Model size(MB)", get_model_size(model)/(1024*1024)))
    model_info.append(("Number of parameters(M)", count_parameters(model)/1e6))
    return tabulate(model_info,tablefmt="grid")

def get_model_configuration(model_yaml_path:str) -> str:
    '''get model configuration '''

    def flatten_dict(d, parent_key='', sep='.'):
        """
        Flatten a nested dictionary.
        
        :param d: Dictionary to flatten.
        :param parent_key: The base key for recursion (used for nested keys).
        :param sep: Separator for flattened keys.
        :return: A flattened dictionary where keys are combined with `sep`.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                # Recursively flatten nested dictionaries
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v,(str,list,tuple,dict)):
                items.append((new_key, v))
            else:
                items.append((new_key, [v]))
        return dict(items)
    
    with open(model_yaml_path, 'r') as f:
        model_dict = yaml.safe_load(f)
    flatten_model_dict = flatten_dict(model_dict)
    return tabulate(flatten_model_dict.items(),headers=['Setting', 'Value'],tablefmt="grid")

def get_exp_info(cfg:Namespace) -> str:
    '''get all experimental information for better debugging'''
    # training_keywords = ["BATCH", "MAXEPOCH","AMP","DEVICE",'K','BT_DIRECTED','BT_COSINE_DYGRAPH']
    training_params = {k: v for k, v in vars(cfg).items()}
    return tabulate(training_params.items(),headers=['Setting','Value'],tablefmt="grid")

def set_random_seed(seed: Optional[int] = None, deterministic: bool = False) -> None:
    """Set random seed.

    Args:
        seed (int): If None or negative, use a generated seed.
        deterministic (bool): If True, set the deterministic option for CUDNN backend.
    """
    if seed is None or seed < 0:
        new_seed = np.random.randint(2**32)
        logger.info(f"Got invalid seed: {seed}, will use the randomly generated seed: {new_seed}")
        seed = new_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set random seed to {seed}.")
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info("The CUDNN is set to deterministic. This will increase reproducibility, "
                    "but may slow down your training considerably.")


def symlink(src: str, dst: str, overwrite: bool = True, **kwargs) -> None:
    """Create a symlink, dst -> src.

    Args:
        src (str): Path to source.
        dst (str): Path to target.
        overwrite (bool): If True, remove existed target. Defaults to True.
    """
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)

def load_config(config_file):
    '''load YAML config files '''
    with open(config_file, 'r') as f:
        return edict(yaml.safe_load(f))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

1. By maintaining a lookup table(self.dets_dict), 
we can quickily retrive the detections infomation for each frame(time-consumption:0.3ms).
2. Simplify the construction of graph data to improce code efficiency

'''

import os 
import torch 
import numpy as np
from tqdm import tqdm
from loguru import logger 
from typing import Union,List,Dict
from torch_geometric.data import Data
from torchvision.io import read_image
from torch_geometric.data import Batch
import torchvision.transforms.functional as F 

__all__ = ['GraphDataset', 'graph_collate_fn']

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self,cfg,mode:str):
        super(GraphDataset, self).__init__()

        self.mode             = mode
        # self.device           = cfg.DEVICE
        self.resize_to_cnn    = cfg.RESIZE_TO_CNN
        self.trackback_window = cfg.TRACKBACK_WINDOW

        self.frame_list       = [] # STORE the seq-name and frame-idx 
        self.dets_dict        = {} # STORE the dets info for each frame
        self.dataset_dir      = cfg.DATA_DIR
        self.acceptable_obj_type  = cfg.ACCEPTABLE_OBJ_TYPE

        if self.mode == 'Train_full':
            self.seq_name_list    = [ seq + f'''-{cfg.DETECTOR}'''for seq in cfg.MOT17_TRAIN_NAME]
            self.start_frame_list = [1 for _ in self.seq_name_list]
            self.end_frame_list   = [ cfg.MOT17ALLFRAMENUM[i] for i in cfg.MOT17_TRAIN_NAME]
        elif self.mode == 'Train_split':
            self.seq_name_list    = [ seq + f'''-{cfg.DETECTOR}'''for seq in cfg.MOT17_TRAIN_NAME]
            self.start_frame_list = [1 for _ in self.seq_name_list]
            self.end_frame_list   = [i-1 for i in cfg.MOT17_VAL_START]
        elif self.mode == 'Validation': 
            self.seq_name_list    = [ seq + f'''-{cfg.DETECTOR}'''for seq in cfg.MOT17_TRAIN_NAME]
            self.start_frame_list = cfg.MOT17_VAL_START
            self.end_frame_list   = [ cfg.MOT17ALLFRAMENUM[i] for i in cfg.MOT17_TRAIN_NAME]
        elif self.mode == 'Test':
            pass
        else:
            raise RuntimeError('![MODE ERROR]')
        
        for seq_name,start_frame,end_frame in tqdm(
            zip(self.seq_name_list,self.start_frame_list,self.end_frame_list),
            desc=f"[{self.mode}] Data-preprocess",total = len(self.seq_name_list),unit='seq'):   # like seq_name - MOT17-04-FRCNN
            
            unique_frameidx_list = []
            self.dets_dict[seq_name] = {} 
            if self.mode == 'Test':
                txt_path = os.path.join(seq_path,'dt','dt.txt') # HOW TO CODE WHEN 'TEST' MODE ? 
                pass 
            seq_path   = os.path.join(self.dataset_dir,seq_name)
            txt_path   = os.path.join(seq_path,'gt','gt.txt') 
            detections = np.loadtxt(txt_path,delimiter=',')
            valid_mask = (
                (detections[:, 2] >= 0) & (detections[:, 3] >= 0) &
                # (detections[:, 4] >= 0) & (detections[:, 5] >= 0) &
                (np.isin(detections[:, 7], [1, 2, 7])) &
                (start_frame <= detections[:, 0]) & (detections[:, 0] <= end_frame)
            )
            sorted_detections = sorted(detections[valid_mask],key=lambda x:x[0])
            for detection in sorted_detections:
                x,y,w,h = map(float,detection[2:6])
                xc , yc = x + w/2 , y + h/2
                frame_idx,tracklet_id = map(int,detection[:2])
                if frame_idx not in unique_frameidx_list:
                    unique_frameidx_list.append(frame_idx)
                    self.frame_list.append(f'''{seq_name}*{frame_idx}''')
                    self.dets_dict[seq_name][frame_idx] = []
                self.dets_dict[seq_name][frame_idx].append([frame_idx,tracklet_id,x,y,w,h,xc,yc])
            if self.mode !=  'Validation':
                self.frame_list.remove(f"{seq_name}*{1}")
        logger.info(f"[{self.mode}] Data-preprocess complete!")
        logger.info(f"[{self.mode}] Total frame number : {len(self.frame_list)}")

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self,idx:int):
        # print(self.frame_list[idx])
        seq_name , current_frame = self.frame_list[idx].split('*')[0], int(self.frame_list[idx].split('*')[1])

        imgs_dir = os.path.join(self.dataset_dir,seq_name,'img1')
        tracklets_dict     = {}
        current_detections = self.dets_dict[seq_name][current_frame]
        cut_from_frame = max(1,current_frame - self.trackback_window)
        for frame_idx in range(cut_from_frame,current_frame):
            past_detections = self.dets_dict[seq_name][frame_idx]
            for past_det in past_detections: # past_det = [frame_idx,tracklet_id,x,y,w,h,xc,yc]
                tracklet_id = past_det[1]
                if tracklet_id not in tracklets_dict:
                    tracklets_dict[tracklet_id] = []
                tracklets_dict[tracklet_id].append(past_det)
        
        tra_graph = self.construct_raw_graph(tracklets_dict,imgs_dir,is_tracklet=True)
        det_graph = self.construct_raw_graph(current_detections,imgs_dir,is_tracklet=False)
        gt_matrix = self.construct_label(current_detections,tracklets_dict)

        return tra_graph,det_graph,gt_matrix
    
    def construct_raw_graph(self,detections:Union[List,Dict],imgs_dir,is_tracklet:bool=False):
        prev_frame_idx = None
        raw_node_attr , location_info = [] , []      
        if is_tracklet:
            # fetch the last item of each tracklet
            detections = [item[-1] for item in list(detections.values())]         
        for det in detections: # det = [frame_idx,tracklet_id,x,y,w,h,xc,yc]
            frame_idx = int(det[0])
            x,y , w,h = map(int,det[2:6])
            xc , yc   = map(float,det[6:])
            if frame_idx != prev_frame_idx:
                prev_frame_idx = frame_idx
                im_path = os.path.join(imgs_dir, f"{frame_idx:06d}.jpg")
                # im_tensor = read_image(im_path).to(torch.float32)
                im_tensor = read_image(im_path).to(torch.float32)
                im_tensor = F.normalize(im_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
            patch = F.crop(im_tensor,y,x,h,w)
            patch = F.resize(patch,self.resize_to_cnn)
            raw_node_attr.append(patch)
            location_info.append([xc,yc,w,h])   # STORE  xc yc  w h
        raw_node_attr = torch.stack(raw_node_attr,dim=0)
        # locate_info = torch.as_tensor(locate_info,dtype=torch.float32)
        location_info = torch.as_tensor(location_info,dtype=torch.float32)
        return Data(x=raw_node_attr,location_info=location_info)
    
    def construct_label(self,current_detections,tracklets_dict):
        '''
        Return a matrix where each column represents a tracklet and each row represents a detection.
        '''
        n_dets = len(current_detections)
        n_tras = len(tracklets_dict.keys())
        # gt_matrix_aug = torch.zeros(n_tras + 1,n_dets + 1)
        # gt_matrix = torch.zeros(n_tras,n_dets,dtype=torch.float32)
        gt_matrix = torch.zeros(n_tras,n_dets,dtype=torch.float32)
        for i,id in enumerate(tracklets_dict.keys()):
            for j,detection in enumerate(current_detections):
                if id == detection[1]:  # the same id
                    gt_matrix[i,j] = 1  
                    break               
        return gt_matrix
    
def graph_collate_fn(batch):
    """
    Custom collate function for DataLoader to handle PyTorch Geometric data.
    """
    tra_graphs, det_graphs, gt_matrices = zip(*batch)

    # Batch 'det_graph' and 'tra_graph' using PyTorch Geometric's Batch class
    tra_graph_batch = Batch.from_data_list(list(tra_graphs))
    det_graph_batch = Batch.from_data_list(list(det_graphs))
    
    return tra_graph_batch, det_graph_batch, list(gt_matrices)
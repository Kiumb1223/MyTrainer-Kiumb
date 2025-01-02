#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

1. By maintaining a lookup table(self.dets_dict), 
we can quickily retrive the detections infomation for each frame(time-consumption:0.3ms).
2. Simplify the construction of graph data to improce code efficiency

'''

import os 
import json
import torch 
import numpy as np
from tqdm import tqdm
from loguru import logger 
import torchvision.io.image as I
from torch_geometric.data import Data
from torch_geometric.data import Batch
from typing import Union,List,Dict,Tuple
import torchvision.transforms.functional as T

__all__ = ['GraphDataset', 'graph_collate_fn']

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self,cfg,mode:str,bt_augmentation:bool = False):
        super(GraphDataset, self).__init__()

        self.mode             = mode
        # self.device           = cfg.DEVICE
        self.resize_to_cnn    = cfg.RESIZE_TO_CNN
        self.trackback_window = cfg.TRACKBACK_WINDOW

        self.frame_list       = [] # STORE the seq-name and frame-idx 
        self.dets_dict        = {} # STORE the dets info for each frame
        self.start_frame_idx  = {} # STORE the start-frame-idx for each seq
        self.dataset_dir      = cfg.DATA_DIR
        self.acceptable_obj_type  = cfg.ACCEPTABLE_OBJ_TYPE

        # data augmentation settings
        self.bt_augmentation  = bt_augmentation
        self.min_ids_to_drop_perc  = cfg.MIN_IDS_TO_DROP_PERC
        self.max_ids_to_drop_perc  = cfg.MAX_IDS_TO_DROP_PERC
        self.min_dets_to_drop_perc = cfg.MIN_DETS_TO_DROP_PERC
        self.max_dets_to_drop_perc = cfg.MAX_DETS_TO_DROP_PERC

        with open(cfg.JSON_PATH,'r') as f:
            if self.mode == 'Train':
                data_json = json.load(f)['train_seq']
            elif self.mode == 'Validation':
                data_json = json.load(f)['valid_seq']
            else:
                raise ValueError('![MODE ERROR]')
        
        for dataset_name in tqdm(data_json.keys(),desc=f"[{self.mode}] Data-preprocess",total=len(data_json.keys()),unit='dataset'):
            
            seq_name_list    = data_json[dataset_name]['seq_name']
            start_frame_list = data_json[dataset_name]['start_frame']
            end_frame_list   = data_json[dataset_name]['end_frame']

            for seq_name,start_frame,end_frame in zip(seq_name_list,start_frame_list,end_frame_list):                
                unique_frameidx_list = []
                self.dets_dict[seq_name] = {} 

                if self.mode == 'Train' or dataset_name in ['MOT17','MOT20']:
                    seq_path   = os.path.join(self.dataset_dir,dataset_name,'train',seq_name)  
                elif self.mode == 'Validation' and dataset_name in ['DanceTrack']:
                    seq_path   = os.path.join(self.dataset_dir,dataset_name,'val',seq_name)                         
                
                txt_path   = os.path.join(seq_path,'gt','gt.txt') 
                detections = np.loadtxt(txt_path,delimiter=',')
                valid_mask = (
                    # (detections[:, 2] >= 0) & (detections[:, 3] >= 0) &
                    # (detections[:, 4] >= 0) & (detections[:, 5] >= 0) &
                    (np.isin(detections[:, 7], [1, 2, 7])) &
                    (start_frame <= detections[:, 0]) & (detections[:, 0] <= end_frame)
                )
                sorted_detections = sorted(detections[valid_mask],key=lambda x:x[0])
                for detection in sorted_detections:
                    x,y,w,h = map(float,detection[2:6])
                    x2,y2,xc,yc = x + w , y + h, x + w/2 , y + h/2
                    frame_idx,tracklet_id = map(int,detection[:2])
                    if frame_idx not in unique_frameidx_list:
                        unique_frameidx_list.append(frame_idx)
                        self.frame_list.append(f"{dataset_name}#{seq_name}#{frame_idx}")
                        self.dets_dict[seq_name][frame_idx] = []
                    self.dets_dict[seq_name][frame_idx].append([frame_idx,tracklet_id,x,y,x2,y2,w,h,xc,yc])
                # if self.mode !=  'Validation':
                self.frame_list.remove(f"{dataset_name}#{seq_name}#{start_frame}")
            
            self.start_frame_idx[dataset_name] = {
                seq_name: start_frame
                for seq_name, start_frame in zip(seq_name_list, start_frame_list)
            }
            
        logger.info(f"[{self.mode}] Total frame number : {len(self.frame_list)}")

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self,idx:int):
        # print(self.frame_list[idx])
        dataset_name,seq_name , current_frame = self.frame_list[idx].split('#')[0],\
            self.frame_list[idx].split('#')[1], int(self.frame_list[idx].split('#')[2])
        if self.mode == 'Train' or dataset_name in ['MOT17','MOT20']:
            imgs_dir = os.path.join(self.dataset_dir,dataset_name,'train',seq_name,'img1')
        elif self.mode == 'Validation' and dataset_name in 'DanceTrack':
            imgs_dir = os.path.join(self.dataset_dir,dataset_name,'val',seq_name,'img1')

        tracklets_dict,current_detections = self._dataAugment(dataset_name,seq_name,current_frame)
        
        tra_graph = self.construct_raw_graph(tracklets_dict,dataset_name,imgs_dir,is_tracklet=True)
        det_graph = self.construct_raw_graph(current_detections,dataset_name,imgs_dir,is_tracklet=False)
        gt_matrix = self.construct_label(current_detections,tracklets_dict)

        return tra_graph,det_graph,gt_matrix
    
    def construct_raw_graph(self,detections:Union[List,Dict],date_type,imgs_dir,is_tracklet:bool=False):
        prev_frame_idx = None
        raw_node_attr , location_info = [] , []      
        if is_tracklet:
            # fetch the last item of each tracklet
            detections = [item[-1] for item in list(detections.values())]         
        for det in detections: # det = [frame_idx,tracklet_id,x,y,x2,y2,w,h,xc,yc]
            frame_idx   = int(det[0])
            x,y,_,_,w,h = map(int,det[2:-2])
            # xc , yc   = map(float,det[6:])
            if frame_idx != prev_frame_idx:
                prev_frame_idx = frame_idx
                im_path = os.path.join(imgs_dir, f"{frame_idx:06d}.jpg" if date_type in ['MOT17','MOT20'] else f"{frame_idx:08d}.jpg")
                im_tensor = I.read_image(im_path).to(torch.float32)
                H , W = im_tensor.shape[1:] 
            
            if x < 0:
                w = w + x  
                x = 0 

            if y < 0:
                h = h + y  
                y = 0  
                
            w = min(w, im_tensor.shape[2] - x)  
            h = min(h, im_tensor.shape[1] - y)
            
            patch = T.crop(im_tensor,y,x,h,w)
            patch = T.resize(patch,self.resize_to_cnn)
            raw_node_attr.append(patch)
            location_info.append(det[2:] + [W,H])   # STORE x,y,x2,y2,w,h,xc,yc ,W,H
        raw_node_attr = torch.stack(raw_node_attr,dim=0)
        location_info = torch.as_tensor(location_info,dtype=torch.float32)
        return Data(x=raw_node_attr,location_info=location_info)
    
    def construct_label(self,current_detections,tracklets_dict):
        '''
        Return a matrix where each column represents a tracklet and each row represents a detection.
        '''
        n_dets = len(current_detections)
        n_tras = len(tracklets_dict.keys())
        gt_matrix = torch.zeros(n_tras,n_dets,dtype=torch.float32)
        for i,id in enumerate(tracklets_dict.keys()):
            for j,detection in enumerate(current_detections):
                if id == detection[1]:  # the same id
                    gt_matrix[i,j] = 1  
                    break               
        return gt_matrix
    
    def _dataAugment(self,dataset_name,seq_name,current_frame) -> Tuple[Dict,List]:
        '''
        Reference : https://github.com/dvl-tum/SUSHI/blob/main/src/data/augmentation.py
        '''
        tracklets_dict     = {}
        current_detections = self.dets_dict[seq_name][current_frame]
        cut_from_frame = max(self.start_frame_idx[dataset_name][seq_name],current_frame - self.trackback_window)

        if self.bt_augmentation:
            # simulate low framerate
            cut_to_frame = max(cut_from_frame+1,current_frame -  np.random.randint(0,5)+1)
        else:
            cut_to_frame = current_frame
        # print(f"{cut_from_frame}->{cut_to_frame}")
        for frame_idx in range(cut_from_frame,cut_to_frame):
            past_detections = self.dets_dict[seq_name][frame_idx]
            for past_det in past_detections: # past_det = [frame_idx,tracklet_id,x,y,w,h,xc,yc]
                tracklet_id = past_det[1]
                if tracklet_id not in tracklets_dict:
                    tracklets_dict[tracklet_id] = []
                tracklets_dict[tracklet_id].append(past_det)
        
        if self.bt_augmentation:
            # simulate missed detections and discontinuous trajectories
            ids_to_drop_perc = np.random.uniform(self.min_ids_to_drop_perc,self.max_ids_to_drop_perc)
            detects_to_drop_perc = np.random.uniform(self.min_dets_to_drop_perc,self.max_dets_to_drop_perc)

            num_ids_to_drop  = np.round(ids_to_drop_perc * len(tracklets_dict.keys())).astype(int)
            num_dets_to_drop = np.round(detects_to_drop_perc * len(current_detections)).astype(int)

            if num_ids_to_drop > 0:
                # drop ids
                ids_to_drop = np.random.choice(list(tracklets_dict.keys()),num_ids_to_drop,replace=False)
                tracklet_ids = np.array(list(tracklets_dict.keys()))
                mask = ~ np.isin(tracklet_ids,ids_to_drop)
                keys_to_keep = tracklet_ids[mask]
                tracklets_dict = {
                    key:tracklets_dict[key] for key in keys_to_keep
                }
            if num_dets_to_drop > 0 :
                # drop dets
                dets_to_drop = np.random.choice(list(range(len(current_detections))),num_dets_to_drop,replace=False)
                current_detections = np.array(current_detections)    
                mask = np.ones(len(current_detections),dtype=bool)
                mask[dets_to_drop] = False
                current_detections = current_detections[mask].tolist()

        return tracklets_dict,current_detections

def graph_collate_fn(batch):
    """
    Custom collate function for DataLoader to handle PyTorch Geometric data.
    """
    tra_graphs, det_graphs, gt_matrices = zip(*batch)

    # Batch 'det_graph' and 'tra_graph' using PyTorch Geometric's Batch class
    tra_graph_batch = Batch.from_data_list(list(tra_graphs))
    det_graph_batch = Batch.from_data_list(list(det_graphs))
    
    return tra_graph_batch, det_graph_batch, list(gt_matrices)
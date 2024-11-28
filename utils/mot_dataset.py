#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
By maintaining a lookup table(self.dets_dict),
we can quickily retrive the detections infomation for each frame(time-consumption:0.3ms).
This allows us to efficiently construct the trajectory and detection graphs(time-consumption:0.6s)
and ground-truth data(time-consumption:0.3ms)
'''

import os
import torch
import numpy as np 
from PIL import Image
from tqdm import tqdm

from loguru import logger
from functools import partial
from torch_cluster import knn
# import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.data import Batch
# from torch_geometric.data import Data,Dataset
from torch.utils.data.dataset import Dataset
from torch_geometric.utils import remove_self_loops
from torchvision.transforms import Compose,Resize,ToTensor,Normalize
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torchvision.transforms.functional as F

__all__ = ['MOTGraph','graph_collate_fn']

class MOTGraph(Dataset):
    def __init__(self,cfg,mode):

        self.k_neighbor       = cfg['K_NEIGHBOR'] + 1 # Add 1 to K_neighbor cuz KNN includes the node itself in the neighbor count.
        self.trackback_window = cfg['TRACKBACK_WINDOW']
        # self.transform        = Compose((Resize(cfg['RESIZE_TO_CNN']), ToTensor(), 
        #                                  Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])))
        self.resize_to_cnn = cfg['RESIZE_TO_CNN']
        self.device        = cfg['DEVICE']
        #---------------------------------#
        #  dets_dict -  {seq_name : {frame_idx:[frame_idx,tra_id,x,y,w,h,xc,yc],....},....}
        #---------------------------------#
        self.frame_list       = [] # STORE the seq-name and frame-idx 
        self.dets_dict        = {} # STORE the dets info for each frame
        self.mode             = mode
        self.dataset_dir      = cfg['DATA_DIR']
        # self.preprocess_dir   = os.path.join(cfg_data['DATA_DIR'],'preprocess')
        self.acceptable_obj_type  = cfg['ACCEPTABLE_OBJ_TYPE']

        if self.mode == 'Train_full':
            self.seq_name_list    = [ seq + f'''-{cfg['DETECTOR']}'''for seq in cfg['MOT17_TRAIN_NAME']]
            self.start_frame_list = [1 for _ in self.seq_name_list]
            self.end_frame_list   = [ cfg['MOT17ALLFRAMENUM'][i] for i in cfg['MOT17_TRAIN_NAME']]
        elif self.mode == 'Train_split':
            self.seq_name_list    = [ seq + f'''-{cfg['DETECTOR']}'''for seq in cfg['MOT17_TRAIN_NAME']]
            self.start_frame_list = [1 for _ in self.seq_name_list]
            self.end_frame_list   = [i-1 for i in cfg['MOT17_VAL_START']]
        elif self.mode == 'Validation': 
            self.seq_name_list    = [ seq + f'''-{cfg['DETECTOR']}'''for seq in cfg['MOT17_TRAIN_NAME']]
            self.start_frame_list = cfg['MOT17_VAL_START']
            self.end_frame_list   = [ cfg['MOT17ALLFRAMENUM'][i] for i in cfg['MOT17_TRAIN_NAME']]
        elif self.mode == 'Test':
            pass
        else:
            raise RuntimeError('![MODE ERROR]')
        
        for seq_name,start_frame,end_frame in zip(self.seq_name_list,self.start_frame_list,self.end_frame_list):   # like seq_name - MOT17-04-FRCNN
            unique_frameidx_list = []
            self.dets_dict[seq_name] = {} 
            if self.mode == 'Test':
                txt_path = os.path.join(seq_path,'dt','dt.txt') # HOW TO CODE WHEN 'TEST' MODE ? 
                pass 
            seq_path   = os.path.join(self.dataset_dir,seq_name)
            txt_path   = os.path.join(seq_path,'gt','gt.txt') 
            detections = np.loadtxt(txt_path,delimiter=',')
            valid_mask = (
                (detections[:, 2] > 0) & (detections[:, 3] > 0) &
                (detections[:, 4] > 0) & (detections[:, 5] > 0) &
                (np.isin(detections[:, 7], [1, 2, 7])) &
                (start_frame <= detections[:, 0]) & (detections[:, 0] <= end_frame)
            )
            sorted_detections = sorted(detections[valid_mask],key=lambda x:x[0])
            for detection in sorted_detections:
                x,y,w,h = map(float,detection[2:6])
                xc , yc = x + w/2 , y + h/2
                frame_idx,tracklet_id = int(detection[0]),int(detection[1])
                if frame_idx not in unique_frameidx_list:
                    unique_frameidx_list.append(frame_idx)
                    self.frame_list.append(f'''{seq_name}*{frame_idx}''')
                    self.dets_dict[seq_name][frame_idx] = []
                self.dets_dict[seq_name][frame_idx].append([frame_idx,tracklet_id,x,y,w,h,xc,yc])
            if self.mode !=  'Validation':
                self.frame_list.remove(f"{seq_name}*{1}")
    def __len__(self):
        return len(self.frame_list)
    
    def __getitem__(self,idx):
        seq_name , current_frame = self.frame_list[idx].split('*')[0] , int(self.frame_list[idx].split('*')[-1])

        #---------------------------------#
        # Construct PATH based on  `seq_name` 
        #---------------------------------#
        seq_path  = os.path.join(self.dataset_dir,seq_name)
        image_dir = os.path.join(seq_path,'img1') 
        #---------------------------------#
        # Load detections from 'self.dets_dict'
        #   - Detection Graph: frame_idx  = current_frame   
        #   - Tracklet  Graph: frame_idx  = [current_frame - trackback_window,current_frame] 
        #       * IF current_frame - trackback_window <= 0 , start from frame 1 
        #---------------------------------#
        current_detections,tracklets_dict = [],{}
        current_detections = self.dets_dict[seq_name][current_frame]
        cut_from_frame   = max(1,current_frame - self.trackback_window)
        for frame_idx in range(cut_from_frame,current_frame):
            past_detections = self.dets_dict[seq_name][frame_idx]
            for past_detection in past_detections:
                tracklet_id = past_detection[1]
                if tracklet_id not in tracklets_dict:
                    tracklets_dict[tracklet_id] = []
                tracklets_dict[tracklet_id].append(past_detection)
        
        #---------------------------------#
        # Construct detection graph & tracklet graph
        # based on `current_detections` and `tracklets_dict`
        #---------------------------------#
        detection_graph , tracklet_graph = self.construct_graph(current_detections,tracklets_dict,self.k_neighbor,image_dir)
        #---------------------------------#
        # Construct gt_matrix_aug
        # based on `current_detections` and `tracklets_dict`
        #---------------------------------#
        gt_matrix = self.construct_label(current_detections,tracklets_dict)

        return {
            'det_graph':detection_graph.to(self.device),
            'tra_graph':tracklet_graph.to(self.device),
            'gt_matrix':gt_matrix.to(self.device)
        }

    # @staticmethod
    def construct_graph(self,current_detections, tracklets_dict,k, image_dir):

        def compute_edge_attr(nodes, edge_indices,is_tracklet=False):
            """Compute edge attributes for a graph."""
            edge_attr = []
            nodes = [node[-1] for node in nodes] if is_tracklet else nodes
            for idx in range(len(edge_indices[0])):
                n1, n2 = edge_indices[0][idx], edge_indices[1][idx]
                w1, h1 , xc1 , yc1 = nodes[n1][4:8]
                w2, h2 , xc2 , yc2 = nodes[n2][4:8]
                edge_attr.append([
                    2 * (xc1 - xc2) / (h1 + h2),
                    2 * (yc1 - yc2) / (h1 + h2),
                    np.log(h1 / h2),
                    np.log(w1 / w2)
                ])
            return torch.as_tensor(edge_attr, dtype=torch.float32)

        def build_graph(detections,k, is_tracklet=False):
            """Generic graph construction function."""
            node_attr, positions = [], [] 
            detections = list(detections.values()) if is_tracklet else detections
            for detection in detections:
                item = detection[-1] if is_tracklet else detection
                # xmin, ymin, width, height = map(int, map(round, item[2:6]))
                frame_id,_,x,y,w,h,xc,yc = map(int,item[0:8])
                prev_frame_idx = None
                if prev_frame_idx != frame_id:
                    image_name = os.path.join(image_dir, f"{int(item[0]):06d}" + '.jpg')
                    image_tensor = read_image(image_name).to(torch.float32).to(self.device)
                    image_tensor = F.normalize(image_tensor,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                patch = F.crop(image_tensor,y,x,h,w)
                patch = F.resize(patch,size=self.resize_to_cnn)
                positions.append([xc,yc])
                node_attr.append(patch)
            
            positions = torch.as_tensor(positions, dtype=torch.float32)
            edge_indices = knn(positions, positions, k=k,num_workers=2)
            edge_indices, _ = remove_self_loops(edge_indices)
            edge_attr = compute_edge_attr(detections, edge_indices,is_tracklet)
            node_attr = torch.stack(node_attr)
            # edge_indices = torch.tensor(edge_indices, dtype=torch.long)

            return Data(x=node_attr, edge_index=edge_indices, edge_attr=edge_attr, pos=positions)

        # Construct detection and tracklet graphs
        detection_graph = build_graph(current_detections,k)
        tracklet_graph = build_graph(tracklets_dict,k, is_tracklet=True)

        return detection_graph, tracklet_graph

    def construct_label(self,current_detections,tracklets_dict):
        '''
        Return a matrix where each column represents a tracklet and each row represents a detection.
        '''
        n_dets = len(current_detections)
        n_tras = len(tracklets_dict.keys())
        # gt_matrix_aug = torch.zeros(n_tras + 1,n_dets + 1)
        gt_matrix = torch.zeros(n_tras,n_dets)
        for i,id in enumerate(tracklets_dict.keys()):
            for j,detection in enumerate(current_detections):
                if id == detection[1]:  # the same object
                    gt_matrix[i,j] = 1  
                    break               
        return gt_matrix

def graph_collate_fn(batch):
    """
    Custom collate function for DataLoader to handle PyTorch Geometric data.
    """
    # Separate out 'det_graph', 'tra_graph', and 'gt_matrix' from the batch
    det_graph_batch = [item['det_graph'] for item in batch]
    tra_graph_batch = [item['tra_graph'] for item in batch]
    gt_matrix_batch = [item['gt_matrix'] for item in batch]
    
    # Batch 'det_graph' and 'tra_graph' using PyTorch Geometric's Batch class
    det_graph_batch = Batch.from_data_list(det_graph_batch)
    tra_graph_batch = Batch.from_data_list(tra_graph_batch)
    
    # # Stack gt_matrix_batch into a tensor
    # gt_matrix_batch = torch.stack(gt_matrix_batch, dim=0)
    
    return det_graph_batch, tra_graph_batch, gt_matrix_batch


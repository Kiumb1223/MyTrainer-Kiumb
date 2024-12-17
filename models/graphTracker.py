#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     graphTracker.py
@Time     :     2024/12/01 14:31:31
@Author   :     Louis Swift
@Desc     :     
'''
import gc
import torch
import numpy as np
from loguru import logger
from typing import Tuple,List
from enum import Enum,unique,auto
from torch_geometric.data import Data
from models.graphModel import GraphModel
import torchvision.transforms.functional as T
from models.graphToolkit import hungarian,box_iou

__all__ = ['TrackManager']

@unique
class LifeSpan(Enum):
    '''the lifespace of each tracker,for trajectory management'''
    Born  = auto()
    Active= auto()
    Sleep = auto()
    Dead  = auto()

class Tracker:
    '''the tracker class'''

    _track_id = 0

    def __init__(self,start_frame,appearance_feat,conf,tlwh,cnt_to_active,cnt_to_sleep,max_cnt_to_dead,feature_list_size):
        
        self.track_id   = None # when state: Born to Active, this will be assigned
        self.track_len  = 0
        self.sleep_cnt  = 0 
        self.active_cnt = 0 
        self.state      = LifeSpan.Born

        self.start_frame   = start_frame 

        self.conf = conf
        self.tlwh = tlwh # (top left x, top left y, width, height)
        self.appearance_feats_list = []
        self.appearance_feats_list.append(appearance_feat) 
        
        self._cnt_to_active     = cnt_to_active
        self._cnt_to_sleep      = cnt_to_sleep
        self._max_cnt_to_dead   = max_cnt_to_dead  
        self._feature_list_size = feature_list_size

    def to_active(self,frame_idx,appearance_feat,conf,tlwh):
        
        assert appearance_feat.shape[-1] == 32 , f'plz confirm the feature size is 32, but got {appearance_feat.shape}'
        if self.state  == LifeSpan.Born:
            age = frame_idx - self.start_frame
            if age >= self._cnt_to_active:
                self.state = LifeSpan.Active
                self.track_id = Tracker.get_track_id()
                # del self._cnt_to_active
        elif self.state == LifeSpan.Sleep:
            self.track_len = 0
            self.sleep_cnt = 0
            self.state     = LifeSpan.Active
        else:
            self.state = LifeSpan.Active
        
        self.active_cnt = 0
        self.track_len += 1 
        self.frame_idx = frame_idx
        self.conf = conf
        self.tlwh = tlwh
        self.appearance_feats_list.append(appearance_feat)

        if len(self.appearance_feats_list) > self._feature_list_size:
            expired_feat = self.appearance_feats_list.pop(0)
            del expired_feat

    def to_sleep(self):  
        if self.state == LifeSpan.Born:
            self.state = LifeSpan.Dead
            return 
        
        if self.state == LifeSpan.Active:
            self.active_cnt += 1
            if self.active_cnt >= self._cnt_to_sleep:
                self.state = LifeSpan.Sleep
            return
        
        self.sleep_cnt += 1
        if self.sleep_cnt >= self._max_cnt_to_dead:
            self.state = LifeSpan.Dead
    
    @property
    def end_frame(self) -> int:
        '''Returns the frame_idx of the object'''
        return self.frame_idx

    @property
    def is_Born(self) -> bool:
        '''Returns True if the object's state is Born'''
        return self.state == LifeSpan.Born

    @property
    def is_Active(self) -> bool:
        '''Returns True if the object's state is Active'''
        return self.state == LifeSpan.Active

    @property
    def is_Sleep(self) -> bool:
        '''Returns True if the object's state is Sleep'''
        return self.state == LifeSpan.Sleep
    
    @property
    def is_Dead(self) -> bool:
        '''Returns True if the object's state is Dead'''
        return self.state == LifeSpan.Dead
    
    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @property
    def location_info(self):
        """Convert bounding box to format `(min x, min y, max x, max y, width, height,center x, center y,)`"""
        ret = self.tlwh.copy().astype(np.float32)
        ret = np.append(ret,self.tlwh[-2:])
        ret = np.append(ret,self.tlwh[-2:])
        ret[2:4] = ret[2:4] + ret[:2]
        ret[-2:] = ret[-2:] / 2+ ret[:2]
        return ret
    
    @staticmethod
    def get_track_id():
        '''get a unique track id'''
        Tracker._track_id += 1
        return Tracker._track_id
    
    @staticmethod
    def clean_cache():
        Tracker._track_id = 0

    def __repr__(self) -> str:
        return f"Tracker(id - {self.track_id} || from {self.start_frame} to {self.end_frame})"

class TrackManager:
    def __init__(self,model :GraphModel,device :str,path_to_weights :str,
                 resize_to_cnn :list =[224,224],match_thresh :float =0.1,det2tra_conf :float =0.7,
                 cnt_to_active :int =3,cnt_to_sleep :int=10,max_cnt_to_dead :int =100,feature_list_size :int =10):
        
        self.device = device
        self.model  = model.eval().to(device)

        self.tracks_list:List[Tracker] = [] # store all the tracks including Born, Active, Sleep, Dead

        self._resize_to_cnn   = resize_to_cnn
        self._match_thresh    = match_thresh
        # necessary attributes when initializing the single track
        self._det2tra_conf    = det2tra_conf
        self._cnt_to_active   = cnt_to_active
        self._cnt_to_sleep    = cnt_to_sleep
        self._max_cnt_to_dead = max_cnt_to_dead
        self._feature_list_size = feature_list_size 

        if path_to_weights:
            try:
                self.model.load_state_dict(torch.load(path_to_weights,map_location='cpu')['model'])
            except KeyError:
                self.model.load_state_dict(torch.load(path_to_weights,map_location='cpu'))
            finally:
                logger.info(f"Load weights from {path_to_weights} successfully")
                
    @torch.no_grad()
    def update(self,frame_idx:int,current_detections:np.ndarray,img_date:torch.Tensor) -> List[Tracker]:
        '''
        current_detections =np.ndarray(tlwh,conf)  and have already filtered by conf > 0.1 
        '''
        output_track_list = []
        first_tracks_list = [track for track in self.tracks_list if track.is_Active ]
        # first_tracks_list = [track for track in self.tracks_list if track.is_Active or track.is_Sleep]
        tra_graph  = self.construct_tra_graph(first_tracks_list)
        det_graph  = self.construct_det_graph(current_detections,img_date)
        match_mtx,match_idx,unmatch_tra,unmatch_det = self._graph_match(tra_graph,det_graph)

        # The input `det_graph` is modified inside `self.model`, 
        # so its state changes after the function call.
        if match_idx != []:         # matched tras and dets 
            tra_idx ,det_idx = match_idx
            for tra_id, det_id in zip(tra_idx,det_idx):
                first_tracks_list[tra_id].to_active(frame_idx,det_graph.x[det_id],
                                current_detections[det_id][4],current_detections[det_id][:4])
                if not first_tracks_list[tra_id].is_Born:
                    output_track_list.append(first_tracks_list[tra_id])        
        for tra_id in unmatch_tra:   # unmatched tras 
            first_tracks_list[tra_id].to_sleep()
        first_tracks_list = self.remove_dead_tracks(first_tracks_list)

        second_tracks_list = [track for track in self.tracks_list if track.is_Born]
        highconf_unmatch_dets = current_detections[unmatch_det][current_detections[unmatch_det,4] >= self._det2tra_conf]
        highconf_to_global_det_idx = {i: unmatch_det[i] for i in range(len(highconf_unmatch_dets))}

        match_mtx,match_idx,unmatch_tra,unmatch_det = self._iou_match(second_tracks_list,highconf_unmatch_dets.copy())
        if match_idx != []:         # matched tras and dets 
            tra_idx ,det_idx = match_idx
            for tra_id, det_id in zip(tra_idx,det_idx):
                global_id = highconf_to_global_det_idx[det_id]
                second_tracks_list[tra_id].to_active(frame_idx,det_graph.x[global_id],
                                highconf_unmatch_dets[det_id][4],highconf_unmatch_dets[det_id][:4])
                if not second_tracks_list[tra_id].is_Born:
                    output_track_list.append(second_tracks_list[tra_id])
        for tra_id in unmatch_tra:# unmatched tras 
            second_tracks_list[tra_id].to_sleep()
        second_tracks_list = self.remove_dead_tracks(second_tracks_list)
        for det_id in unmatch_det:
            global_id = highconf_to_global_det_idx[det_id]
            second_tracks_list.append(
                Tracker(frame_idx,det_graph.x[global_id],
                        highconf_unmatch_dets[det_id][4],highconf_unmatch_dets[det_id][:4],
                        self._cnt_to_active,self._cnt_to_sleep,self._max_cnt_to_dead,self._feature_list_size)
            )
        
        self.tracks_list = first_tracks_list + second_tracks_list
        return output_track_list

    def construct_tra_graph(self,tracks_list:List[Tracker]) -> Data:
        '''construct graph of tracks including ACTIVE'''
        if not tracks_list: # if no tracks 
            return Data(num_nodes=0)
        
        node_attr , location_info = [] , []
        for track in tracks_list:
            node_attr.append(track.appearance_feats_list[-1])
            location_info.append(track.location_info)
        node_attr = torch.stack(node_attr,dim=0).to(self.device)
        location_info = torch.as_tensor(location_info,dtype=torch.float32).to(self.device)
        return Data(x=node_attr,location_info=location_info)
    
    def construct_det_graph(self,current_detections:np.ndarray,img_date:torch.Tensor) -> Data:
        '''construct raw graph of detections'''
        img_tensor  = img_date.to(self.device).to(torch.float32) / 255.0
        raw_node_attr , location_info = [] , []
        im_tensor = T.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for det in current_detections:
            x,y , w,h = map(int,det[:4])
            xc , yc   = x + w/2 , y + h/2
            x2 , y2   = x + w   , y + h
            if x < 0:
                w = w + x  
                x = 0 

            if y < 0:
                h = h + y  
                y = 0  
                
            w = min(w, im_tensor.shape[2] - x)  
            h = min(h, im_tensor.shape[1] - y)

            patch = T.crop(im_tensor,y,x,h,w)
            patch = T.resize(patch,self._resize_to_cnn)
            raw_node_attr.append(patch)
            location_info.append([x,y,x2,y2,w,h,xc,yc])  # STORE x,y,x2,y2,w,h,xc,yc
        raw_node_attr = torch.stack(raw_node_attr,dim=0).to(self.device)
        location_info = torch.as_tensor(location_info,dtype=torch.float32).to(self.device)
        return Data(x=raw_node_attr,location_info=location_info)

    def _graph_match(self,tra_graph:Data,det_graph:Data):
        ''' first phase to match via graph model'''
        pred_mtx = self.model(tra_graph.to(self.device),det_graph.to(self.device))
        match_mtx,match_idx,unmatch_tra,unmatch_det = hungarian(pred_mtx[:-1,:-1].cpu().numpy(),self._match_thresh)
        return match_mtx,match_idx,unmatch_tra,unmatch_det

    def _iou_match(self,tracks_list,highconf_unmatch_dets:np.ndarray):      
        ''' second phase to match via IOU'''
        if tracks_list == []:
            tras_tlbr = np.array([])
        else:
            tras_tlbr = np.vstack([
                track.tlbr for track in tracks_list 
            ],dtype=np.float32)

        dets_tlbr = highconf_unmatch_dets[:,:4]
        dets_tlbr[:,2:] = dets_tlbr[:,2:] + dets_tlbr[:,:2]

        iou  = box_iou(tras_tlbr,dets_tlbr)
        match_mtx,match_idx,unmatch_tra,unmatch_det = hungarian(iou,0.1)

        return match_mtx,match_idx,unmatch_tra,unmatch_det

    def remove_dead_tracks(self,tracks_list):
        """Remove all trackers whose state is Dead"""
        return [track for track in tracks_list if not track.is_Dead]

    def clean_cache(self):
        '''clean cache of all tracks'''
        self.tracks_list.clear()
        Tracker.clean_cache()
        gc.collect()
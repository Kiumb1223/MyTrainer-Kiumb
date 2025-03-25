#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     graphTracker.py
@Time     :     2024/12/01 14:31:31
@Author   :     Louis Swift
@Desc     :     
'''

import gc
import math
import yaml
import torch
import numpy as np
from loguru import logger
from typing import Union,List
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

    def __init__(self,
            start_frame:int,
            app_feat:torch.Tensor,
            conf:float,
            geometric_info:np.ndarray,
            cnt_to_active:int,
            cnt_to_sleep:int,
            max_cnt_to_dead:int,
            feature_list_size:int
        ):
        
        self.track_id   = None # when state: Born to Active, this will be assigned
        self.track_len  = 0
        self.sleep_cnt  = 0 
        self.active_cnt = 0 
        self.state      = LifeSpan.Born

        self.start_frame   = start_frame 
        self.frame_idx     = start_frame

        self.conf = conf
        # self.tlwh = tlwh # (top left x, top left y, width, height)
        self.geometric_info  = geometric_info

        self.app_feats_list  = []
        self.app_feats_list.append(app_feat) 
        
        self._cnt_to_active     = cnt_to_active
        self._cnt_to_sleep      = cnt_to_sleep
        self._max_cnt_to_dead   = max_cnt_to_dead  
        self._feature_list_size = feature_list_size

    def to_active(self,frame_idx,app_feat,conf,geometric_info):
        assert app_feat.shape[-1] == 32 , f'plz confirm the feature size is 32, but got {app_feat.shape}'
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
        # self.tlwh = tlwh
        self.geometric_info = geometric_info
        self.app_feats_list.append(app_feat)

        if len(self.app_feats_list) > self._feature_list_size:
            expired_feat = self.app_feats_list.pop(0)
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
    def tlwh(self):
        """Convert bounding box to format `(top left x, top left y, width, height)`."""
        top_left_x = self.geometric_info[0]
        top_left_y = self.geometric_info[1]
        width   = self.geometric_info[4]
        height  = self.geometric_info[5]
        return [top_left_x, top_left_y, width, height]

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`."""
        min_x = self.geometric_info[0]
        min_y = self.geometric_info[1]
        max_x = self.geometric_info[2]
        max_y = self.geometric_info[3]
        return [min_x, min_y, max_x, max_y]
    
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
    def __init__(self,model :GraphModel,device :str,path_to_weights :str, tracking_dict :dict):
       
        self.device = device
        self.model  = model.eval().to(device)

        self.fusion_method = tracking_dict['FUSION_METHOD']
        self.EMA_lambda    = tracking_dict['EMA_LAMBDA']

        self.tracks_list:List[Tracker] = [] # store all the tracks including Born, Active, Sleep, Dead

        self._resize_to_cnn   = tracking_dict['RESIZE_TO_CNN']
        self._det_conf_gate   = tracking_dict['DET_CONF_GATE']
        self._first_match_thresh  = tracking_dict['FIRST_MATCH_THRESH']
        self._second_match_lambda = tracking_dict['SECOND_MATCH_LAMBDA']
        self._second_match_thresh = tracking_dict['SECOND_MATCH_THRESH']
        self._third_match_thresh  = tracking_dict['THIRD_MATCH_THRESH']
        # necessary attributes when initializing the single track
        self._det2tra_conf    = tracking_dict['DET2TRA_CONF']
        self._cnt_to_active   = tracking_dict['CNT_TO_ACTIVE']
        self._cnt_to_sleep    = tracking_dict['CNT_TO_SLEEP']
        self._max_cnt_to_dead = tracking_dict['MAX_CNT_TO_DEAD']
        self._feature_list_size = tracking_dict['FEATURE_LIST_SIZE']

        if path_to_weights:
            try:
                self.model.load_state_dict(torch.load(path_to_weights,map_location='cpu')['model'])
            except KeyError:
                self.model.load_state_dict(torch.load(path_to_weights,map_location='cpu'))
            finally:
                logger.info(f"Load weights from {path_to_weights} successfully")
        else:
            logger.info(f"No weights loaded, use default weights")
    @torch.no_grad()
    def update(self,cur_frame:int,current_detections:np.ndarray,img_date:torch.Tensor) -> List[Tracker]:
        '''
        current_detections =np.ndarray(tlwh,conf)  and have already filtered by conf > 0.1 
        '''
        output_track_list  = []

        first_dets_list , second_dets_list = [] , []
        for idx,det in enumerate(current_detections):
           # format : <top left x> <top left y> <width> <height> <conf>
            if det[4] >= self._det_conf_gate:
                first_dets_list.append(det)
            else:
                second_dets_list.append(det)
        
        first_tras_list , second_tras_list , third_tras_list = [] , [] , []
        for track in self.tracks_list:
            if track.is_Active :
            # if track.bt_prev_node :
                first_tras_list.append(track)
            elif track.is_Sleep:
            # elif track.is_Sleep:
                second_tras_list.append(track)
            elif track.is_Born :
            # elif track.is_Born :
                third_tras_list.append(track)


        #------------------------------------------------------------------#
        #                       First matching phase
        #------------------------------------------------------------------#
        tra_graph  = self.construct_tra_graph(first_tras_list)
        det_graph  = self.construct_det_graph(first_dets_list,img_date)
        match_mtx,match_idx,unmatch_tra,unmatch_det = self._graph_match(tra_graph,det_graph,self._first_match_thresh)
        # The inputs `det_graph` & `tra_graph` are modified inside `self.model`, 
        # so their state changes after the function call.
        det_graph.geometric_info = det_graph.geometric_info.cpu().numpy()

        if match_idx and len(match_idx[0]) > 0 :         # matched tras and dets 
            # @BUG match_idx = [[],[]] 
            tra_idx ,det_idx = match_idx
            tra_app_feats  = tra_graph.x[tra_idx]
            tra_conf_list  = [first_tras_list[i].conf for i in tra_idx]
            det_app_feats  = det_graph.x[det_idx]
            det_conf_list  = [first_dets_list[i][4] for i in det_idx]

            smooth_app_feats  = self.smooth_feature(tra_app_feats,det_app_feats,tra_conf_list,det_conf_list,self.fusion_method)
            for i,(tra_id, det_id) in enumerate(zip(tra_idx,det_idx)):
                first_tras_list[tra_id].to_active(
                    cur_frame,smooth_app_feats[i].squeeze(),
                    det_conf_list[i],det_graph.geometric_info[det_id]
                )
                if not first_tras_list[tra_id].is_Born and not first_tras_list[tra_id].is_Sleep:
                    output_track_list.append(first_tras_list[tra_id])        
        for tra_id in unmatch_tra:   # unmatched tras 
            # first_match_list[tra_id].to_sleep()
            if not first_tras_list[tra_id].is_Born:
                second_tras_list.append(first_tras_list[tra_id])
            else:
                third_tras_list.append(first_tras_list[tra_id])

        id_map , third_dets_list = [] , []
        for det_id in unmatch_det:
            id_map.append(det_id)
            third_dets_list.append(first_dets_list[det_id])   # prepare for the third matching phase

        #------------------------------------------------------------------#
        #                       Second matching phase
        #------------------------------------------------------------------#
        if second_tras_list :
            if second_dets_list:
            # if []:
                #---------------------------------#
                # need to encode the necessary features of dets whose confidence is relatively lower
                #---------------------------------#
                second_det_graph = self.construct_det_graph(second_dets_list,img_date)
                second_det_graph.x.to(self.device)
                second_det_graph.geometric_info = second_det_graph.geometric_info.numpy()
                diff_t = cur_frame - np.asarray([track.end_frame for track in second_tras_list]).reshape(len(second_tras_list),1).repeat(len(second_dets_list),1).astype(np.float32)
                self.model.gen_appFeats(second_det_graph)
                # second_dets_conf = current_detections[second_dets_list,4]
                match_mtx,match_idx,unmatch_tra,unmatch_det = self._iou_reid_match(
                        second_tras_list,second_det_graph.geometric_info[:,:4],
                        second_det_graph.x,diff_t,self._second_match_thresh
                    )
                
                if match_idx and len(match_idx[0]) > 0 :    # matched tras and dets
                    tra_idx ,det_idx = match_idx   
                    tra_app_feats, tra_conf_list = [] , []
                    for i in tra_idx:
                        tra_app_feats.append(second_tras_list[i].app_feats_list[-1])
                        tra_conf_list.append(second_tras_list[i].conf)
                    tra_app_feats  = torch.stack(tra_app_feats,dim=0).to(self.device).to(torch.float32)

                    det_app_feats  = second_det_graph.x[det_idx]
                    det_conf_list  = [second_dets_list[i][4] for i in det_idx]

                    smooth_app_feats  = self.smooth_feature(tra_app_feats,det_app_feats,tra_conf_list,det_conf_list,self.fusion_method)
                    for i,(tra_id, det_id) in enumerate(zip(tra_idx,det_idx)):
                        second_tras_list[tra_id].to_active(
                            cur_frame,smooth_app_feats[i].squeeze(),
                            det_conf_list[i],second_det_graph.geometric_info[det_id]
                        )
                        output_track_list.append(second_tras_list[tra_id])
                for tra_id in unmatch_tra:
                    second_tras_list[tra_id].to_sleep()
                #---------------------------------#
                # Because of the low confidence , it is necessary to abandon the unmatched detecions
                #---------------------------------#
            else:
                for tra_id in range(len(second_tras_list)):
                    second_tras_list[tra_id].to_sleep()
        #------------------------------------------------------------------#
        #                       Third matching phase
        #------------------------------------------------------------------#
        third_dets_app  = det_graph.x[id_map]
        third_dets_geometric_info = det_graph.geometric_info[id_map]

        match_mtx,match_idx,unmatch_tra,unmatch_det = self._iou_match(third_tras_list,third_dets_geometric_info[:,:4],self._third_match_thresh)
        if match_idx and len(match_idx[0]) > 0 :         # matched tras and dets 
            tra_idx ,det_idx = match_idx
            tra_app_feats,tra_conf_list = [] , []
            for i in tra_idx:
                tra_app_feats.append(third_tras_list[i].app_feats_list[-1])
                tra_conf_list.append(third_tras_list[i].conf)
            tra_app_feats  = torch.stack(tra_app_feats,dim=0).to(self.device).to(torch.float32)
            
            det_app_feats  = third_dets_app[det_idx]
            det_conf_list  = [third_dets_list[i][4] for i in det_idx]

            smooth_app_feats  = self.smooth_feature(tra_app_feats,det_app_feats,tra_conf_list,det_conf_list,self.fusion_method)          
            for i,( tra_id, det_id ) in enumerate(zip(tra_idx,det_idx)):
                third_tras_list[tra_id].to_active(
                    cur_frame,smooth_app_feats[i].squeeze(),
                    det_conf_list[i],third_dets_geometric_info[det_id]
                )
                if not third_tras_list[tra_id].is_Born:
                    output_track_list.append(third_tras_list[tra_id])
        for tra_id in unmatch_tra:  # unmatched tras 
            third_tras_list[tra_id].to_sleep()
        for det_id in unmatch_det:
            if third_dets_list[det_id][4] >= self._det2tra_conf:
                third_tras_list.append(
                    Tracker(
                        cur_frame,third_dets_app[det_id],
                        third_dets_list[det_id][4],
                        third_dets_geometric_info[det_id],
                        self._cnt_to_active,self._cnt_to_sleep,self._max_cnt_to_dead,self._feature_list_size
                    )
                )

        self.tracks_list = self.remove_invalid_tracks(first_tras_list + second_tras_list + third_tras_list)
        return output_track_list

    def construct_tra_graph(self,tracks_list:List[Tracker]) -> Data:
        '''construct graph of tracks including ACTIVE'''
        if not tracks_list: # if no tracks 
            return Data(num_nodes=0)
        
        x , geometric_info = [] , []
        for track in tracks_list:
            x.append(track.app_feats_list[-1])
            geometric_info.append(track.geometric_info)
        x = torch.stack(x,dim=0)
        geometric_info = torch.as_tensor(geometric_info,dtype=torch.float32)
        return Data(x=x,geometric_info=geometric_info)
    
    def construct_det_graph(self,dets_list :Union[list,np.ndarray],img_date:torch.Tensor) -> Data:
        '''construct raw graph of detections'''

        h_im,w_im  = img_date.shape[1:]
        img_date   = img_date.to(self.device).to(torch.float32)
        
        raw_x , geometric_info = [] , []
        for det in dets_list:
            x,y,w,h = map(int,det[:4])
            w , h   = min(w, w+x)   , min(h,h+y)
            x , y   = max(x ,0)     , max(y,0)
            w , h   = min(w, w_im-x), min(h,h_im-y)
            xc , yc   = x + w / 2 , y + h / 2
            x2 , y2   = x + w     , y + h

            patch = T.crop(img_date,int(y),int(x),int(h),int(w))
            patch = T.resize(patch,self._resize_to_cnn)
            raw_x.append(patch)
            geometric_info.append([x,y,x2,y2,w,h,xc,yc,w_im,h_im])  # STORE x,y,x2,y2,w,h,xc,yc, W,H
        raw_x = torch.stack(raw_x,dim=0)
        geometric_info = torch.as_tensor(geometric_info,dtype=torch.float32)
        return Data(x=raw_x,geometric_info=geometric_info)

    def _graph_match(self,tra_graph:Data,det_graph:Data,match_thresh:float):
        ''' first phase to match via graph model'''
        pred_mtx = self.model.inference(tra_graph.to(self.device),det_graph.to(self.device))
        match_mtx,match_idx,unmatch_tra,unmatch_det = hungarian(pred_mtx.cpu().numpy(),match_thresh)
        return match_mtx,match_idx,unmatch_tra,unmatch_det

    def _iou_reid_match(self,tracks_list:list,dets_tlbr:np.ndarray,dets_app_feats:torch.Tensor,diff_t:np.ndarray,match_thresh:float):
        ''' second phase to match via IOU and Cosine Distance '''
        def dynamic_weight(t,lambda_):
            return np.exp(-lambda_ * t)
        
        tras_tlbr = [] 
        tras_tlbr , tras_app_feats = [] ,[]
        for track in tracks_list:
            tras_tlbr.append(track.tlbr)
            tras_app_feats.append(track.app_feats_list[-1])

        tras_tlbr = np.vstack(tras_tlbr).astype(np.float32)
        tras_app_feats = torch.stack(tras_app_feats,dim=0).to(dets_app_feats)

        iou = box_iou(tras_tlbr,dets_tlbr,iou_type='hiou')


        n1   = torch.norm(tras_app_feats,dim=-1,keepdim=True)
        n2   = torch.norm(dets_app_feats,dim=-1,keepdim=True)
        corr = (torch.mm(tras_app_feats,dets_app_feats.T) / torch.mm(n1,n2.T)).cpu().numpy()
        # cos = F.cosine_similarity(tras_app_feats,dets_app_feats).reshape(num_tras,num_dets).cpu().numpy()
        # weight = dynamic_weight(diff_t,self._second_match_lambda)
        weight =  self._second_match_lambda
        affinity_mtx = (1 - weight) * corr + weight * iou
        # affinity_mtx =  iou

        match_mtx,match_idx,unmatch_tra,unmatch_det = hungarian(affinity_mtx,match_thresh)

        return match_mtx,match_idx,unmatch_tra,unmatch_det

    def _iou_match(self,tracks_list,highconf_unmatch_dets_tlbr:np.ndarray,match_thresh:float):      
        ''' third phase to match via IOU'''
        if tracks_list == []:
            tras_tlbr = np.array([])
        else:
            tras_tlbr = np.vstack([
                track.tlbr for track in tracks_list 
            ]).astype(np.float32)

        iou  = box_iou(tras_tlbr,highconf_unmatch_dets_tlbr)
        match_mtx,match_idx,unmatch_tra,unmatch_det = hungarian(iou,match_thresh)

        return match_mtx,match_idx,unmatch_tra,unmatch_det

    def smooth_feature(self,
                prev_feats :torch.Tensor,cur_feats :torch.Tensor,
                prev_conf_list:list,cur_conf_list:list,
                fusion_method:str='DFF'
        ) -> torch.Tensor:

        if   fusion_method == 'DFF' or not prev_feats.numel(): # Direct Feature Fusion OR EMPTY TRAGRAPH
            return  cur_feats.split(1,0)
        elif fusion_method == 'CWFF': # Confidence-weight Feature Fusion
            prev_conf_tensor = torch.as_tensor(prev_conf_list).unsqueeze(1).to(prev_feats)
            cur_conf_tensor  = torch.as_tensor(cur_conf_list).unsqueeze(1).to(prev_feats)
            smooth_feature   = ( prev_conf_tensor * prev_feats + cur_conf_tensor * cur_feats ) / (prev_conf_tensor + cur_conf_tensor)
        elif fusion_method == 'EMAFF': # Expositential Moving Average Feature Fusion
            smooth_feature = (1 - self.EMA_lambda) * prev_feats + self.EMA_lambda * cur_feats
        elif fusion_method == 'CA-EMA':
            prev_conf_tensor = torch.as_tensor(prev_conf_list).unsqueeze(1).to(prev_feats)
            cur_conf_tensor  = torch.as_tensor(cur_conf_list).unsqueeze(1).to(prev_feats)
            CA_lambda = cur_conf_tensor / (prev_conf_tensor + cur_conf_tensor)
            # CA_lambda = torch.pow(cur_conf_tensor,2) 
            # CA_lambda = cur_conf_tensor

            smooth_feature = (1 - CA_lambda) * prev_feats + CA_lambda * cur_feats
            
        # return F.normalize(smooth_feature,dim=1).split(1,0)
        if smooth_feature.dim() > 1 :
            return smooth_feature.split(1,0)
        else:
            return smooth_feature
                
    def remove_invalid_tracks(self,tracks_list):
        """Remove all trackers whose state is Dead and maintain the uniqueness of id """
        id_list = []
        new_tracks_list = []
        for track in tracks_list:
            if track.is_Dead:
                continue
            elif track.is_Born:
                new_tracks_list.append(track)
            elif track.track_id not in id_list:
                id_list.append(track.track_id)
                new_tracks_list.append(track)
        return new_tracks_list

    def clean_cache(self):
        '''clean cache of all tracks'''
        self.tracks_list.clear()
        Tracker.clean_cache()
        gc.collect()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     graphTracker.py
@Time     :     2024/12/01 14:31:31
@Author   :     Louis Swift
@Desc     :     
'''
import gc
import yaml
import torch
import numpy as np
from loguru import logger
from typing import Tuple,List
import torch.nn.functional as F
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
            appearance_feat:torch.Tensor,
            node_feat:torch.Tensor,
            conf:float,
            geometric_info:list,
            cnt_to_active:int,
            cnt_to_sleep:int,
            max_cnt_to_dead:int,
            feature_list_size:int
        ):
        
        self.bt_prev_node = True

        self.track_id   = None # when state: Born to Active, this will be assigned
        self.track_len  = 0
        self.sleep_cnt  = 0 
        self.active_cnt = 0 
        self.state      = LifeSpan.Born

        self.start_frame   = start_frame 
        self.frame_idx     = start_frame

        self.conf = conf
        # self.tlwh = tlwh # (top left x, top left y, width, height)
        self.geometric_info = geometric_info
        self.node_feats_list = []
        self.appearance_feats_list = []
        self.node_feats_list.append(node_feat) 
        self.appearance_feats_list.append(appearance_feat) 
        
        self._cnt_to_active     = cnt_to_active
        self._cnt_to_sleep      = cnt_to_sleep
        self._max_cnt_to_dead   = max_cnt_to_dead  
        self._feature_list_size = feature_list_size

    def to_active(self,frame_idx,appearance_feat,node_feat,conf,geometric_info):
        self.bt_prev_node = True
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
        # self.tlwh = tlwh
        self.geometric_info = geometric_info
        self.node_feats_list.append(node_feat) 
        self.appearance_feats_list.append(appearance_feat)

        if len(self.appearance_feats_list) > self._feature_list_size:
            expired_feat = self.appearance_feats_list.pop(0)
            del expired_feat
        if len(self.node_feats_list) > self._feature_list_size:
            expired_feat = self.node_feats_list.pop(0)
            del expired_feat

    def to_sleep(self):  
        self.bt_prev_node = False
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
    def __init__(self,model :GraphModel,device :str,path_to_weights :str, path_to_tracking_cfg :str):
       
        with open(path_to_tracking_cfg,'r') as f:
            tracking_dict = yaml.load(f.read(),Loader=yaml.FullLoader)
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
    def update(self,frame_idx:int,current_detections:np.ndarray,img_date:torch.Tensor) -> List[Tracker]:
        '''
        current_detections =np.ndarray(tlwh,conf)  and have already filtered by conf > 0.1 
        '''
        output_track_list  = []

        high_dets_list , second_dets_list = [] , []
        for det in current_detections:
           # format : <top left x> <top left y> <width> <height> <conf>
            if det[4] >= self._det_conf_gate:
                high_dets_list.append(det)
            else:
                second_dets_list.append(det)
        
        first_match_list , second_match_list , third_match_list = [] , [] , []
        for track in self.tracks_list:
            if track.bt_prev_node or track.is_Active :
                first_match_list.append(track)
            elif track.is_Sleep:
                second_match_list.append(track)
            elif not track.bt_prev_node and track.is_Born :
                third_match_list.append(track)


        #------------------------------------------------------------------#
        #                       First matching phase
        #------------------------------------------------------------------#
        tra_graph  = self.construct_tra_graph(first_match_list)
        det_graph  = self.construct_det_graph(high_dets_list,img_date)
        match_mtx,match_idx,unmatch_tra,unmatch_det = self._graph_match(tra_graph,det_graph)
        # The inputs `det_graph` & `tra_graph` are modified inside `self.model`, 
        # so their state changes after the function call.
        if match_idx and len(match_idx[0]) > 0 :         # matched tras and dets 
            # @BUG match_idx = [[],[]] 
            tra_idx ,det_idx = match_idx
            tra_app_feats  = tra_graph.x[tra_idx]
            tra_node_feats = tra_graph.node_feats[tra_idx]
            tra_conf_list  = [first_match_list[i].conf for i in tra_idx]
            det_app_feats  = det_graph.x[det_idx]
            det_node_feats = det_graph.node_feats[det_idx]
            det_conf_list  = [current_detections[i][4] for i in det_idx]

            smooth_app_feats  = self.smooth_feature(tra_app_feats,det_app_feats,tra_conf_list,det_conf_list)
            smooth_node_feats = self.smooth_feature(tra_node_feats,det_node_feats,tra_conf_list,det_conf_list)
            for i,(tra_id, det_id) in enumerate(zip(tra_idx,det_idx)):
                first_match_list[tra_id].to_active(
                    frame_idx,smooth_app_feats[i].squeeze(),smooth_node_feats[i].squeeze(),
                    det_conf_list[i],det_graph.geometric_info[det_id].cpu().numpy()
                )
                if not first_match_list[tra_id].is_Born and not first_match_list[tra_id].is_Sleep:
                    output_track_list.append(first_match_list[tra_id])        
        for tra_id in unmatch_tra:   # unmatched tras 
            # first_match_list[tra_id].to_sleep()
            if not first_match_list[tra_id].is_Born:
                second_match_list.append(first_match_list[tra_id])
            else:
                third_match_list.append(first_match_list[tra_id])

        id_map , unmatched_det_list = [] , []
        for det_id in unmatch_det:
            id_map.append(det_id)
            unmatched_det_list.append(high_dets_list[det_id])   # prepare for the third matching phase

        #------------------------------------------------------------------#
        #                       Second matching phase
        #------------------------------------------------------------------#
        if second_match_list :
            # 发现一个问题 
            # 不能从低置信度中寻找！
            match_mtx,match_idx,unmatch_tra,unmatch_det = self._iou_reid_match(second_match_list,second_dets_tlbr,second_dets_app)

        
        #------------------------------------------------------------------#
        #                       Third matching phase
        #------------------------------------------------------------------#
        # highconf_unmatch_dets = current_detections[unmatch_det][current_detections[unmatch_det,4] >= self._det2tra_conf]
        highconf_unmatch_x    = det_graph.x[unmatch_det][current_detections[unmatch_det,4] >= self._det2tra_conf]
        highconf_unmatch_dets_conf = current_detections[unmatch_det][current_detections[unmatch_det,4] >= self._det2tra_conf][:,4].tolist()
        highconf_unmatch_dets_geometric_info = det_graph.geometric_info[unmatch_det][current_detections[unmatch_det,4] >= self._det2tra_conf].cpu().numpy()
        # highconf_to_global_det_idx = [ unmatch_det[i] for i in range(len(highconf_unmatch_dets)) ] 

        match_mtx,match_idx,unmatch_tra,unmatch_det = self._iou_match(third_match_list,highconf_unmatch_dets_geometric_info[:,:4])
        if match_idx and len(match_idx[0]) > 0 :         # matched tras and dets 
            tra_idx ,det_idx = match_idx
            if third_match_list: # if tra_graph is not empty
                tra_app_feats = [third_match_list[i].appearance_feats_list[-1] for i in tra_idx]
                tra_app_feats = torch.stack(tra_app_feats,dim=0)
                tra_conf_list = [third_match_list[i].conf for i in tra_idx]
            else:
                tra_app_feats = None
                tra_conf_list = None
    
            # for det_id in det_idx:
            #     # global_id = highconf_to_global_det_idx[det_id]
            #     det_feats.append(highconf_unmatch_x[det_id])
            #     # det_geometric_info.append(det_graph.geometric_info[det_id].cpu().numpy())
            det_app_feats = highconf_unmatch_x[det_idx]
            det_conf_list = [highconf_unmatch_dets_conf[i] for i in det_idx]
            smooth_app_feats = self.smooth_feature(tra_app_feats,det_app_feats,tra_conf_list,det_conf_list)          

            for i,( tra_id, det_id ) in enumerate(zip(tra_idx,det_idx)):
                third_match_list[tra_id].to_active(frame_idx,smooth_app_feats[i].squeeze(),det_conf_list[i],highconf_unmatch_dets_geometric_info[det_id])
                if not third_match_list[tra_id].is_Born:
                    output_track_list.append(third_match_list[tra_id])
        for tra_id in unmatch_tra:  # unmatched tras 
            third_match_list[tra_id].to_sleep()
        for det_id in unmatch_det:
            # global_id = highconf_to_global_det_idx[det_id]
            third_match_list.append(
                Tracker(frame_idx,highconf_unmatch_x[det_id],
                        highconf_unmatch_dets_conf[det_id],highconf_unmatch_dets_geometric_info[det_id],
                        self._cnt_to_active,self._cnt_to_sleep,self._max_cnt_to_dead,self._feature_list_size)
            )




        self.tracks_list = self.remove_dead_tracks(first_match_list + second_match_list + third_match_list)
        return output_track_list

    def construct_tra_graph(self,tracks_list:List[Tracker]) -> Data:
        '''construct graph of tracks including ACTIVE'''
        if not tracks_list: # if no tracks 
            return Data(num_nodes=0)
        
        node_feats , geometric_info = [] , []
        for track in tracks_list:
            node_feats.append(track.node_feats_list[-1])
            geometric_info.append(track.geometric_info)
        node_feats = torch.stack(node_feats,dim=0).to(self.device)
        geometric_info = torch.as_tensor(geometric_info,dtype=torch.float32).to(self.device)
        return Data(node_feats=node_feats,geometric_info=geometric_info)
    
    def construct_det_graph(self,high_dets_list:list,img_date:torch.Tensor) -> Data:
        '''construct raw graph of detections'''

        h_im,w_im  = img_date.shape[1:]
        raw_node_attr , geometric_info = [] , []
        img_date   = img_date.to(self.device).to(torch.float32) / 255.0
        for det in high_dets_list:
            x,y,w,h = det[:4]
            w , h   = min(w, w+x)   , min(h,h+y)
            x , y   = max(x ,0)     , max(y,0)
            w , h   = min(w, w_im-x), min(h,h_im-y)
            xc , yc   = x + w / 2 , y + h / 2
            x2 , y2   = x + w     , y + h

            patch = T.crop(img_date,int(y),int(x),int(h),int(w))
            patch = T.resize(patch,self._resize_to_cnn)
            raw_node_attr.append(patch)
            geometric_info.append([x,y,x2,y2,w,h,xc,yc,w_im,h_im])  # STORE x,y,x2,y2,w,h,xc,yc, W,H
        raw_node_attr  = torch.stack(raw_node_attr,dim=0).to(self.device)
        geometric_info = torch.as_tensor(geometric_info,dtype=torch.float32).to(self.device)
        return Data(x=raw_node_attr,geometric_info=geometric_info)

    def _graph_match(self,tra_graph:Data,det_graph:Data):
        ''' first phase to match via graph model'''
        pred_mtx = self.model.inference(tra_graph.to(self.device),det_graph.to(self.device))
        match_mtx,match_idx,unmatch_tra,unmatch_det = hungarian(pred_mtx.cpu().numpy(),self._first_match_thresh)
        return match_mtx,match_idx,unmatch_tra,unmatch_det

    def _iou_reid_match(self,tracks_list:list,dets_tlbr,dets_app_feats:torch.Tensor):
        ''' second phase to match via IOU and Cosine Distance'''
        
        tras_tlbr , tras_app_feats = [] ,[]
        for track in tracks_list:
            tras_tlbr.append(track.tlbr)
            tras_app_feats.append(track.node_feats_list[-1])

        tras_tlbr = np.vstack(tras_tlbr).astype(np.float32)
        tras_app_feats = tras_app_feats.to(dets_app_feats)

        iou = box_iou(tras_tlbr,dets_tlbr)
        cos = F.cosine_similarity(tras_app_feats,dets_app_feats).cpu().numpy()

        affinity_mtx = self._second_match_lambda * cos + (1-self._second_match_lambda) * iou
        match_mtx,match_idx,unmatch_tra,unmatch_det = hungarian(affinity_mtx,self._second_match_thresh)

        return match_mtx,match_idx,unmatch_tra,unmatch_det

    def _iou_match(self,tracks_list,highconf_unmatch_dets_tlbr:np.ndarray):      
        ''' third phase to match via IOU'''
        if tracks_list == []:
            tras_tlbr = np.array([])
        else:
            tras_tlbr = np.vstack([
                track.tlbr for track in tracks_list 
            ]).astype(np.float32)

        iou  = box_iou(tras_tlbr,highconf_unmatch_dets_tlbr)
        match_mtx,match_idx,unmatch_tra,unmatch_det = hungarian(iou,self._third_match_thresh)

        return match_mtx,match_idx,unmatch_tra,unmatch_det

    def smooth_feature(self,prev_feats :torch.Tensor,cur_feats :torch.Tensor,prev_conf_list:list,cur_conf_list:list) -> torch.Tensor:

        if self.fusion_method == 'DFF' or prev_feats is None: # Direct Feature Fusion OR EMPTY TRAGRAPH
            return  cur_feats.split(1,0)
        elif self.fusion_method == 'CWFF': # Confidence-weight Feature Fusion
            prev_conf_tensor = torch.as_tensor(prev_conf_list).unsqueeze(1).to(prev_feats)
            cur_conf_tensor  = torch.as_tensor(cur_conf_list).unsqueeze(1).to(prev_feats)
            smooth_feature   = ( prev_conf_tensor * prev_feats + cur_conf_tensor * cur_feats ) / (prev_conf_tensor + cur_conf_tensor)
        elif self.fusion_method == 'EMAFF': # Expositential Moving Average Feature Fusion
            smooth_feature = (1 - self.EMA_lambda) * prev_feats + self.EMA_lambda * cur_feats
        elif self.fusion_method == 'CA-EMA':
            prev_conf_tensor = torch.as_tensor(prev_conf_list).unsqueeze(1).to(prev_feats)
            cur_conf_tensor  = torch.as_tensor(cur_conf_list).unsqueeze(1).to(prev_feats)
            CA_lambda = cur_conf_tensor / (prev_conf_tensor + cur_conf_tensor)
            # CA_lambda = torch.pow(cur_conf_tensor,2) 
            # CA_lambda = cur_conf_tensor

            smooth_feature = (1 - CA_lambda) * prev_feats + CA_lambda * cur_feats
            
        # return F.normalize(smooth_feature,dim=1).split(1,0)
        return smooth_feature.split(1,0)
        
    def remove_dead_tracks(self,tracks_list):
        """Remove all trackers whose state is Dead"""
        return [track for track in tracks_list if not track.is_Dead]

    def clean_cache(self):
        '''clean cache of all tracks'''
        self.tracks_list.clear()
        Tracker.clean_cache()
        gc.collect()
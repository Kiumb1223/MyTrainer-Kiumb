#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :GraphModel.py
:Description:
:EditTime   :2024/11/20 15:47:09
:Author     :Kiumb
'''

import copy
import yaml
import torch 
import torch.nn as nn 
from functools import partial
from torch_geometric.data import Batch,Data
from models.core.graphConv import SDgraphConv
from models.graphToolkit import sinkhorn_unrolled,calc_iou,calc_cosineSim
from models.core.graphLayers import NodeEncoder,EdgeEncoder,SequentialBlock,AffinityLayer

__all__ =['GraphModel']

class GraphModel(nn.Module):
    def __init__(self,model_yaml_path):
        super().__init__()

        with open(model_yaml_path,'r') as f:
            model_dict = yaml.load(f.read(),Loader=yaml.FullLoader)
            
        self.k = model_dict['K_NEIGHBOR']
        self.bt_mask = model_dict['BT_DIST_MASK'] 
        self.dist_thresh = model_dict['DIST_THRESH']
        #---------------------------------#
        # Encoder Layer
        #---------------------------------#
        self.nodeEncoder = NodeEncoder(model_dict['node_encoder'])
        self.edgeEncoder = EdgeEncoder(model_dict['edge_encoder'])

        #---------------------------------#
        # Graph Layer
        #---------------------------------#
        self.graphconvLayer = SDgraphConv(
                static_graph_conv_dict = model_dict['static_graph_conv'],
                dynamic_graph_conv_dict = model_dict['dynamic_graph_conv'],
                fuse_model_dict = model_dict['fuse_model']
            )
        
        #---------------------------------#
        # Affinity Layer
        #---------------------------------#

        self.affinityLayer = SequentialBlock(
            model_dict['affinity_model']['dims_list'],
            model_dict['affinity_model']['layer_type'],model_dict['affinity_model']['layer_bias'],
            model_dict['affinity_model']['norm_type'],model_dict['affinity_model']['activate_func'],
        )
# /        self.affinityLayer = AffinityLayer(model_dict['affinity_model'])

        #---------------------------------#
        # Sinkhorn Layer 
        #---------------------------------#
        self.alpha   = nn.Parameter(torch.ones(1))
        self.eplison = nn.Parameter(torch.zeros(1))
        self.sinkhornLayer = partial(sinkhorn_unrolled,num_sink = model_dict['SINKHORN_ITERS'])
        

    def forward(self,tra_graph_batch: Batch ,det_graph_batch: Batch) -> list:
        ''' Training Process'''
        tra_graph_batch = self.nodeEncoder(tra_graph_batch)
        tra_graph_batch = self.edgeEncoder(tra_graph_batch,self.k,tra_graph_batch.batch)
        
        det_graph_batch = self.nodeEncoder(det_graph_batch)
        det_graph_batch = self.edgeEncoder(det_graph_batch,self.k,det_graph_batch.batch)        


        #---------------------------------#
        # Feed the detection graph and trajectory graph into the graph network
        # and return the node feature for each graph 
        #---------------------------------#

        tra_node_feats = self.graphconvLayer(tra_graph_batch,self.k)
        det_node_feats = self.graphconvLayer(det_graph_batch,self.k)
        
        #---------------------------------#
        # Optimal transport
        # > Reference: https://github.com/magicleap/SuperGluePretrainedNetwork
        #   1. compute the affinity matrix
        #   2. perform matrix augumentation 
        #---------------------------------#
        pred_mtx_list = []
        num_graphs        = tra_graph_batch.num_graphs # Actually equal to 'batch-size'
        tra_batch_indices = tra_graph_batch.batch      # Batch indices for trajectory graph
        det_batch_indices = det_graph_batch.batch      # Batch indices for detection graph
        for graph_idx in range(num_graphs):

            # Slice node features for the current graph 
            tra_node = tra_node_feats[tra_batch_indices == graph_idx]  
            det_node = det_node_feats[det_batch_indices == graph_idx]  
            
            tra_app  = tra_graph_batch.x[tra_batch_indices == graph_idx]
            det_app  = det_graph_batch.x[det_batch_indices == graph_idx]

            tra_xyxy = tra_graph_batch.geometric_info[tra_batch_indices == graph_idx,:4]
            det_xyxy = det_graph_batch.geometric_info[det_batch_indices == graph_idx,:4]

            # 1. Compute affinity matrix for the current graph 
            # node_sim = calc_cosineSim(tra_node,det_node)
            node_sim = calc_cosineSim(tra_node,det_node).unsqueeze(-1)
            app_sim  = calc_cosineSim(tra_app,det_app).unsqueeze(-1)
            iou      = calc_iou(tra_xyxy,det_xyxy,iou_type='hiou').unsqueeze(-1)
            corr = torch.cat([node_sim,app_sim,iou],dim=-1)
            corr = self.affinityLayer(corr).squeeze(-1)

            if self.bt_mask: # compute mask to filter out some unmatched nodes
                dist_mask = ( torch.cdist(tra_graph_batch.geometric_info[tra_batch_indices == graph_idx,6:8],det_graph_batch.geometric_info[det_batch_indices == graph_idx,6:8]) <= self.dist_thresh ).float()
                # corr = node_sim  * dist_mask
                corr = corr  * dist_mask

            # 2. Prepare the augmented affinity matrix for Sinkhorn
            m , n = corr.shape
            bins0 = self.alpha.expand(m, 1)
            bins1 = self.alpha.expand(1, n)
            alpha = self.alpha.expand(1, 1)
            couplings = torch.cat([torch.cat([corr,bins0],dim=-1),
                                torch.cat([bins1,alpha],dim=-1)],dim=0)
            # norm  = 1 / ( m + n )  
            a_aug = torch.full((m+1,),1,device=self.alpha.device,dtype=torch.float32) 
            b_aug = torch.full((n+1,),1,device=self.alpha.device,dtype=torch.float32) 
            a_aug[-1] =  n
            b_aug[-1] =  m            
           

            pred_mtx = self.sinkhornLayer( - couplings,a_aug,b_aug,
                                          lambd_sink = torch.exp(self.eplison) + 0.03)
            
            pred_mtx_list.append(pred_mtx[:-1,:-1])

        return pred_mtx_list 

    def inference(self,tra_graph :Data ,det_graph :Data ) -> torch.Tensor:
        ''' Inference Process [In Track Management phase]'''
        #---------------------------------#
        # This condition handles the test phase and  when processing the first frame, where 
        # the trajectory graph (tra_graph_batch) is not available (i.e., it lacks 'geometric_info').
        # In such cases, the model simply encodes the detection graph (det_graph_batch) nodes
        # and returns an empty list, bypassing the rest of the forward pass.
        #---------------------------------#
        
        if tra_graph.num_nodes == 0:
            self.nodeEncoder(det_graph)
            return torch.zeros((tra_graph.num_nodes,det_graph.num_nodes),dtype=torch.float32)

        #---------------------------------#
        # Initialize the Node and edge embeddings
        #---------------------------------#
        if tra_graph.x.dim() != 2:
            tra_graph = self.nodeEncoder(tra_graph)
        tra_graph = self.edgeEncoder(tra_graph,self.k)

        det_graph = self.nodeEncoder(det_graph)
        det_graph = self.edgeEncoder(det_graph,self.k)        


        #---------------------------------#
        # Feed the detection graph and trajectory graph into the graph network
        # and return the node feature for each graph 
        #---------------------------------#

        tra_node_feats = self.graphconvLayer(tra_graph,self.k)
        det_node_feats = self.graphconvLayer(det_graph,self.k)

        # node_sim = calc_cosineSim(tra_node_feats,det_node_feats)
        node_sim = calc_cosineSim(tra_node_feats,det_node_feats).unsqueeze(-1)
        app_sim  = calc_cosineSim(tra_graph.x,det_graph.x).unsqueeze(-1)
        iou      = calc_iou(tra_graph.geometric_info[:,:4],det_graph.geometric_info[:,:4],iou_type='hiou').unsqueeze(-1)
        
        corr = torch.cat([node_sim,app_sim,iou],dim=-1)
        corr = self.affinityLayer(corr).squeeze(-1)



        if self.bt_mask: # compute mask to filter out some unmatched nodes
            dist_mask = ( torch.cdist(tra_graph.geometric_info[:,6:8],det_graph.geometric_info[:,6:8]) <= self.dist_thresh ).float()
            corr = corr * dist_mask
            # corr = node_sim * dist_mask

        m , n = corr.shape
        bins0 = self.alpha.expand(m, 1)
        bins1 = self.alpha.expand(1, n)
        alpha = self.alpha.expand(1, 1)
        couplings = torch.cat([torch.cat([corr,bins0],dim=-1),
                               torch.cat([bins1,alpha],dim=-1)],dim=0)
        # norm  = 1 / ( m + n )  
        a_aug = torch.full((m+1,),1,device=self.alpha.device,dtype=torch.float32) 
        b_aug = torch.full((n+1,),1,device=self.alpha.device,dtype=torch.float32) 
        a_aug[-1] =  n
        b_aug[-1] =  m

        pred_mtx = self.sinkhornLayer( - couplings,a_aug,b_aug,
                                lambd_sink = torch.exp(self.eplison) + 0.03) 
        return pred_mtx[:-1,:-1]
    
    def gen_appFeats(self,det_graph :Data):
        '''
        In Track Management phase,
        input:
            det_graph(x :torch.Tensor || Size:(node_num,3,256,128),geometric_info)
        retures:
            det_graph(x :torch.Tensor || Size:(node_num,32),geometric_info)
        '''
        self.nodeEncoder(det_graph)
    
    def gen_nodeFeats(self,tra_graph :Data) -> torch.Tensor:
        '''
        In Track Management phase,
        input:
            tra_graph(x :torch.Tensor || Size:(node_num,32),geometric_info)
        retures:
            node_feats: torch.Tensor || Size:(node_num,198)
        '''

        assert tra_graph.x.shape == (tra_graph.num_nodes,32)
        tra_graph = self.edgeEncoder(tra_graph,self.k)
        tra_node_feats = self.graphconvLayer(tra_graph,self.k)
        return tra_node_feats
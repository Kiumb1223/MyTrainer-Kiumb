#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :Tracker.py
:Description:
:EditTime   :2024/11/20 15:47:09
:Author     :Kiumb
'''

import torch 
import torch.nn as nn 
from typing import Union
from loguru import logger
from functools import partial
from torch_geometric.data import Batch,Data
from models.core.graphConv import GraphConv
from models.graphToolkit import Sinkhorn,sinkhorn_unrolled

__all__ =['GraphModel']

class GraphModel(nn.Module):

    def __init__(self,cfg):
        super().__init__()

        self.k = cfg.K_NEIGHBOR
        self.device = cfg.DEVICE
        # Graph Layer
        self.graphconvLayer = GraphConv(cfg.NODE_EMBED_SIZE,cfg.EDGE_EMBED_SIZE)
        # Sinkhorn Layer 
        # self.alpha   = nn.Parameter(torch.ones(1))
        self.eplison = nn.Parameter(torch.zeros(1))

        # Maybe occur some error when using :class:Sinkhorn
        # self.sinkhornLayer = Sinkhorn.apply
        # self.sinkhorn_iters=cfg.SINKHORN_ITERS

        self.sinkhornLayer = partial(sinkhorn_unrolled,num_sink = cfg.SINKHORN_ITERS)
        

    def forward(self,tra_graph_batch:Union[Batch,Data] ,det_graph_batch:Union[Batch,Data]) -> list:
        
        #---------------------------------#
        # This condition handles the test phase and  when processing the first frame, where 
        # the trajectory graph (tra_graph_batch) is not available (i.e., it lacks 'location_info').
        # In such cases, the model simply encodes the detection graph (det_graph_batch) nodes
        # and returns an empty list, bypassing the rest of the forward pass.
        #---------------------------------#
        if not hasattr(tra_graph_batch,'location_info'):
            self.graphconvLayer.nodeEncoder(det_graph_batch)
            return [(None,det_graph_batch.num_nodes)]
        
        
        if not hasattr(tra_graph_batch,'num_graphs'): # Data type 
            tra_graph_batch = Batch.from_data_list([tra_graph_batch]) 
            det_graph_batch = Batch.from_data_list([det_graph_batch]) 
            tra_graph_batch.to(self.device)
            det_graph_batch.to(self.device)


        num_graphs        = tra_graph_batch.num_graphs # Actually equal to 'batch-size'
        tra_batch_indices = tra_graph_batch.batch      # Batch indices for trajectory graph
        det_batch_indices = det_graph_batch.batch      # Batch indices for detection graph

        #---------------------------------#
        # Feed the detection graph and trajectory graph into the graph network
        # and return the node feature for each graph 
        #---------------------------------#

        tra_node_feats = self.graphconvLayer(tra_graph_batch,self.k)
        det_node_feats = self.graphconvLayer(det_graph_batch,self.k)
        
        #---------------------------------#
        # Optimal transport
        # > Reference:https://github.com/magicleap/SuperGluePretrainedNetwork
        #   1. compute the affinity matrix
        #   2. perform matrix augumentation 
        #---------------------------------#
        pred_mtx_list = []
        for graph_idx in range(num_graphs):

            # Slice node features for the current graph 
            tra_feats = tra_node_feats[tra_batch_indices == graph_idx]  
            det_feats = det_node_feats[det_batch_indices == graph_idx]  

            # 1. Compute affinity matrix for the current graph 
            corr = torch.mm(tra_feats,det_feats.transpose(1,0))
            n1   = torch.norm(tra_feats,dim=-1,keepdim=True)
            n2   = torch.norm(det_feats,dim=-1,keepdim=True)
            cost = - corr / torch.mm(n1,n2.transpose(1,0))  

            # 2. Prepare the augmented cost matrix for Sinkhorn
            m , n = cost.shape
            # bins0 = self.alpha.expand(m, 1)
            # bins1 = self.alpha.expand(1, n)
            # alpha = self.alpha.expand(1, 1)
            # couplings = torch.cat([torch.cat([cost,bins0],dim=-1),
            #                        torch.cat([bins1,alpha],dim=-1)],dim=0)
            # norm  = 1 / (m+n)  
            # a_aug = torch.full((m+1,),norm,device=self.device,dtype=torch.float32) 
            # b_aug = torch.full((n+1,),norm,device=self.device,dtype=torch.float32) 
            # a_aug[-1] = norm * n
            # b_aug[-1] = norm * m
            # a_aug = torch.full((m+1,),1,device=self.device,dtype=torch.float32) 
            # b_aug = torch.full((n+1,),1,device=self.device,dtype=torch.float32) 
            a = torch.full((m,),0.9,device=self.device,dtype=torch.float32) 
            b = torch.full((n,),0.9,device=self.device,dtype=torch.float32) 

            # pred_mtx = self.sinkhornLayer(couplings,a_aug,b_aug,
            #                             self.sinkhorn_iters,torch.exp(self.eplison) + 0.03)

            
            # to original possibility space 
            pred_mtx = self.sinkhornLayer(cost,a,b,
                                        #   lambd_sink = torch.exp(self.eplison) + 0.03) * (m + n)
                                          lambd_sink = torch.exp(self.eplison) + 0.03)
            
            pred_mtx_list.append(pred_mtx)
            # if self.training:
            #     return self.compute_loss(output,gt_matrix)

        return pred_mtx_list 
    
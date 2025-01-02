#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :GraphModel.py
:Description:
:EditTime   :2024/11/20 15:47:09
:Author     :Kiumb
'''

import yaml
import torch 
import torch.nn as nn 
from functools import partial
from torch_geometric.data import Batch,Data
from models.core.graphConv import SDgraphConv
from models.graphToolkit import sinkhorn_unrolled
from models.core.graphEncoder import NodeEncoder,EdgeEncoder

__all__ =['TrainingGraphModel','GraphModel']

class TrainingGraphModel(nn.Module):
    '''This model structure is spefically designed for training purposes.'''
    def __init__(self,model_yaml_path):
        super().__init__()

        with open(model_yaml_path,'r') as f:
            model_dict = yaml.load(f.read(),Loader=yaml.FullLoader)
            
        self.k = model_dict['K_NEIGHBOR']
        self.bt_mask = model_dict['BT_DIST_MASK'] 
        self.dist_thresh = model_dict['DIST_THRESH']
        # Encoder Layer
        self.nodeEncoder = NodeEncoder(model_dict['node_encoder'])
        self.edgeEncoder = EdgeEncoder(model_dict['edge_encoder'])

        # Graph Layer
        self.graphconvLayer = SDgraphConv(
            static_graph_model_dict = model_dict['static_graph_model'],
            dynamic_graph_model_dict = model_dict['dynamic_graph_model'],
            fuse_model_dict = model_dict['fuse_model']
            )
        
        # Sinkhorn Layer 
        self.alpha   = nn.Parameter(torch.ones(1))
        self.eplison = nn.Parameter(torch.zeros(1))

        self.sinkhornLayer = partial(sinkhorn_unrolled,num_sink = model_dict['SINKHORN_ITERS'])
        

    def forward(self,tra_graph_batch: Batch ,det_graph_batch: Batch) -> list:
        
        tra_graph_batch = self.nodeEncoder(tra_graph_batch)
        tra_graph_batch = self.edgeEncoder(tra_graph_batch,self.k)
        
        det_graph_batch = self.nodeEncoder(det_graph_batch)
        det_graph_batch = self.edgeEncoder(det_graph_batch,self.k)        


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
            tra_feats = tra_node_feats[tra_batch_indices == graph_idx]  
            det_feats = det_node_feats[det_batch_indices == graph_idx]  
            # 1. Compute affinity matrix for the current graph 
            n1   = torch.norm(tra_feats,dim=-1,keepdim=True)
            n2   = torch.norm(det_feats,dim=-1,keepdim=True)

            if self.bt_mask: # compute mask to filter out some unmatched nodes
                dist_mask = ( torch.cdist(tra_graph_batch.location_info[tra_batch_indices == graph_idx][:,6:8],
                                        det_graph_batch.location_info[det_batch_indices == graph_idx][:,6:8]) <= self.dist_thresh ).float()
                corr = ( torch.mm(tra_feats,det_feats.transpose(1,0)) / torch.mm(n1,n2.transpose(1,0)) ) * dist_mask
            else:
                corr = torch.mm(tra_feats,det_feats.transpose(1,0)) / torch.mm(n1,n2.transpose(1,0))
                
            # 2. Prepare the augmented affinity matrix for Sinkhorn
            m , n = corr.shape
            bins0 = self.alpha.expand(m, 1)
            bins1 = self.alpha.expand(1, n)
            alpha = self.alpha.expand(1, 1)
            couplings = torch.cat([torch.cat([corr,bins0],dim=-1),
                                   torch.cat([bins1,alpha],dim=-1)],dim=0)
            norm  = 1 / ( m + n )  
            a_aug = torch.full((m+1,),norm,device=self.alpha.device,dtype=torch.float32) 
            b_aug = torch.full((n+1,),norm,device=self.alpha.device,dtype=torch.float32) 
            a_aug[-1] = norm * n
            b_aug[-1] = norm * m

            # to original possibility space 
            pred_mtx = self.sinkhornLayer(1 - couplings,a_aug,b_aug,
                                          lambd_sink = torch.exp(self.eplison) + 0.03) * (m + n)
            
            pred_mtx_list.append(pred_mtx)

        return pred_mtx_list 


class GraphModel(nn.Module):
    '''This model structure is spefically designed for evalution purposes.'''
    def __init__(self,model_yaml_path):
        super().__init__()

        with open(model_yaml_path,'r') as f:
            model_dict = yaml.load(f.read(),Loader=yaml.FullLoader)

        self.k = model_dict['K_NEIGHBOR']
        self.bt_mask = model_dict['BT_DIST_MASK'] 
        self.dist_thresh = model_dict['DIST_THRESH']
        # Encoder Layer
        self.nodeEncoder = NodeEncoder(model_dict['node_encoder'])
        self.edgeEncoder = EdgeEncoder(model_dict['edge_encoder'])

        # Graph Layer
        self.graphconvLayer = SDgraphConv(
            static_graph_model_dict = model_dict['static_graph_model'],
            dynamic_graph_model_dict = model_dict['dynamic_graph_model'],
            fuse_model_dict = model_dict['fuse_model']
            )
        
        # Sinkhorn Layer 
        self.alpha   = nn.Parameter(torch.ones(1))
        self.eplison = nn.Parameter(torch.zeros(1))

        self.sinkhornLayer = partial(sinkhorn_unrolled,num_sink = model_dict['SINKHORN_ITERS'])
        
        

    def forward(self,tra_graph :Data ,det_graph :Data ) -> torch.Tensor:
        
        #---------------------------------#
        # This condition handles the test phase and  when processing the first frame, where 
        # the trajectory graph (tra_graph_batch) is not available (i.e., it lacks 'location_info').
        # In such cases, the model simply encodes the detection graph (det_graph_batch) nodes
        # and returns an empty list, bypassing the rest of the forward pass.
        #---------------------------------#
        if tra_graph.num_nodes == 0:
            self.nodeEncoder(det_graph)
            return torch.zeros((tra_graph.num_nodes,det_graph.num_nodes),dtype=torch.float32)
        
        #---------------------------------#
        # Initialize the Node and edge embeddings
        #---------------------------------#
        
        if len(tra_graph.x.shape) != 2  :   # [N, node_embed_size] or [N,3,224,128]
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
        
        #---------------------------------#
        # Optimal transport
        # > Reference: https://github.com/magicleap/SuperGluePretrainedNetwork
        #   1. compute the affinity matrix
        #   2. perform matrix augumentation 
        #---------------------------------#

        # 1. Compute affinity matrix for the current graph 
        n1   = torch.norm(tra_node_feats,dim=-1,keepdim=True)
        n2   = torch.norm(det_node_feats,dim=-1,keepdim=True)
        if self.bt_mask: # compute mask to filter out some unmatched nodes
            dist_mask = ( torch.cdist(tra_graph.location_info[:,6:8],det_graph.location_info[:,6:8]) <= self.dist_thresh ).float()
            corr = ( torch.mm(tra_node_feats,det_node_feats.transpose(1,0)) / torch.mm(n1,n2.transpose(1,0)) ) * dist_mask
        else:
            corr = torch.mm(tra_node_feats,det_node_feats.transpose(1,0)) / torch.mm(n1,n2.transpose(1,0))

        # 2. Prepare the augmented affinity matrix for Sinkhorn
        m , n = corr.shape
        bins0 = self.alpha.expand(m, 1)
        bins1 = self.alpha.expand(1, n)
        alpha = self.alpha.expand(1, 1)
        couplings = torch.cat([torch.cat([corr,bins0],dim=-1),
                               torch.cat([bins1,alpha],dim=-1)],dim=0)
        norm  = 1 / ( m + n )  
        a_aug = torch.full((m+1,),norm,device=self.alpha.device,dtype=torch.float32) 
        b_aug = torch.full((n+1,),norm,device=self.alpha.device,dtype=torch.float32) 
        a_aug[-1] = norm * n
        b_aug[-1] = norm * m

        # to original possibility space 
        pred_mtx = self.sinkhornLayer(1 - couplings,a_aug,b_aug,
                                        lambd_sink = torch.exp(self.eplison) + 0.03) * (m + n)
        
        return pred_mtx 
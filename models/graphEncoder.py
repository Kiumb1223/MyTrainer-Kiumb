#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from typing import Union,Optional
from models.graphToolkit import knn
from torch_geometric.data import Batch,Data

__all__ = ['NodeEncoder','EdgeEncoder']

class NodeEncoder(nn.Module):
    
    def __init__(self, node_embed_size:int):
        super(NodeEncoder, self).__init__()
        # get the pretrained densenet model
        backbone = models.densenet121(pretrained=True)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))
        
        self.head     = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1), # 1024
            nn.Linear(1024,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,node_embed_size)
        )
        # Optional freeze model weights
        for param in self.backbone.parameters():
            param.requires_grad = False # Freeze

    def forward(self, graph:Union[Batch,Data]) -> Union[Batch,Data]:

        graph.x = self.backbone(graph.x)
        graph.x = self.head(graph.x)
        
        return graph
    
class EdgeEncoder(nn.Module):

    def __init__(self, edge_embed_size:int):
        super(EdgeEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5,18),
            nn.BatchNorm1d(18),
            nn.ReLU(inplace=True),
            nn.Linear(18,edge_embed_size),
            nn.BatchNorm1d(edge_embed_size),
            nn.ReLU(inplace=True)
        )
    def forward(self,graph:Union[Batch,Data],k:int) -> Union[Batch,Data]:
        
        assert len(graph.x.shape) == 2 , 'Encode node attribute first!'
        
        with torch.no_grad():
            graph.edge_index = self.construct_edge_index(graph,k,bt_cosine=False,bt_self_loop=True)
            raw_edge_attr    = self.compute_edge_attr(graph)
        
        graph.edge_attr = self.encoder(raw_edge_attr)
        
        return graph 

    def construct_edge_index(self,batch: Union[Batch,Data], k ,bt_cosine: Optional[bool]=False, bt_self_loop: Optional[bool]=True) -> torch.Tensor:
        """
        Construct edge_index in either the Batch or Data.
        > construct KNN for each subgraph in the Batch
        
        Args:
            batch (Batch): Batch object containing multiple graphs.
            bt_cosine (bool): Whether to use cosine distance.
            bt_self_loop (bool): Whether to include self-loop.
            
        Returns:
            edge_index (Tensor): Edge indices of KNN for all graphs. 'soure_to_target'
        """

        if not hasattr(batch,'num_graphs'): # Date Type
            edge_index = knn(batch.location_info,k, bt_cosine=bt_cosine, bt_self_loop=bt_self_loop,bt_edge_index=True)
            return edge_index
        
        # Batch Type
        all_edge_index = []
        for i in range(batch.num_graphs):
            start, end = batch.ptr[i:i+2]
            
            sub_positions = batch.location_info[start:end,-2:]
            
            indices,k2 = knn(sub_positions, k, bt_cosine=bt_cosine, bt_self_loop=bt_self_loop, bt_edge_index=False)
            
            source_indices = indices + start
            target_indices = torch.arange(start, end, device=batch.location_info.device).repeat_interleave(k2)
            
            edge_index = torch.stack([source_indices.flatten(), target_indices], dim=0)
            
            all_edge_index.append(edge_index)
        
        edge_index = torch.cat(all_edge_index, dim=1)
        
        return edge_index 

    def compute_edge_attr(self,batch:Union[Batch,Data]) -> torch.Tensor:
        '''
        Compute edge_attr in the either Batch or Data

        Returns:
            edge_attr (Tensor): the shape is [num_nodes,5].
        '''
        source_indice = batch.edge_index[0]
        target_indice = batch.edge_index[1]
        
        source_x      = batch.x[source_indice]
        target_x      = batch.x[target_indice]

        source_info   = batch.location_info[source_indice]
        target_info   = batch.location_info[target_indice]

        # location_info = [w,h,xc,yc]
        feat1 = 2 * (source_info[:,-2] - target_info[:,-2]) / (source_info[:,-3] + target_info[:,-3] + 1e-8)
        feat2 = 2 * (source_info[:,-1] - target_info[:,-1]) / (source_info[:,-3] + target_info[:,-3] + 1e-8)
        feat3 = torch.log(source_info[:,-3] / (target_info[:,-3] + 1e-8) )
        feat4 = torch.log(source_info[:,-4] / (target_info[:,-4] + 1e-8) )
        feat5 = F.cosine_similarity(source_x,target_x,dim=1)
        edge_attr = torch.stack([feat1,feat2,feat3,feat4,feat5],dim=1)

        return edge_attr


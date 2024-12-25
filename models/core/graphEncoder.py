#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Union
from torchvision import models
import torch.nn.functional as F
from models.graphToolkit import knn
from torch_geometric.data import Batch,Data
import torchvision.transforms.functional as T
from models.core.fastReid import load_fastreid_model



__all__ = ['NodeEncoder','EdgeEncoder']

# class NodeEncoder(nn.Module):
    
#     def __init__(self, fastreid_model:str,node_embed_size:int):
#         super(NodeEncoder, self).__init__()
#         # get the pretrained densenet model
#         self.backbone  = load_fastreid_model(fastreid_model)
        
#         self.head      = nn.Sequential(
#             nn.Linear(2048,512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512,128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128,node_embed_size),
#             nn.ReLU(inplace=True),
#         )
#         # freeze model weights
#         for param in self.backbone.parameters():
#             param.requires_grad = False # Freeze

#     def forward(self, graph:Union[Batch,Data]) -> Union[Batch,Data]:
#         with torch.no_grad():
#             graph.x = T.normalize(graph.x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
#             self.backbone.eval()
#             graph.x = self.backbone(graph.x)
#         graph.x = self.head(graph.x)
        
#         return graph
class NodeEncoder(nn.Module):
    
    def __init__(self, fastreid_model:str,node_embed_size:int):
        super(NodeEncoder, self).__init__()
        # get the pretrained densenet model
        backbone = models.densenet121(pretrained=True)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))
        
        self.head     = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1), # 1024
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128,node_embed_size),
            nn.ReLU(inplace=True),
        )
        #  freeze model weights
        params = list(self.backbone.parameters())

        for param in params:
            param.requires_grad = False

        for param in params[-3:]:
            param.requires_grad = True

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.head:
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, graph:Union[Batch,Data]) -> Union[Batch,Data]:
        graph.x = T.normalize(graph.x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        graph.x = self.backbone(graph.x)
        graph.x = self.head(graph.x)
        
        return graph       
class EdgeEncoder(nn.Module):

    def __init__(self, 
                edge_embed_size:int,
                bt_cosine :bool=False, 
                bt_self_loop :bool=True,
                bt_directed :bool=True
                ):
        super(EdgeEncoder, self).__init__()

        self.bt_cosine = bt_cosine
        self.bt_self_loop = bt_self_loop
        self.bt_directed = bt_directed

        self.encoder = nn.Sequential(
            nn.Linear(6,edge_embed_size),
            nn.BatchNorm1d(edge_embed_size),
            nn.ReLU(inplace=True),
            nn.Linear(edge_embed_size,edge_embed_size),
            nn.BatchNorm1d(edge_embed_size),
            nn.ReLU(inplace=True),
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.encoder:
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)    

    def forward(self,graph:Union[Batch,Data],k:int) -> Union[Batch,Data]:
        
        assert len(graph.x.shape) == 2 , 'Encode node attribute first!'
        
        with torch.no_grad():
            graph.edge_index = self.construct_edge_index(graph,k,bt_cosine=self.bt_cosine,bt_self_loop=self.bt_self_loop,bt_directed=self.bt_directed)
            raw_edge_attr    = self.compute_edge_attr(graph)
        
        graph.edge_attr = self.encoder(raw_edge_attr)
        
        return graph 

    def construct_edge_index(self,batch: Union[Batch,Data], k ,bt_cosine:bool=False, bt_self_loop: bool=True,bt_directed :bool=False) -> torch.Tensor:
        """
        Construct edge_index in either the Batch or Data.
        > construct KNN for each subgraph in the Batch
        
        Args:
            batch (Batch): Batch object containing multiple graphs.
            bt_cosine (bool): Whether to use cosine distance.
            bt_self_loop (bool): Whether to include self-loop.
            bt_directed (bool): return the directed graph or the undirected one.

            
        Returns:
            edge_index (Tensor): Edge indices of KNN for all graphs. 'soure_to_target'
        """

        if not hasattr(batch,'num_graphs'): # Date Type
            edge_index = knn(batch.location_info[:,-2:],k, bt_cosine=bt_cosine, bt_self_loop=bt_self_loop,bt_directed=bt_directed)
            return edge_index
        
        # Batch Type
        all_edge_index = []
        for i in range(batch.num_graphs):
            start, end = batch.ptr[i:i+2]
            
            sub_positions = batch.location_info[start:end,-2:]
            
            edge_index = knn(sub_positions, k, bt_cosine=bt_cosine, bt_self_loop=bt_self_loop,bt_directed=bt_directed)
            
            all_edge_index.append(edge_index + start)
        
        edge_index = torch.cat(all_edge_index, dim=1)
        
        return edge_index 

    def compute_edge_attr(self,batch:Union[Batch,Data],flow:str='source_to_target') -> torch.Tensor:
        '''
        Compute edge_attr in the either Batch or Data

        Returns:
            edge_attr (Tensor): the shape is [num_nodes,5].
        '''
        
        if flow == 'source_to_target':
            source_indice = batch.edge_index[0]
            target_indice = batch.edge_index[1]
        elif flow == 'target_to_source':
            source_indice = batch.edge_index[1]
            target_indice = batch.edge_index[0]
        else:
            raise ValueError('flow must be either source_to_target or target_to_source')
        
        source_x      = batch.x[source_indice]
        target_x      = batch.x[target_indice]

        source_info   = batch.location_info[source_indice]
        target_info   = batch.location_info[target_indice]

        # location_info = [x,y,x2,y2,w,h,xc,yc]
        feat1 = 2 * (source_info[:,-2] - target_info[:,-2]) / (source_info[:,-3] + target_info[:,-3])
        feat2 = 2 * (source_info[:,-1] - target_info[:,-1]) / (source_info[:,-3] + target_info[:,-3])
        feat3 = torch.log(source_info[:,-3] / (target_info[:,-3]) )
        feat4 = torch.log(source_info[:,-4] / (target_info[:,-4]) )
        feat5 = 1 - self._calculate_diou(source_info,target_info)
        feat6 = 1 - F.cosine_similarity(source_x,target_x,dim=1)

        edge_attr = torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim=1)

        return edge_attr

    def _calculate_diou(self,source_info, target_info):
        # location_info = [x,y,x2,y2,w,h,xc,yc]
        inter_x1 = torch.max(source_info[:,0], target_info[:,0])
        inter_y1 = torch.max(source_info[:,1], target_info[:,1])
        inter_x2 = torch.min(source_info[:,2], target_info[:,2])
        inter_y2 = torch.min(source_info[:,3], target_info[:,3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        source_area = source_info[:, 4] * source_info[:, 5]
        target_area = target_info[:, 4] * target_info[:, 5]
        union_area = source_area + target_area - inter_area

        inter_xc = (inter_x1 + inter_x2) / 2
        inter_yc = (inter_y1 + inter_y2) / 2

        source_xc = source_info[:, 6]
        source_yc = source_info[:, 7]
        target_xc = target_info[:, 6]
        target_yc = target_info[:, 7]
        union_xc = (source_xc * source_area + target_xc * target_area) / (union_area + 1e-8)
        union_yc = (source_yc * source_area + target_yc * target_area) / (union_area + 1e-8)

        dist = torch.sqrt((inter_xc - union_xc) ** 2 + (inter_yc - union_yc) ** 2)

        diou = inter_area / (union_area + 1e-8) - dist ** 2 / (union_area + 1e-8)

        return diou
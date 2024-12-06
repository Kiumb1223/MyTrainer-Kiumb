#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
'''
import torch
import torch.nn as nn
from typing import Union
from torch.nn import Module
from models.graphToolkit import knn
from torch_geometric.data import Batch,Data
from torch_geometric.nn import MessagePassing
from models.core.graphEncoder import NodeEncoder,EdgeEncoder

__all__ = ['GraphConv']

class StaticConv(MessagePassing):

    def __init__(self, in_channels:int, out_channels:int):

        super().__init__(aggr='max') #  "Max" aggregation.
        if out_channels % 2 != 0:
            mid_channels = out_channels // 2
        else:
            mid_channels = (in_channels + out_channels) // 2  
        self.updateFunc = nn.Sequential(
            nn.Linear(in_channels,mid_channels,bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(mid_channels,out_channels,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(out_channels,out_channels,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x:torch.Tensor, edge_index:torch.Tensor,edge_attr:torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index,edge_attr=edge_attr,x=x)
    
    def message(self, x_i:torch.Tensor, x_j:torch.Tensor,edge_attr:torch.Tensor) -> torch.Tensor:
        return torch.cat([x_i,x_j - x_i,edge_attr], dim=1)  
    
    def update(self, inputs:torch.Tensor) -> torch.Tensor:
        return self.updateFunc(inputs)
    

class DynamicGonv(MessagePassing):
    def __init__(self, in_channels:int, out_channels:int):

        super().__init__(aggr='max')

        self.msgFunc = nn.Sequential(
            nn.Linear(in_channels,out_channels,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels,out_channels,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self,x:torch.Tensor,k:int) -> torch.Tensor:
        assert self.flow == 'source_to_target' 

        with torch.no_grad():
            edge_index = knn(x,k,bt_cosine=True,bt_self_loop=True,bt_edge_index=True) 
            
        return self.propagate(edge_index,x=x)
    
    def message(self, x_i:torch.Tensor,x_j:torch.Tensor) -> torch.Tensor:
        tmp_msg = torch.cat([x_i,x_j-x_i],dim=1)
        return self.msgFunc(tmp_msg)
    
class GraphConv(Module):
    def __init__(self,node_embed_size:int,edge_embed_size:int):
        super().__init__()

        self.nodeEncoder = NodeEncoder(node_embed_size)
        self.edgeEncoder = EdgeEncoder(edge_embed_size)

        self.sg1Func  = StaticConv(2*node_embed_size + edge_embed_size,node_embed_size)
        self.sg2Func  = StaticConv(2*node_embed_size + edge_embed_size,node_embed_size*2)
        self.sg3Func  = StaticConv(2*(node_embed_size*2) + edge_embed_size,node_embed_size*3)

        self.dg1Func  = DynamicGonv(2*node_embed_size,node_embed_size*2)
        self.dg2Func  = DynamicGonv(2*(node_embed_size*2),node_embed_size*3)

        self.fuse1Func = nn.Sequential(
            nn.Linear(node_embed_size*11,1024,bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.fuse2Func = nn.Sequential(
            nn.Linear(node_embed_size*11+1024,512,bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256,bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,128,bias=False),
        )
    def forward(self,graph:Union[Batch,Data],k:int) -> torch.Tensor:
        #---------------------------------#
        # Initialize the Node and edge embeddings
        #---------------------------------#
        
        if len(graph.x.shape) != 2  :   # [N, node_embed_size] or [N,3,224,224]
            graph = self.nodeEncoder(graph)
        
        graph = self.edgeEncoder(graph,k)

        node_embedding = graph.x
        edge_embedding = graph.edge_attr
        edge_index     = graph.edge_index
        

        node_embedding_sg1 = self.sg1Func(node_embedding,edge_index,edge_embedding)      # torch.Size([32, 32])
        node_embedding_sg2 = self.sg2Func(node_embedding_sg1,edge_index,edge_embedding)  # torch.Size([32, 64])
        node_embedding_sg3 = self.sg3Func(node_embedding_sg2,edge_index,edge_embedding)  # torch.Size([32, 96])


        node_embedding_dg1 = self.dg1Func(node_embedding_sg1,k)  # torch.Size([32,64])
        node_embedding_dg2 = self.dg2Func(node_embedding_dg1,k)  # torch.Size([32, 96])


        node_embedding_cat1  = torch.cat([node_embedding_sg1,node_embedding_dg1,node_embedding_dg2,
                                          node_embedding_sg2,node_embedding_sg3],dim=1)     # torch.Size([32, 352])
        node_embedding_fuse1 = self.fuse1Func(node_embedding_cat1)  # torch.Size([32, 1024])
        node_embedding_cat2  = torch.cat([node_embedding_fuse1,node_embedding_cat1],dim=1)  # torch.Size([32, 1376])
        node_embedding_output= self.fuse2Func(node_embedding_cat2)  # torch.Size([32, 128])
        
        return node_embedding_output

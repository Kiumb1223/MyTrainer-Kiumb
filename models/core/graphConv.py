#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
'''
import torch
import torch.nn as nn
from torch.nn import Module
from models.graphToolkit import knn
from torch_geometric.nn import MessagePassing

__all__ = ['GraphConv']

class StaticConv(MessagePassing):

    def __init__(self, in_channels:int, out_channels:int,node_embed_size:int):

        super().__init__(aggr='max') #  "Max" aggregation.
        if out_channels % 2 != 0:
            mid_channels = out_channels // 2
        else:
            mid_channels = (in_channels + out_channels) // 2  
        self.msgFunc = nn.Sequential(
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

        self.lin = nn.Sequential(
            nn.Linear( node_embed_size ,out_channels,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Linear( out_channels ,out_channels,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data,a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x:torch.Tensor, edge_index:torch.Tensor,edge_attr:torch.Tensor) -> torch.Tensor:
        return self.lin(x) + self.propagate(edge_index,edge_attr=edge_attr,x=x)
        # return self.propagate(edge_index,edge_attr=edge_attr,x=x)
    
    def message(self, x_i:torch.Tensor, x_j:torch.Tensor,edge_attr:torch.Tensor) -> torch.Tensor:
        '''
        x_i : target nodes 
        x_j : source nodes
        '''
        return self.msgFunc(torch.cat([x_i,x_j - x_i,edge_attr], dim=1))
    
    # def update(self, inputs:torch.Tensor) -> torch.Tensor:
    #     return self.updateFunc(inputs)
    

class DynamicGonv(MessagePassing):
    def __init__(self, in_channels:int, out_channels:int,node_embed_size:int,
                 bt_cosine :bool=False, bt_self_loop :bool=True, bt_directed :bool=True):

        super().__init__(aggr='max')

        self.bt_cosine = bt_cosine
        self.bt_self_loop = bt_self_loop
        self.bt_directed = bt_directed

        self.msgFunc = nn.Sequential(
            nn.Linear(in_channels,out_channels,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels,out_channels,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )

        self.lin = nn.Sequential(
            nn.Linear( node_embed_size ,out_channels,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear( out_channels ,out_channels,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data,a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self,x:torch.Tensor,k:int) -> torch.Tensor:
        assert self.flow == 'source_to_target' 

        with torch.no_grad():
            edge_index = knn(x,k,bt_cosine=self.bt_cosine,bt_self_loop=self.bt_self_loop,bt_directed=self.bt_directed) 
        out = self.propagate(edge_index,x=x)
        return self.lin(x) + out
        # return out
    
    def message(self, x_i:torch.Tensor,x_j:torch.Tensor) -> torch.Tensor:
        '''
        x_i : target nodes 
        x_j : source nodes
        '''
        return self.msgFunc(torch.cat([x_i,x_j-x_i],dim=1))
    

class GraphConv(Module):
    def __init__(self,node_embed_size:int,edge_embed_size:int,
                bt_cosine :bool=False, bt_self_loop :bool=True, bt_directed :bool=True
                ):

        super().__init__()

        self.sg1Func  = StaticConv(2*node_embed_size + edge_embed_size,node_embed_size,node_embed_size)
        self.sg2Func  = StaticConv(2*node_embed_size + edge_embed_size,node_embed_size*2,node_embed_size)
        self.sg3Func  = StaticConv(2*(node_embed_size*2) + edge_embed_size,node_embed_size*3,(node_embed_size*2))

        self.dg1Func  = DynamicGonv(2*node_embed_size,node_embed_size*2,node_embed_size,bt_cosine,bt_self_loop,bt_directed)
        self.dg2Func  = DynamicGonv(2*(node_embed_size*2),node_embed_size*3,(node_embed_size*2),bt_cosine,bt_self_loop,bt_directed)

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

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.fuse1Func:
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight.data,a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.fuse2Func:
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight.data,a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,node_embedding:torch.Tensor,edge_embedding:torch.Tensor,edge_index:torch.Tensor,k:int) -> torch.Tensor:

        node_embedding_sg1 = self.sg1Func(node_embedding,edge_index,edge_embedding)      # torch.Size([32, 32])
        node_embedding_sg2 = self.sg2Func(node_embedding_sg1,edge_index,edge_embedding)  # torch.Size([32, 64])
        node_embedding_sg3 = self.sg3Func(node_embedding_sg2,edge_index,edge_embedding)  # torch.Size([32, 96])


        node_embedding_dg1 = self.dg1Func(node_embedding_sg1,k)  # torch.Size([32,64])
        node_embedding_dg2 = self.dg2Func(node_embedding_dg1,k)  # torch.Size([32,96])


        node_embedding_cat1  = torch.cat([node_embedding_sg1,node_embedding_dg1,node_embedding_dg2,
                                          node_embedding_sg2,node_embedding_sg3],dim=1)     # torch.Size([32, 352])
        node_embedding_fuse1 = self.fuse1Func(node_embedding_cat1)  # torch.Size([32, 1024])
        node_embedding_cat2  = torch.cat([node_embedding_fuse1,node_embedding_cat1],dim=1)  # torch.Size([32, 1376])
        node_embedding_output= self.fuse2Func(node_embedding_cat2)  # torch.Size([32, 128])
        
        return node_embedding_output

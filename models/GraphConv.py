#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
'''
import torch
import torch.nn as nn
from torch.nn import Module
from models.GraphEncode import NodeEncoder,EdgeEncoder
from torch_geometric.nn import MessagePassing,knn_graph

__all__ = ['GraphConv']

class StaticConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
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
    def forward(self, x, edge_index,edge_attr):
        return self.propagate(edge_index,edge_attr=edge_attr,x=x)
    def message(self, x_i, x_j,edge_attr):
        return torch.cat([x_i,x_j - x_i,edge_attr], dim=1)  
    def update(self, inputs):
        return self.updateFunc(inputs)
    

class DynamicGonv(MessagePassing):
    def __init__(self, in_channels, out_channels,num_works = 1):
        super().__init__(aggr='max')
        self.num_works = num_works
        self.msgFunc = nn.Sequential(
            nn.Linear(in_channels,out_channels,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels,out_channels,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self,x,k):
        # edge_index = knn(x, x,k,num_workers=self.num_works)
        # edge_index,_ = remove_self_loops(edge_index)
        # Avoiding numerical overflow in Mixed Precision Training
        edge_index = knn_graph(x.float(),k,loop=False,cosine=False,num_workers=self.num_works)
        return self.propagate(edge_index,x=x)
    def message(self, x_i,x_j):
        tmp_msg = torch.cat([x_j-x_i,x_i],dim=1)
        return self.msgFunc(tmp_msg)
    
class GraphConv(Module):
    def __init__(self,k,node_embed_size,edge_embed_size,num_works):
        super().__init__()
        self.k = k
        self.num_works = num_works

        self.nodeEncoder = NodeEncoder(node_embed_size)
        self.edgeEncoder = EdgeEncoder(edge_embed_size)

        self.sg1Func  = StaticConv(2*node_embed_size + edge_embed_size,node_embed_size)
        self.sg2Func  = StaticConv(2*node_embed_size + edge_embed_size,node_embed_size*2)
        self.sg3Func  = StaticConv(2*(node_embed_size*2) + edge_embed_size,node_embed_size*3)

        self.dg1Func  = DynamicGonv(2*node_embed_size,node_embed_size*2,self.num_works)
        self.dg2Func  = DynamicGonv(2*(node_embed_size*2),node_embed_size*3,self.num_works)

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
    def forward(self,graph):
        #---------------------------------#
        # Initialize the Node and edge embeddings
        #---------------------------------#
        graph.x = self.nodeEncoder(graph.x)
        graph.edge_attr = self.edgeEncoder(graph)

        node_embedding = graph.x
        edge_embedding = graph.edge_attr
        edge_index     = graph.edge_index
        
        #---------------------------------#
        # 
        #---------------------------------#
        node_embedding_sg1 = self.sg1Func(node_embedding,edge_index,edge_embedding)      # torch.Size([32, 32])
        node_embedding_sg2 = self.sg2Func(node_embedding_sg1,edge_index,edge_embedding)  # torch.Size([32, 64])
        node_embedding_sg3 = self.sg3Func(node_embedding_sg2,edge_index,edge_embedding)  # torch.Size([32, 96])

        #---------------------------------#
        #
        #---------------------------------#
        node_embedding_dg1 = self.dg1Func(node_embedding_sg1,self.k)  # torch.Size([32,64])
        node_embedding_dg2 = self.dg2Func(node_embedding_dg1,self.k)  # torch.Size([32, 96])

        #---------------------------------#
        #
        #---------------------------------#
        node_embedding_cat1  = torch.cat([node_embedding_sg1,node_embedding_dg1,node_embedding_dg2,\
                                          node_embedding_sg2,node_embedding_sg3],dim=1)     # torch.Size([32, 352])
        node_embedding_fuse1 = self.fuse1Func(node_embedding_cat1)  # torch.Size([32, 1024])
        node_embedding_cat2  = torch.cat([node_embedding_fuse1,node_embedding_cat1],dim=1)  # torch.Size([32, 1376])
        node_embedding_output= self.fuse2Func(node_embedding_cat2)  # torch.Size([32, 128])
        return node_embedding_output

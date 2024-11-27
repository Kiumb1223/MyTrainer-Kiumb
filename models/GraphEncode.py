#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
'''
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

__all__ = ['NodeEncoder','EdgeEncoder']


class NodeEncoder(nn.Module):
    def __init__(self, node_embed_size):
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

    def forward(self, images):
        out = self.backbone(images)
        out = self.head(out)
        return out
    
class EdgeEncoder(nn.Module):
    def __init__(self, Edge_embed_size):
        super(EdgeEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(5,18),
            nn.ReLU(inplace=True),
            nn.Linear(18,Edge_embed_size),
            nn.ReLU(inplace=True)
        )
    def forward(self,graph):
        source_nodes = graph['edge_index'][0]
        target_nodes = graph['edge_index'][1]
        source_feats = graph['x'][source_nodes]
        target_feats = graph['x'][target_nodes]
        cosine_similiary = F.cosine_similarity(source_feats,target_feats,dim=1)
        cosine_similiary = cosine_similiary.unsqueeze(1)
        update_edge_attr = torch.cat([graph['edge_attr'],cosine_similiary],dim=1)
        update_edge_attr = self.encoder(update_edge_attr)

        return update_edge_attr

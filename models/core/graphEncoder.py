#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Union
from torchvision import models
import torch.nn.functional as F
from torch_geometric.data import Batch,Data
import torchvision.transforms.functional as T
from models.graphToolkit import knn,calc_iouFamily
from models.core.fastReid import load_fastreid_model


__all__ = ['SequentialBlock','NodeEncoder','EdgeEncoder']

class SequentialBlock(nn.Module):
    '''
    A dynamic module constructor that uses `layer_type` to dynamically select layer types 
    and build a configurable network.
    '''
    def __init__(self, in_dim :int, out_dim :int, hidden_dim:Union[list,tuple], 
                 layer_type :str, layer_bias :bool,
                 use_batchnorm :bool ,activate_func :str, lrelu_slope:float =0.0,
                 final_activation :bool=True):
        '''
        :param in_dim: Input dimension of the first layer.
        :param out_dims: A list of output dimensions for each layer (supports multiple layers).
        :param layer_type: The type of layer to use, e.g., 'linear' (fully connected), 'conv1d'.
        :param layer_bias: Whether to use bias in Conv1d or Linear layers.        
        :param batch_norm: Whether to add BatchNorm after each layer.
        :param activate_func: Activation function type, e.g., 'relu', 'lrelu', 'sigmoid', 'tanh', etc.
        :param lrelu_slope: Negative slope for LeakyReLU activation.
        :param final_activation: Whether to add activation after the last layer (default is True).
        '''
        super(SequentialBlock, self).__init__()

        layer_type = layer_type.lower()
        activate_func = activate_func.lower()

        activation_map = {
            'relu': nn.ReLU(inplace=True),
            'lrelu': nn.LeakyReLU(negative_slope=lrelu_slope, inplace=True),
        }
        
        assert isinstance(hidden_dim, (list, tuple)), 'modules_dims must be either a list or a tuple, but got {}'.format(type(hidden_dim))
        assert layer_type in ['linear','conv1d'] , f"Unsupported layer type: {layer_type}. "
        assert activate_func in activation_map , f"Unsupported activation function: {activate_func}. " + f"Supported functions are: {list(activation_map.keys())}"
        
        layers = []
        if out_dim is None:
            # If out_dim is None, use the last element of hidden_dim as the output dimension
            dims_list = hidden_dim
        else:
            dims_list = hidden_dim + [out_dim]
        activate_layer = activation_map[activate_func]
        length = len(dims_list)
        for cnt,dim in enumerate(dims_list):
            if layer_type == 'conv1d':
                layers.append(nn.Conv1d(in_dim, dim, kernel_size=1, bias=layer_bias))
            elif layer_type == 'linear':
                layers.append(nn.Linear(in_dim, dim, bias=layer_bias))

            if cnt < length - 1 or final_activation:
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(dim))
                layers.append(activate_layer)
            in_dim = dim
        self.layers = nn.Sequential(*layers)

        self._initialize_weights(lrelu_slope)

    def _initialize_weights(self,lrelu_slope):
        for m in self.layers:
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight.data,a=lrelu_slope)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data,a=lrelu_slope,mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        return self.layers(input)

class NodeEncoder(nn.Module):
    ''' graph-in and graph-out Module'''
    def __init__(self, node_model_dict :dict):
        
        super(NodeEncoder, self).__init__()
        in_dim , *mid_dim , out_dim = node_model_dict['dims_list']
        self.head = nn.ModuleList()
        self.backbone = self.gen_backbone(node_model_dict['backbone'],node_model_dict['wight_path'])
        if node_model_dict['backbone'] == 'densenet121':
            self.head = nn.Sequential(
                nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(1),  # 1024
                ),
                SequentialBlock(
                    in_dim, out_dim, mid_dim,
                    node_model_dict['layer_type'], node_model_dict['layer_bias'],
                    node_model_dict['use_batchnorm'], node_model_dict['activate_func']
                )
            )
        else:
            self.head = SequentialBlock(
                    in_dim, out_dim, mid_dim,
                    node_model_dict['layer_type'], node_model_dict['layer_bias'],
                    node_model_dict['use_batchnorm'], node_model_dict['activate_func']
                )
        #  freeze model weights
        params = list(self.backbone.parameters())
        for param in params:
            param.requires_grad = False

        for param in params[-3:]:
            param.requires_grad = True


    def gen_backbone(self,backbone:str,weight_path:str):
        assert backbone in ['densenet121','fastreid']
        if backbone == 'densenet121':
            backbone_tmp = models.densenet121(pretrained=True)
            backbone = nn.Sequential(*(list(backbone_tmp.children())[:-1]))
            return backbone

    def forward(self, graph :Union[Batch,Data]) -> Union[Batch,Data]:
        graph.x = T.normalize(graph.x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        graph.x = self.backbone(graph.x)
        graph.x = self.head(graph.x)
        
        return graph   
        
class EdgeEncoder(nn.Module):
    ''' graph-in and graph-out Module'''
    def __init__(self, edge_model_dict :dict):
        super(EdgeEncoder, self).__init__()
        # assert edge_model_dict['edge_type'] in ["ImgNorm4","SrcNorm4","TgtNorm4", "MeanSizeNorm4", "MeanHeightNorm4",
        #         "MeanWidthNorm4","CorverxNorm4","MaxNorm4","IOU5", "DIOU5", "DIOU-Cos6", "IouFamily8"]
        in_dim , *mid_dim , out_dim = edge_model_dict['dims_list']
        
        self.edge_type    = edge_model_dict['edge_type']
        self.bt_cosine    = edge_model_dict['bt_cosine']
        self.bt_self_loop = edge_model_dict['bt_self_loop']
        self.bt_directed  = edge_model_dict['bt_directed']

        self.encoder = SequentialBlock(in_dim, out_dim, mid_dim,
                        edge_model_dict['layer_type'], edge_model_dict['layer_bias'],
                        edge_model_dict['use_batchnorm'],edge_model_dict['activate_func'])
  
    def forward(self,graph:Union[Batch,Data],k:int) -> Union[Batch,Data]:
        
        assert len(graph.x.shape) == 2 , 'Encode node attribute first!'
        
        with torch.no_grad():
            graph.edge_index = self.construct_edge_index(graph,k,
                                bt_cosine=self.bt_cosine,bt_self_loop= self.bt_self_loop,bt_directed=self.bt_directed)
            raw_edge_attr    = self.compute_edge_attr(graph)
        
        graph.edge_attr = self.encoder(raw_edge_attr)
        
        return graph 
    
    @staticmethod
    def construct_edge_index(batch: Union[Batch,Data], k, bt_cosine: bool=False,
        bt_self_loop: bool=False,bt_directed: bool=True) -> torch.Tensor:
        """
        Construct edge_index in either the Batch or Data.
        > construct KNN for each subgraph in the Batch
        
        Args:
            batch (Batch): Batch object containing multiple graphs.
            bt_cosine (bool): Whether to use cosine distance.
            bt_self_loop (bool): Whether to include self-loop (i.e., whether to consider itself as its own neighbor).
            bt_directed (bool): return the directed graph or the undirected one.

            
        Returns:
            edge_index (Tensor): Edge indices of KNN for all graphs. 'soure_to_target'
        """

        if not hasattr(batch,'num_graphs'): # Date Type
            edge_index = knn(batch.location_info[:,6:8],k, bt_cosine=bt_cosine,bt_self_loop= bt_self_loop,bt_directed=bt_directed)
            return edge_index
        
        # Batch Type
        all_edge_index = []
        for i in range(batch.num_graphs):
            start, end = batch.ptr[i:i+2]
            
            sub_positions = batch.location_info[start:end,6:8]
            
            edge_index = knn(sub_positions, k, bt_cosine= bt_cosine,bt_self_loop= bt_self_loop,bt_directed= bt_directed)
            
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

        return self._calc_edge_type(source_info,target_info,source_x,target_x)

    def _calc_edge_type(self,source_info,target_info,source_x,target_x):

        # location_info = [x,y,x2,y2,w,h,xc,yc,W,H]
        if self.edge_type == 'ImgNorm4':
            feat1 = (source_info[:,6] - target_info[:,6]) /  source_info[:,8]
            feat2 = (source_info[:,7] - target_info[:,7]) /  source_info[:,9]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            return torch.stack([feat1,feat2,feat3,feat4],dim =1)
        if self.edge_type == 'SrcNorm4':
            feat1 = (source_info[:,6] - target_info[:,6]) /  source_info[:,4]
            feat2 = (source_info[:,7] - target_info[:,7]) /  source_info[:,5]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            return torch.stack([feat1,feat2,feat3,feat4],dim =1)
        if self.edge_type == 'TgtNorm4':
            feat1 = (source_info[:,6] - target_info[:,6]) /  target_info[:,4]
            feat2 = (source_info[:,7] - target_info[:,7]) /  target_info[:,5]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            return torch.stack([feat1,feat2,feat3,feat4],dim =1)
        if self.edge_type == 'MeanSizeNorm4':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,4] + target_info[:,4])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            return torch.stack([feat1,feat2,feat3,feat4],dim =1)
        if self.edge_type == 'MeanHeightNorm4':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            return torch.stack([feat1,feat2,feat3,feat4],dim =1)
        if self.edge_type == 'MeanWidthNorm4':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,4] + target_info[:,4])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,4] + target_info[:,4])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            return torch.stack([feat1,feat2,feat3,feat4],dim =1)
        if self.edge_type == 'ConvexNorm4':
            converx_bbox_lt = torch.max(source_info[:, 2:4], target_info[:, 2:4])
            converx_bbox_rb = torch.min(source_info[:, :2], target_info[:, :2])
            converx_bbox_wh = torch.clamp((converx_bbox_lt - converx_bbox_rb), min=0)  # smallest enclosing bbox 
            feat1 = source_info[:,6] - target_info[:,6] /  converx_bbox_wh[:, 0]
            feat2 = source_info[:,7] - target_info[:,7] /  converx_bbox_wh[:, 1]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            return torch.stack([feat1,feat2,feat3,feat4],dim =1)
        if self.edge_type == 'MaxNorm4':
            max_bbox_wh = torch.max(source_info[:, 4:6], target_info[:, 4:6])
            feat1 = source_info[:,6] - target_info[:,6] /  max_bbox_wh[:, 0]
            feat2 = source_info[:,7] - target_info[:,7] /  max_bbox_wh[:, 1]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            return torch.stack([feat1,feat2,feat3,feat4],dim =1)
        if self.edge_type == 'IOUd5':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='iou')
            return torch.stack([feat1,feat2,feat3,feat4,feat5],dim =1)
        if self.edge_type == 'IOU5':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = calc_iouFamily(source_info,target_info,iou_type='iou')
            return torch.stack([feat1,feat2,feat3,feat4,feat5],dim =1)
        if self.edge_type == 'GIOUd5':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            return torch.stack([feat1,feat2,feat3,feat4,feat5],dim =1)
        if self.edge_type == 'GIOU5':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = calc_iouFamily(source_info,target_info,iou_type='giou')
            return torch.stack([feat1,feat2,feat3,feat4,feat5],dim =1)
        if self.edge_type == 'DIOUd5':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1- calc_iouFamily(source_info,target_info,iou_type='diou')
            return torch.stack([feat1,feat2,feat3,feat4,feat5],dim =1)
        if self.edge_type == 'DIOU5':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = calc_iouFamily(source_info,target_info,iou_type='diou')
            return torch.stack([feat1,feat2,feat3,feat4,feat5],dim =1)
        if self.edge_type == 'CIOUd5':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1- calc_iouFamily(source_info,target_info,iou_type='ciou')
            return torch.stack([feat1,feat2,feat3,feat4,feat5],dim =1)
        if self.edge_type == 'CIOU5':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = calc_iouFamily(source_info,target_info,iou_type='ciou')
            return torch.stack([feat1,feat2,feat3,feat4,feat5],dim =1)
        if self.edge_type == 'EIOUd5':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1- calc_iouFamily(source_info,target_info,iou_type='eiou')
            return torch.stack([feat1,feat2,feat3,feat4,feat5],dim =1)
        if self.edge_type == 'EIOU5':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = calc_iouFamily(source_info,target_info,iou_type='eiou')
            return torch.stack([feat1,feat2,feat3,feat4,feat5],dim =1)
        if self.edge_type == 'DIOUd-Cos6':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='diou')
            feat6 = 1 - F.cosine_similarity(source_x,target_x,dim=1)
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)
        if self.edge_type == 'IouFamily8_vanilla':

            #---------------------------------#
            #  Info about smallest enclosing bbox
            #---------------------------------#
            converx_bbox_lt = torch.max(source_info[:, 2:4], target_info[:, 2:4])
            converx_bbox_rb = torch.min(source_info[:, :2], target_info[:, :2])   
            converx_bbox_wh = torch.clamp((converx_bbox_lt - converx_bbox_rb), min=0)   # converx bbox 

            outer_diag = (converx_bbox_wh[:, 0] ** 2) + (converx_bbox_wh[:, 1] ** 2)    # convex diagonal squard length
            inter_diag = (source_info[:, 6] - target_info[:, 6]) ** 2 + (source_info[:, 7] - target_info[:, 7]) ** 2

            converx_bbox_w_square = converx_bbox_wh[:,0] ** 2 
            converx_bbox_h_square = converx_bbox_wh[:,1] ** 2 
            dis_w =  (source_info[:, 4] - target_info[:, 4]) ** 2
            dis_h =  (source_info[:, 5] - target_info[:, 5]) ** 2

            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            feat4 = inter_diag / ( outer_diag + 1e-8 )

            feat5 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat6 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat7 = dis_w / (converx_bbox_w_square + 1e-8)
            feat8 = dis_h / (converx_bbox_h_square + 1e-8)
            
            return torch.stack([feat1,feat2,feat3,feat4,
                                feat5,feat6,feat7,feat8],dim =1)
        
        if self.edge_type == 'IouFamily8_convex':

            #---------------------------------#
            #  Info about smallest enclosing bbox
            #---------------------------------#
            converx_bbox_lt = torch.max(source_info[:, 2:4], target_info[:, 2:4])
            converx_bbox_rb = torch.min(source_info[:, :2], target_info[:, :2])   
            converx_bbox_wh = torch.clamp((converx_bbox_lt - converx_bbox_rb), min=0)   # convex bbox 

            outer_diag = (converx_bbox_wh[:, 0] ** 2) + (converx_bbox_wh[:, 1] ** 2)    # convex diagonal squard length
            inter_diag = (source_info[:, 6] - target_info[:, 6]) ** 2 + (source_info[:, 7] - target_info[:, 7]) ** 2

            converx_bbox_w_square = converx_bbox_wh[:,0] ** 2 
            converx_bbox_h_square = converx_bbox_wh[:,1] ** 2 
            dis_w =  (source_info[:, 4] - target_info[:, 4]) ** 2
            dis_h =  (source_info[:, 5] - target_info[:, 5]) ** 2

            feat1 = source_info[:,6] - target_info[:,6] /  converx_bbox_wh[:, 0]
            feat2 = source_info[:,7] - target_info[:,7] /  converx_bbox_wh[:, 1]
            feat3 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            feat4 = inter_diag / ( outer_diag + 1e-8 )

            feat5 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat6 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat7 = dis_w / (converx_bbox_w_square + 1e-8)
            feat8 = dis_h / (converx_bbox_h_square + 1e-8)
            
            return torch.stack([feat1,feat2,feat3,feat4,
                                feat5,feat6,feat7,feat8],dim =1)
        if self.edge_type == 'IouFamily8-separate':

            #---------------------------------#
            #  Info about smallest enclosing bbox
            #---------------------------------#
            converx_bbox_lt = torch.max(source_info[:, 2:4], target_info[:, 2:4])
            converx_bbox_rb = torch.min(source_info[:, :2], target_info[:, :2])   
            converx_bbox_wh = torch.clamp((converx_bbox_lt - converx_bbox_rb), min=0)   # convex bbox 

            outer_diag = (converx_bbox_wh[:, 0] ** 2) + (converx_bbox_wh[:, 1] ** 2)    # convex diagonal squard length
            inter_diag = (source_info[:, 6] - target_info[:, 6]) ** 2 + (source_info[:, 7] - target_info[:, 7]) ** 2

            dis_w =  (source_info[:, 4] - target_info[:, 4]) ** 2
            dis_h =  (source_info[:, 5] - target_info[:, 5]) ** 2
            max_bbox_wh_square = torch.max(source_info[:, 4:6], target_info[:, 4:6]) ** 2 

            feat1 = source_info[:,6] - target_info[:,6] /  converx_bbox_wh[:, 0]
            feat2 = source_info[:,7] - target_info[:,7] /  converx_bbox_wh[:, 1]
            feat3 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            feat4 = inter_diag / ( outer_diag + 1e-8 )

            feat5 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat6 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat7 = dis_w / (max_bbox_wh_square[:,0] + 1e-8)
            feat8 = dis_h / (max_bbox_wh_square[:,1] + 1e-8)
            
            return torch.stack([feat1,feat2,feat3,feat4,
                                feat5,feat6,feat7,feat8],dim =1)


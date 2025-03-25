#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import torch
import torch.nn as nn
from functools import partial
from torchvision import models
import torch.nn.functional as F
from typing import Union,Optional
import torch_geometric.nn.norm as norm
from torch_geometric.data import Batch,Data
from torch_geometric.nn import MessagePassing
import torchvision.transforms.functional as T
from models.graphToolkit import knn,calc_iouFamily
from models.core.fastReid import load_fastreid_model,load_ckpt_FastReid

__all__ = ['SequentialBlock','AffinityLayer','NodeEncoder','NodeUpdater','EdgeUpdater','EdgeEncoder']

def parse_layer_dimension(layer_dims: Union[list, tuple]):
    """
    Parses the input layer dimensions and returns the input, output, and hidden dimensions.

    This utility function is designed to handle various configurations of `layer_dims`, 
    ensuring robust parsing for layer dimension specifications. The input can be a list 
    or tuple of integers representing the dimensions for a sequence of layers.

    Args:
        layer_dims (Union[list, tuple]): A list or tuple containing layer dimensions.
            - If empty, returns (None, None, None).
            - If it contains one item, it is treated as both the input and output dimensions 
              (hidden dimensions will be an empty list).
            - If it contains multiple items, the first is treated as the input dimension,
              the last as the output dimension, and the middle items as hidden dimensions.

    Returns:
        tuple: A tuple containing:
            - in_dim (int or None): The input dimension. None if `layer_dims` is empty.
            - out_dim (int or None): The output dimension. None if `layer_dims` is empty.
            - hidden_dim (list): A list of hidden dimensions. Empty list if only one dimension is provided.

    Raises:
        AssertionError: If `layer_dims` is not a list or tuple.

    """

    assert isinstance(layer_dims,(list,tuple)), "layer_dims must be a list or tuple"
    
    if len(layer_dims) == 0:
        return None, None, None
    elif len(layer_dims) == 1:
        return layer_dims[0], layer_dims[0], []
    
    in_dim , *hidden_dim , out_dim = layer_dims
    return in_dim, hidden_dim, out_dim 

class l2Norm(nn.Module):
    def __init__(self,dim=1,eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
    def forward(self,x):
        return x / (torch.norm(x,p=2,dim=self.dim,keepdim=True) + self.eps)


class SequentialBlock(nn.Module):
    '''
    A dynamic module constructor that uses `layer_type` to dynamically select layer types 
    and build a configurable network.
    '''
    def __init__(self,
            dims_list  :Union[list,tuple], 
            layer_type :str, layer_bias :bool,
            norm_type  :str ,
            activate_func :str, lrelu_slope:float =0.0,
            final_activation :bool=True
        ):
        '''
        :param dims_list: A list of  dimensions for each layer.
        :param layer_type: The type of layer to use, e.g., 'linear', 'conv1d'.
        :param layer_bias: Whether to use bias in Conv1d or Linear layers.        
        :param norm_type: Thy type of normalization layer to use , e.g., 'None','batchNorm' , 'LayerNorm' ,'graphNorm'.
        :param activate_func: Activation function type, e.g., 'relu', 'lrelu'.
        :param lrelu_slope: Negative slope for LeakyReLU activation.
        :param final_activation: Whether to add activation after the last layer (default is True).
        '''
        super(SequentialBlock, self).__init__()
        activation_map = {
            'relu': nn.ReLU(inplace=True),
            'lrelu': nn.LeakyReLU(negative_slope=lrelu_slope, inplace=True),
            'sigmoid': nn.Sigmoid(),
        }
        normalization_map = {
            'none'      : None,
            'batchnorm' : nn.BatchNorm1d,
            'layernorm' : nn.LayerNorm,
            'graphnorm' : norm.GraphNorm,
            'l2norm'    : l2Norm,
        }
        layer_type    = layer_type.lower()
        activate_func = activate_func.lower()
        norm_type     = norm_type.lower()
        
        assert layer_type in ['linear','conv1d'] , f"Unsupported layer type: {layer_type}. "
        assert activate_func in activation_map , f"Unsupported activation function: {activate_func}. " + f"Supported functions are: {list(activation_map.keys())}"
        assert norm_type in normalization_map , f"Unsupported normalization layer type: {norm_type}."
        
        assert len(dims_list) > 0 , "dims_list should not be empty"
        #---------------------------------#
        #  dims_list : [in_dim, hidden_dim , out_dim]
        #---------------------------------#
        in_dim = dims_list[0]
        if len(dims_list) > 1 :
            dims_list = dims_list[1:]

        layers = []

        length = len(dims_list)

        norm_layer     = normalization_map[norm_type]   
        activate_layer = activation_map[activate_func]
        
        for cnt,dim in enumerate(dims_list):
            if layer_type == 'conv1d':
                layers.append(nn.Conv1d(in_dim, dim, kernel_size=1, bias=layer_bias))
            elif layer_type == 'linear':
                layers.append(nn.Linear(in_dim, dim, bias=layer_bias))

            if cnt < length - 1 or final_activation:
                if norm_layer is not None:
                    layers.append(
                        norm_layer(dim) if norm_type != 'l2norm' else norm_layer()
                    )
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
            elif isinstance(m,nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data,a=lrelu_slope,mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.BatchNorm1d) or isinstance(m,nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,norm.GraphNorm):
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()

    def forward(self,input,batch :Optional[torch.Tensor]=None):
        for layer in self.layers:
            if isinstance(layer,norm.GraphNorm):
                if input.dim() == 3 : # conv1d
                    input = layer(input.squeeze(-1),batch).unsqueeze(-1)
                else:  
                    input = layer(input,batch)
            else:
                input = layer(input)
        return input
    
class PositionEmbeddingSineGraph(nn.Module):
    """
    This class generates position embeddings for the nodes in a graph. The position encoding is based on the 
    center point coordinates of the bounding boxes (bboxes) of detected objects in the image.
    """

    def __init__(self, num_pos_feats, temperatureH=10000, temperatureW=10000, normalize=False, scale=None):
        super().__init__()

        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        
        if scale is None:
            scale = 2 * math.pi  # Default scaling factor
        self.scale = scale

    def forward(self, xcyc: torch.Tensor, wh_im: torch.Tensor):
        """
        Args:
            xcyc (Tensor): Tensor of shape (num_nodes, 2), representing the center coordinates (x, y)
                            of each bounding box in the image.
            wh_im (Tensor): Tensor of shape (num_nodes, 2), representing the width and height of the image for each node.
        
        Returns:
            Tensor: A tensor of position embeddings of shape (num_nodes, 2 * num_pos_feats)
        """
        # Normalize the coordinates to [0, 1] range based on image dimensions
        x_embed = xcyc[:, 0] / wh_im[:,0]
        y_embed = xcyc[:, 1] / wh_im[:,1]
        
        if self.normalize:
            eps = 1e-6
            # Scale the coordinates to the range [0, 2*pi]
            x_embed = x_embed * self.scale
            y_embed = y_embed * self.scale

        # Calculate the position encoding for x and y coordinates
        dim_tx = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xcyc.device)
        dim_tx = self.temperatureW ** (2 * (dim_tx // 2) / self.num_pos_feats)
        pos_x = x_embed[:, None] / dim_tx

        dim_ty = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xcyc.device)
        dim_ty = self.temperatureH ** (2 * (dim_ty // 2) / self.num_pos_feats)
        pos_y = y_embed[:, None] / dim_ty

        # Apply sine and cosine transformations
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=1).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=1).flatten(1)

        # Concatenate both position embeddings and return the final position encoding
        pos = torch.cat((pos_x, pos_y), dim=1)
        
        return pos

class AffinityLayer(nn.Module):
    def __init__(self, affinity_encode_model_dict :dict):
        super(AffinityLayer, self).__init__()

        self.use_attn = affinity_encode_model_dict['use_attn']
        self.softmax = nn.Softmax(dim=-1)
        self.linear = SequentialBlock(
                affinity_encode_model_dict['dims_list'],
                affinity_encode_model_dict['layer_type'], affinity_encode_model_dict['layer_bias'],
                affinity_encode_model_dict['norm_type'], 
                affinity_encode_model_dict['activate_func'],affinity_encode_model_dict['lrelu_slope']                
        )
    def forward(self, x):
        """
        x : [num_node_tra,num_node_det,3]
        output_tensor : [num_node_tra,num_node_det]
        """
        num_tra,num_det,c = x.shape

        if self.use_attn:
            #---------------------------------#
            #  self-attention mechanism
            #---------------------------------#
            x = x.reshape(num_tra * num_det,c)
            attn_scores = torch.matmul(x,x.transpose(-2,-1)) / c ** 0.5
            attn_prob   = self.softmax(attn_scores)
            x = torch.matmul(attn_prob,x) 
            x = x.reshape(num_tra,num_det,c)
        # linear 
        x = self.linear(x)
        return x.squeeze(-1)

    
class NodeEncoder(nn.Module):
    ''' graph-in and graph-out Module'''
    def __init__(self, node_encode_model_dict :dict):
        
        super(NodeEncoder, self).__init__()
        self.backbone_type = node_encode_model_dict['backbone']
        assert node_encode_model_dict['dims_list'] is not [] , '[node_encoder] dims_list is empty'


        self.backbone = self.gen_backbone(node_encode_model_dict['backbone'],node_encode_model_dict['weight_path'])
        if node_encode_model_dict['backbone'] == 'densenet121':
            self.head = nn.Sequential(
                nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(1),  # 1024
                ),
                SequentialBlock(
                    node_encode_model_dict['dims_list'],
                    node_encode_model_dict['layer_type'], node_encode_model_dict['layer_bias'],
                    node_encode_model_dict['norm_type'], 
                    node_encode_model_dict['activate_func'],node_encode_model_dict['lrelu_slope']
                )
            )
            #  freeze model weights
            params = list(self.backbone.parameters())
            for param in params:
                param.requires_grad = False

            for param in params[-3:]:
                param.requires_grad = True
        else:
            self.head = SequentialBlock(
                    node_encode_model_dict['dims_list'],
                    node_encode_model_dict['layer_type'], node_encode_model_dict['layer_bias'],
                    node_encode_model_dict['norm_type'], 
                    node_encode_model_dict['activate_func'],node_encode_model_dict['lrelu_slope']
                )
            #  freeze model weights
            for param in self.backbone.parameters():
                param.required_grad = False

        #---------------------------------#
        #  Position Encoding
        #---------------------------------#
        if node_encode_model_dict['bt_pe']:
            self.fusion_way = node_encode_model_dict['fusion_way']
            self.pos_encoder = PositionEmbeddingSineGraph(
                node_encode_model_dict['dims_list'][-1] // 2,
                node_encode_model_dict['temperatureH'],
                node_encode_model_dict['temperatureW'],
                node_encode_model_dict['normalize'],
                node_encode_model_dict['scale'],
            )
            #---------------------------------#
            # Linear Transform Layer
            #---------------------------------#
            # self.linear =nn.Linear(
            #     node_encode_model_dict['dims_list'][-1] * 2, node_encode_model_dict['dims_list'][-1],bias=False
            # )
            
        else:
            self.pos_encoder = None

    def gen_backbone(self,backbone:str,weight_path:str):
        # assert backbone in ['densenet121','fastreid']
        if backbone == 'densenet121':
            backbone_tmp = models.densenet121(pretrained=True)
            backbone = nn.Sequential(*(list(backbone_tmp.children())[:-1]))
            return backbone
        elif backbone.startswith('fastreid_'):
            backbone = load_fastreid_model(backbone)
            return load_ckpt_FastReid(backbone,weight_path)

    def forward(self, graph :Union[Data,Batch]) -> Union[Data,Batch]:
        if graph.x.dim() == 4 :
            if self.backbone_type.startswith('fastreid_'):
                self.backbone.eval()
                with torch.no_grad():
                    graph.x = self.backbone(graph.x)
            else:
                graph.x = T.normalize(graph.x , mean=[0.485*255, 0.456*255, 0.406*255], std=[0.229*255, 0.224*255, 0.225*255]) 
                graph.x = self.backbone(graph.x)
            graph.x = self.head(graph.x)
        
        graph.app = graph.x.clone()
        if self.pos_encoder:
            pe = self.pos_encoder(graph.geometric_info[:,6:8],graph.geometric_info[:,8:10]) 
            if self.fusion_way == 'add':
                graph.x = graph.x + pe
            elif self.fusion_way =='concate':
                graph.x = torch.cat([graph.x,pe],dim=-1)
                # graph.x = self.linear(graph.x)
        return graph


class NodeUpdater(MessagePassing):
    def __init__(self,
            idx : int, 
            node_update_model_dict: dict
        ):

        super().__init__(aggr=node_update_model_dict['aggr']) 
        
        msg_model_dict = node_update_model_dict['message_model']
        res_model_dict = node_update_model_dict['res_model']
        upd_model_dict = node_update_model_dict['update_model']

        self.msg_layer    = SequentialBlock(
                dims_list  = msg_model_dict['dims_list'][idx],
                layer_type = msg_model_dict['layer_type'], layer_bias = msg_model_dict['layer_bias'],
                norm_type  = msg_model_dict['norm_type'], 
                activate_func = msg_model_dict['activate_func'], lrelu_slope = msg_model_dict['lrelu_slope']
            )
        if res_model_dict['dims_list'][idx] :
            self.res_layer = SequentialBlock(
                    dims_list  = res_model_dict['dims_list'][idx],
                    layer_type = res_model_dict['layer_type'], layer_bias = res_model_dict['layer_bias'],
                    norm_type  = res_model_dict['norm_type'],
                    activate_func = res_model_dict['activate_func'], lrelu_slope = res_model_dict['lrelu_slope']
                )
        else:
            self.res_layer = lambda x , batch: x
        self.upd_layer = SequentialBlock(
                dims_list  = upd_model_dict['dims_list'][idx],
                layer_type = upd_model_dict['layer_type'] , layer_bias = upd_model_dict['layer_bias'],
                norm_type  = upd_model_dict['norm_type']  , 
                activate_func = upd_model_dict['activate_func'] , lrelu_slope = upd_model_dict['lrelu_slope']
            )


    def forward(self,x :torch.Tensor,edge_index:torch.Tensor,edge_attr:torch.Tensor,batch:Optional[torch.Tensor]=None) -> torch.Tensor:
        # return self.lin(x) + self.propagate(edge_index,edge_attr=edge_attr,x=x)
        if batch is not None:
            edge_batch = batch[edge_index[0]]
        else:
            edge_batch = None
        return self.propagate(edge_index,edge_attr=edge_attr,x=x,batch=batch,edge_batch=edge_batch)
    
    def message(self, x_i:torch.Tensor, x_j:torch.Tensor,edge_attr:torch.Tensor,edge_batch:torch.Tensor) -> torch.Tensor:
        '''
        x_i : target nodes 
        x_j : source nodes
        '''
        return self.msg_layer(torch.cat([edge_attr,x_j - x_i], dim=1),edge_batch)


    def update(self, msg:torch.Tensor,x:torch.Tensor,batch:torch.Tensor) -> torch.Tensor:
        x = self.res_layer(x)
        # return self.upd_layer(x + msg,batch)
        return self.upd_layer(x + msg,batch)

class EdgeUpdater(nn.Module):

    def __init__(self, 
            idx :str,
            edge_update_model_dict :dict,
        ):
        super(EdgeUpdater, self).__init__()

        assert edge_update_model_dict['edge_mode'] in ['NodeConcat', 'NodeDiff'], 'edge_mode is not in [NodeConcat, NodeDiff]'

        self.edge_mode = edge_update_model_dict['edge_mode']
        self.layers = SequentialBlock(
                edge_update_model_dict['dims_list'][idx],
                edge_update_model_dict['layer_type'], edge_update_model_dict['layer_bias'],
                edge_update_model_dict['norm_type'],
                edge_update_model_dict['activate_func'],edge_update_model_dict['lrelu_slope']
            )
    def forward(self,x:torch.Tensor,edge_index:torch.Tensor,edge_attr:torch.Tensor,batch :Optional[torch.Tensor]=None):
        x_source = x[edge_index[0]]
        x_target = x[edge_index[1]]
        if self.edge_mode == 'NodeConcat':
            res = torch.cat([x_source,x_target,edge_attr],dim=-1)
        elif self.edge_mode == 'NodeDiff':
            res = torch.cat([x_source-x_target,edge_attr],dim=-1)
        return self.layers(res,batch)


class EdgeEncoder(nn.Module):
    ''' graph-in and graph-out Module'''
    def __init__(self, 
            edge_encode_model_dict :dict,
        ):
        super(EdgeEncoder, self).__init__()
        
        self.edge_type    = edge_encode_model_dict['edge_type']
        self.bt_cosine    = edge_encode_model_dict['bt_cosine']
        self.bt_self_loop = edge_encode_model_dict['bt_self_loop']
        self.bt_directed  = edge_encode_model_dict['bt_directed']

        self.encoder  = SequentialBlock(
                edge_encode_model_dict['dims_list'],
                edge_encode_model_dict['layer_type'], edge_encode_model_dict['layer_bias'],
                edge_encode_model_dict['norm_type'],
                edge_encode_model_dict['activate_func'],edge_encode_model_dict['lrelu_slope']
            )
  
    def forward(self,graph:Union[Batch,Data],k:int,batch:Optional[torch.Tensor]=None) -> Union[Batch,Data]:
        
        assert len(graph.x.shape) == 2 , 'Encode node attribute first!'
        
        graph.edge_index = self.construct_edge_index(graph,k,
                        bt_cosine=self.bt_cosine,bt_self_loop= self.bt_self_loop,bt_directed=self.bt_directed)
        raw_edge_attr    = self.compute_edge_attr(graph)
        
        edge_batch = batch[graph.edge_index[-1]] if graph.batch is not None else None
        # edge_batch = graph.edge_index[-1]
        graph.edge_attr = self.encoder(raw_edge_attr,edge_batch)
        return graph 
    
    @staticmethod
    def construct_edge_index(batch: Union[Batch,Data], k, bt_cosine: bool=False,
        bt_self_loop: bool=False,bt_directed: bool=True,bt_input_x:bool = False) -> torch.Tensor:
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
            input =  batch.x if bt_input_x else batch.geometric_info[:,6:8]
            edge_index = knn(input,k, bt_cosine=bt_cosine,bt_self_loop= bt_self_loop,bt_directed=bt_directed)
            return edge_index
        
        # Batch Type
        all_edge_index = []
        for i in range(batch.num_graphs):
            start, end = batch.ptr[i:i+2]
            
            sub_input = batch.x[start:end,:] if bt_input_x else batch.geometric_info[start:end,6:8]
            
            edge_index = knn(sub_input, k, bt_cosine= bt_cosine,bt_self_loop= bt_self_loop,bt_directed= bt_directed)
            
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

        source_info   = batch.geometric_info[source_indice]
        target_info   = batch.geometric_info[target_indice]

        return self._calc_edge_type(source_info,target_info,source_x,target_x)

    def _calc_edge_type(self,source_info,target_info,source_x,target_x):

        # geometric_info = [x,y,x2,y2,w,h,xc,yc,W,H]

        #---------------------------------#
        #  4-dims
        #---------------------------------#
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
        if self.edge_type == 'TgtNorm4-v2':
            feat1 = (source_info[:,6] - target_info[:,6]) /  target_info[:,4]
            feat2 = (source_info[:,7] - target_info[:,7]) /  target_info[:,5]
            feat3 = (source_info[:,4] - (target_info[:,4])) / target_info[:,4]
            feat4 = (source_info[:,5] - (target_info[:,5])) / target_info[:,5]
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
            feat1 = (source_info[:,6] - target_info[:,6]) /  converx_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  converx_bbox_wh[:, 1]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            return torch.stack([feat1,feat2,feat3,feat4],dim =1)
        if self.edge_type == 'MaxNorm4':
            max_bbox_wh = torch.max(source_info[:, 4:6], target_info[:, 4:6])
            feat1 = (source_info[:,6] - target_info[:,6]) /  max_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  max_bbox_wh[:, 1]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            return torch.stack([feat1,feat2,feat3,feat4],dim =1)
        if self.edge_type == 'ConvexNorm4-v2':
            converx_bbox_lt = torch.max(source_info[:, 2:4], target_info[:, 2:4])
            converx_bbox_rb = torch.min(source_info[:, :2], target_info[:, :2])
            converx_bbox_wh = torch.clamp((converx_bbox_lt - converx_bbox_rb), min=0)  # smallest enclosing bbox 
            feat1 = (source_info[:,6] - target_info[:,6]) /  converx_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  converx_bbox_wh[:, 1]
            # feat1 = source_info[:,6] - target_info[:,6] /  converx_bbox_wh[:, 0]
            # feat2 = source_info[:,7] - target_info[:,7] /  converx_bbox_wh[:, 1]
            feat3 = (source_info[:,4] - target_info[:,4]) /  converx_bbox_wh[:, 0]
            feat4 = (source_info[:,5] - target_info[:,5]) /  converx_bbox_wh[:, 1]
            return torch.stack([feat1,feat2,feat3,feat4],dim =1)
        if self.edge_type == 'MaxNorm4-v2':
            max_bbox_wh = torch.max(source_info[:, 4:6], target_info[:, 4:6])
            feat1 = (source_info[:,6] - target_info[:,6]) /  max_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  max_bbox_wh[:, 1]
            feat3 = (source_info[:,4] - target_info[:,4]) /  max_bbox_wh[:, 0]
            feat4 = (source_info[:,5] - target_info[:,5]) /  max_bbox_wh[:, 1]
            return torch.stack([feat1,feat2,feat3,feat4],dim =1)
    
        #---------------------------------#
        #  5-dims
        #---------------------------------#
        
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
        
        if self.edge_type == 'GIOUd5-v2':
            max_bbox_wh = torch.max(source_info[:, 4:6], target_info[:, 4:6])
            feat1 = (source_info[:,6] - target_info[:,6]) /  max_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  max_bbox_wh[:, 1]
            feat3 = (source_info[:,4] - target_info[:,4]) /  max_bbox_wh[:, 0]
            feat4 = (source_info[:,5] - target_info[:,5]) /  max_bbox_wh[:, 1]
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            return torch.stack([feat1,feat2,feat3,feat4,feat5],dim =1)
        if self.edge_type == 'GIOU5-v2':
            max_bbox_wh = torch.max(source_info[:, 4:6], target_info[:, 4:6])
            feat1 = (source_info[:,6] - target_info[:,6]) /  max_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  max_bbox_wh[:, 1]
            feat3 = (source_info[:,4] - target_info[:,4]) /  max_bbox_wh[:, 0]
            feat4 = (source_info[:,5] - target_info[:,5]) /  max_bbox_wh[:, 1]
            feat5 = calc_iouFamily(source_info,target_info,iou_type='giou')
            return torch.stack([feat1,feat2,feat3,feat4,feat5],dim =1)
        
        
 
        if self.edge_type == 'DIOUd5':
            # feat1 = 2 * (target_info[:,6] - source_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            # feat2 = 2 * (target_info[:,7] - source_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            # feat3 = torch.log(target_info[:,4] / source_info[:,4])
            # feat4 = torch.log(target_info[:,5] / source_info[:,5])
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / target_info[:,4])
            feat4 = torch.log(source_info[:,5] / target_info[:,5])
            feat5 = 1- calc_iouFamily(source_info,target_info,iou_type='diou')
            return torch.stack([feat1,feat2,feat3,feat4,feat5],dim =1)
        if self.edge_type == 'DIOU5':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = calc_iouFamily(source_info,target_info,iou_type='diou')
            return torch.stack([feat1,feat2,feat3,feat4,feat5],dim =1)
        
        
        if self.edge_type == 'DIOU5-v2':
            max_bbox_wh = torch.max(source_info[:, 4:6], target_info[:, 4:6])
            feat1 = (source_info[:,6] - target_info[:,6]) /  max_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  max_bbox_wh[:, 1]
            feat3 = (source_info[:,4] - target_info[:,4]) /  max_bbox_wh[:, 0]
            feat4 = (source_info[:,5] - target_info[:,5]) /  max_bbox_wh[:, 1]
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
        if self.edge_type == 'CIOU5-v2':
            max_bbox_wh = torch.max(source_info[:, 4:6], target_info[:, 4:6])
            feat1 = (source_info[:,6] - target_info[:,6]) /  max_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  max_bbox_wh[:, 1]
            feat3 = (source_info[:,4] - target_info[:,4]) /  max_bbox_wh[:, 0]
            feat4 = (source_info[:,5] - target_info[:,5]) /  max_bbox_wh[:, 1]
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


        #---------------------------------#
        #  6 - dims 
        #---------------------------------#

        if self.edge_type == 'Max-GIOUd-Cosd6':
            max_bbox_wh = torch.max(source_info[:, 4:6], target_info[:, 4:6])
            feat1 = (source_info[:,6] - target_info[:,6]) /  max_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  max_bbox_wh[:, 1]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            feat6 = 1 - F.cosine_similarity(source_x,target_x,dim=1)
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)
        if self.edge_type == 'Max-GIOUd-Cos6':
            max_bbox_wh = torch.max(source_info[:, 4:6], target_info[:, 4:6])
            feat1 = (source_info[:,6] - target_info[:,6]) /  max_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  max_bbox_wh[:, 1]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            feat6 = F.cosine_similarity(source_x,target_x,dim=1)
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)
        if self.edge_type == 'Tgt-GIOUd-Cosd6':
            feat1 = (source_info[:,6] - target_info[:,6]) /  target_info[:,4]
            feat2 = (source_info[:,7] - target_info[:,7]) /  target_info[:,5]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            feat6 = 1 - F.cosine_similarity(source_x,target_x,dim=1)
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)
        if self.edge_type == 'Tgt-GIOUd-Cos6':
            feat1 = (source_info[:,6] - target_info[:,6]) /  target_info[:,4]
            feat2 = (source_info[:,7] - target_info[:,7]) /  target_info[:,5]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            feat6 = F.cosine_similarity(source_x,target_x,dim=1)
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)
        if self.edge_type == 'DIOUd-Cosd6':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='diou')
            feat6 = 1 - F.cosine_similarity(source_x,target_x,dim=1)
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)
        if self.edge_type == 'Tgt-DIOUd-Cosd6':
            feat1 = (source_info[:,6] - target_info[:,6]) /  target_info[:,4]
            feat2 = (source_info[:,7] - target_info[:,7]) /  target_info[:,5]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='diou')
            feat6 = 1 - F.cosine_similarity(source_x,target_x,dim=1)
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)
        if self.edge_type == 'GIOUd-Cosd6':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            feat6 = 1 - F.cosine_similarity(source_x,target_x,dim=1)
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)
        if self.edge_type == 'CIOUd-Cosd6':
            feat1 = 2 * (source_info[:,6] - target_info[:,6]) /  (source_info[:,5] + target_info[:,5])
            feat2 = 2 * (source_info[:,7] - target_info[:,7]) /  (source_info[:,5] + target_info[:,5])
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='ciou')
            feat6 = 1 - F.cosine_similarity(source_x,target_x,dim=1)
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)
        if self.edge_type == 'Max-CIOUd-Cosd6':
            max_bbox_wh = torch.max(source_info[:, 4:6], target_info[:, 4:6])
            feat1 = (source_info[:,6] - target_info[:,6]) /  max_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  max_bbox_wh[:, 1]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='ciou')
            feat6 = 1 - F.cosine_similarity(source_x,target_x,dim=1)
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)
        if self.edge_type == 'Max-CIOUd-Cos6':
            max_bbox_wh = torch.max(source_info[:, 4:6], target_info[:, 4:6])
            feat1 = (source_info[:,6] - target_info[:,6]) /  max_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  max_bbox_wh[:, 1]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='ciou')
            feat6 = F.cosine_similarity(source_x,target_x,dim=1)
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)
        if self.edge_type == 'Tgt-CIOUd-Cosd6':
            feat1 = (source_info[:,6] - target_info[:,6]) /  target_info[:,4]
            feat2 = (source_info[:,7] - target_info[:,7]) /  target_info[:,5]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='ciou')
            feat6 = 1 - F.cosine_similarity(source_x,target_x,dim=1)
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)
        if self.edge_type == 'Tgt-CIOUd-Cos6':
            feat1 = (source_info[:,6] - target_info[:,6]) /  target_info[:,4]
            feat2 = (source_info[:,7] - target_info[:,7]) /  target_info[:,5]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='ciou')
            feat6 = F.cosine_similarity(source_x,target_x,dim=1)
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)
        
        if self.edge_type == 'IouFamily6-convex':
            converx_bbox_lt = torch.max(source_info[:, 2:4], target_info[:, 2:4])
            converx_bbox_rb = torch.min(source_info[:, :2], target_info[:, :2])   
            converx_bbox_wh = torch.clamp((converx_bbox_lt - converx_bbox_rb), min=0)   # converx bbox 

            outer_diag = (converx_bbox_wh[:, 0] ** 2) + (converx_bbox_wh[:, 1] ** 2)    # convex diagonal squard length
            inter_diag = (source_info[:, 6] - target_info[:, 6]) ** 2 + (source_info[:, 7] - target_info[:, 7]) ** 2

            feat1 = (source_info[:,6] - target_info[:,6]) /  converx_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  converx_bbox_wh[:, 1]
            feat3 = (source_info[:,4] - target_info[:,4]) /  converx_bbox_wh[:, 0]
            feat4 = (source_info[:,5] - target_info[:,5]) /  converx_bbox_wh[:, 1]
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            feat6 = inter_diag / ( outer_diag + 1e-8 )
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)      
         
        if self.edge_type == 'IouFamily6-max':
            converx_bbox_lt = torch.max(source_info[:, 2:4], target_info[:, 2:4])
            converx_bbox_rb = torch.min(source_info[:, :2], target_info[:, :2])   
            converx_bbox_wh = torch.clamp((converx_bbox_lt - converx_bbox_rb), min=0)   # converx bbox 

            outer_diag = (converx_bbox_wh[:, 0] ** 2) + (converx_bbox_wh[:, 1] ** 2)    # convex diagonal squard length
            inter_diag = (source_info[:, 6] - target_info[:, 6]) ** 2 + (source_info[:, 7] - target_info[:, 7]) ** 2
            max_bbox_wh = torch.max(source_info[:, 4:6], target_info[:, 4:6])
            feat1 = (source_info[:,6] - target_info[:,6]) /  max_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  max_bbox_wh[:, 1]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            feat6 = inter_diag / ( outer_diag + 1e-8 )
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)       
        
        if self.edge_type == 'IouFamily6-enclose':
            converx_bbox_lt = torch.max(source_info[:, 2:4], target_info[:, 2:4])
            converx_bbox_rb = torch.min(source_info[:, :2], target_info[:, :2])   
            converx_bbox_wh = torch.clamp((converx_bbox_lt - converx_bbox_rb), min=0)   # converx bbox 

            outer_diag = (converx_bbox_wh[:, 0] ** 2) + (converx_bbox_wh[:, 1] ** 2)    # convex diagonal squard length
            inter_diag = (source_info[:, 6] - target_info[:, 6]) ** 2 + (source_info[:, 7] - target_info[:, 7]) ** 2

            feat1 = (source_info[:,6] - target_info[:,6]) /  converx_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  converx_bbox_wh[:, 1]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            feat6 = inter_diag / ( outer_diag + 1e-8 )
            return torch.stack([feat1,feat2,feat3,feat4,feat5,feat6],dim =1)       
        
        
        #---------------------------------#
        #  8 -dims
        #---------------------------------#

        if self.edge_type == 'IouFamily8-vanilla':

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
        
        if self.edge_type == 'IouFamily8-convex':

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

            feat1 = (source_info[:,6] - target_info[:,6]) /  converx_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  converx_bbox_wh[:, 1]
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

            feat1 = (source_info[:,6] - target_info[:,6]) /  converx_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  converx_bbox_wh[:, 1]
            feat3 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            feat4 = inter_diag / ( outer_diag + 1e-8 )

            feat5 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat6 = torch.log(source_info[:,5] / (target_info[:,5]))
            feat7 = dis_w / (max_bbox_wh_square[:,0] + 1e-8)
            feat8 = dis_h / (max_bbox_wh_square[:,1] + 1e-8)
            
            return torch.stack([feat1,feat2,feat3,feat4,
                                feat5,feat6,feat7,feat8],dim =1)
        if self.edge_type == 'IouFamily8-vanilla-seq':

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
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))

            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            feat6 = inter_diag / ( outer_diag + 1e-8 )
            feat7 = dis_w / (converx_bbox_w_square + 1e-8)
            feat8 = dis_h / (converx_bbox_h_square + 1e-8)
            
            return torch.stack([feat1,feat2,feat3,feat4,
                                feat5,feat6,feat7,feat8],dim =1)
        
        if self.edge_type == 'IouFamily8-convex-seq':

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

            feat1 = (source_info[:,6] - target_info[:,6]) /  converx_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  converx_bbox_wh[:, 1]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))

            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            feat6 = inter_diag / ( outer_diag + 1e-8 )
            feat7 = dis_w / (converx_bbox_w_square + 1e-8)
            feat8 = dis_h / (converx_bbox_h_square + 1e-8)
            
            return torch.stack([feat1,feat2,feat3,feat4,
                                feat5,feat6,feat7,feat8],dim =1)
        if self.edge_type == 'IouFamily8-separate-seq':

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

            feat1 = (source_info[:,6] - target_info[:,6]) /  converx_bbox_wh[:, 0]
            feat2 = (source_info[:,7] - target_info[:,7]) /  converx_bbox_wh[:, 1]
            feat3 = torch.log(source_info[:,4] / (target_info[:,4]))
            feat4 = torch.log(source_info[:,5] / (target_info[:,5]))

            feat5 = 1 - calc_iouFamily(source_info,target_info,iou_type='giou')
            feat6 = inter_diag / ( outer_diag + 1e-8 )
            feat7 = dis_w / (max_bbox_wh_square[:,0] + 1e-8)
            feat8 = dis_h / (max_bbox_wh_square[:,1] + 1e-8)
            
            return torch.stack([feat1,feat2,feat3,feat4,
                                feat5,feat6,feat7,feat8],dim =1)


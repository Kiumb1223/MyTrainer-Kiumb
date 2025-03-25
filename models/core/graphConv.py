#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Union,Optional
from torch_geometric.data import Batch,Data
from torch_geometric.nn import MessagePassing
from models.core.graphLayers import SequentialBlock,NodeUpdater,EdgeEncoder,EdgeUpdater

__all__ = ['SDgraphConv']

class StaticConv(nn.Module):
    '''
        reference: from torch_geometric.nn import MetaLayer, with a minor modification
    '''
    def __init__(self,
            idx : int,
            static_graph_conv_dict: dict,
        ):
        super(StaticConv,self).__init__()
        node_update_model_dict = static_graph_conv_dict['node_update_model']
        edge_update_model_dict = static_graph_conv_dict['edge_update_model']
        self.node_update_model = NodeUpdater(idx,node_update_model_dict)
        # if edge_update_model_dict['dims_list'][idx] == []:
        if True:
            self.edge_update_model = None
        else:
            self.edge_update_model = EdgeUpdater(idx,edge_update_model_dict)
    def forward(self,
            x :torch.Tensor , edge_index:torch.Tensor,
            edge_attr:torch.Tensor , batch:Optional[torch.Tensor]=None
        ) -> torch.Tensor:

        if self.edge_update_model is not None:
            edge_attr = self.edge_update_model(x,edge_index,edge_attr,batch)
        
        x = self.node_update_model(x,edge_index,edge_attr,batch)

        return x,edge_attr


class DynamicConv(MessagePassing):
    def __init__(self,
            idx :int,
            dynamic_graph_conv_dict :dict
        ):

        super().__init__(aggr=dynamic_graph_conv_dict['aggr'])

        self.bt_cosine     = dynamic_graph_conv_dict['bt_cosine']
        self.bt_self_loop  = dynamic_graph_conv_dict['bt_self_loop']
        self.bt_directed   = dynamic_graph_conv_dict['bt_directed']

        message_model_dict = dynamic_graph_conv_dict['message_model']
        update_model_dict  = dynamic_graph_conv_dict['update_model']

        self.msg_layer = SequentialBlock(
                dims_list  = message_model_dict['dims_list'][idx], 
                layer_type = message_model_dict['layer_type'], layer_bias = message_model_dict['layer_bias'],
                norm_type  = message_model_dict['norm_type'] , 
                activate_func = message_model_dict['activate_func'], lrelu_slope = message_model_dict['lrelu_slope']
            )

        if update_model_dict['dims_list'][idx] != []:
            self.update_layer = SequentialBlock(
                dims_list  = update_model_dict['dims_list'][idx], 
                layer_type = update_model_dict['layer_type'], layer_bias = update_model_dict['layer_bias'],
                norm_type  = update_model_dict['norm_type'] , 
                activate_func = update_model_dict['activate_func'], lrelu_slope = update_model_dict['lrelu_slope']
                )
        else:
            self.update_layer = lambda msg,batch : msg

    def forward(self,x :torch.Tensor,graph :Union[Batch,Data],k:int,batch:Optional[torch.Tensor]=None) -> torch.Tensor:
        
        graph.x = x
        
        edge_index = EdgeEncoder.construct_edge_index(graph,k,bt_cosine=self.bt_cosine,bt_self_loop=self.bt_self_loop,bt_directed=self.bt_directed,bt_input_x=True) 
        if batch is not None:
            edge_batch = batch[edge_index[-1]] # source to target
            # edge_batch = edge_index[-1]
        else:
            edge_batch = None

        x = self.propagate(edge_index,x=x,batch=batch,edge_batch=edge_batch)
        
        return x
    
    def message(self, x_i:torch.Tensor,x_j:torch.Tensor,edge_batch:Optional[torch.Tensor]=None) -> torch.Tensor:
        '''
        x_i : target nodes 
        x_j : source nodes
        '''
        return self.msg_layer(torch.cat([x_i,x_j-x_i],dim=1),edge_batch)        
    def update(self, msg:torch.Tensor,batch:Optional[torch.Tensor]=None) -> torch.Tensor:
        return self.update_layer(msg,batch)    

class SDgraphConv(nn.Module):
    '''
    graph in and node emb out 
    '''
    def __init__(self,
            static_graph_conv_dict :dict, 
            dynamic_graph_conv_dict :dict,
            fuse_model_dict :dict
        ):

        super().__init__()

        #---------------------------------#
        # Instantiate static graph convolutional layer
        #---------------------------------#
        self.sgConv_num = static_graph_conv_dict['layers_num']
        for i in range(self.sgConv_num):
            sgConv = StaticConv(
                i,static_graph_conv_dict
            )
            self.add_module(f'sgConv_{i+1}',sgConv)
        #---------------------------------# 
        # Instantiate dynamic graph convolutional layer
        #---------------------------------#
        self.dgConv_num = dynamic_graph_conv_dict['layers_num']
        for i in range(self.dgConv_num):
            dgConv = DynamicConv(
                i,dynamic_graph_conv_dict
            )
            self.add_module(f'dgConv_{i+1}',dgConv)
        #---------------------------------#
        # Instantiate fusion layer
        # there are two fusion layers
        #---------------------------------#
        self.fuseLayer_1 = SequentialBlock(
                dims_list  = fuse_model_dict['dims_list'][0],
                layer_type = fuse_model_dict['layer_type'], layer_bias = fuse_model_dict['layer_bias'],
                norm_type  = fuse_model_dict['norm_type'], 
                activate_func = fuse_model_dict['activate_func'], lrelu_slope = fuse_model_dict['lrelu_slope']
            )
        self.fuseLayer_2 = SequentialBlock(
                dims_list  = fuse_model_dict['dims_list'][1],
                layer_type = fuse_model_dict['layer_type'], layer_bias = fuse_model_dict['layer_bias'],
                norm_type  = fuse_model_dict['norm_type'], 
                activate_func = fuse_model_dict['activate_func'], lrelu_slope = fuse_model_dict['lrelu_slope'],
                final_activation = False
            )
        
    def forward(self,graph:Union[Batch,Data],k:int) -> torch.Tensor:
        
        assert graph.x is not None         and graph.edge_index is not None \
           and graph.edge_attr is not None and graph.geometric_info is not None
        
        # make a copy 
        if hasattr(graph,'num_graphs'): # Batch Type
            bt_batch = True
            graph_copy = Batch(
                x = None,
                geometric_info=graph.geometric_info.clone(),
                batch=graph.batch.clone(),
                ptr=graph.ptr.clone(),
            )
        else: # Data type 
            bt_batch = False
            graph_copy = Data( 
                x = None,
                geometric_info=graph.geometric_info.clone(),
            )

        x , edge_attr = graph.x,graph.edge_attr
        sgConv_x_list , dgConv_x_list = [] , []
        
        #---------------------------------#
        # Static Graph Convolution Layers
        #---------------------------------#
        for i in range(self.sgConv_num):
            sgConv = getattr(self,f'sgConv_{i+1}')
            x_res ,edge_attr_res= sgConv(x, graph.edge_index,
                        edge_attr, batch = graph.batch if bt_batch else None)
            x , edge_attr = x_res ,edge_attr_res 
            sgConv_x_list.append(x_res)

        #---------------------------------#
        # Dynamic Graph Convolution Layers
        #---------------------------------#
        x = sgConv_x_list[0]
        for i in range(self.dgConv_num):
            dgConv = getattr(self,f'dgConv_{i+1}')
            x_res = dgConv(x,graph_copy,k,batch = graph.batch if bt_batch else None)
            x = x_res
            dgConv_x_list.append(x_res)


        #---------------------------------#
        #  Fusion Module
        #---------------------------------#
        node_emb_cat1 = torch.cat([sgConv_x_list[0],*dgConv_x_list,*sgConv_x_list[1:]],dim=1).unsqueeze(-1)
        node_emb_fuse1 = self.fuseLayer_1(node_emb_cat1,batch = graph.batch if bt_batch else None) 
        node_emb_cat2  = torch.cat([node_emb_fuse1,node_emb_cat1],dim=1)  
        node_emb_output= self.fuseLayer_2(node_emb_cat2,batch = graph.batch if bt_batch else None)
        
        return node_emb_output.squeeze(-1)

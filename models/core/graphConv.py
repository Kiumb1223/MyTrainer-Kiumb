#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Union
from torch.nn import Module
from torch_geometric.data import Batch,Data
from torch_geometric.nn import MessagePassing
from models.core.graphEncoder import SequentialBlock , EdgeEncoder 

__all__ = ['SDgraphConv']

class StaticConv(MessagePassing):
    '''graph in and graph out '''
    def __init__(self,nodeEmb_in : int,dims_list :list,aggr :str, layer_type :str, layer_bias :bool,
                 use_batchnorm :bool,activate_func :str,lrelu_slope :float=0.0):

        super().__init__(aggr=aggr) 
        

        msg_in , *msg_out , update_out = dims_list
        
        self.msg_func    = SequentialBlock(
            in_dim=msg_in, out_dim=None, hidden_dim= msg_out,
            layer_type =layer_type, layer_bias=layer_bias,
            use_batchnorm=use_batchnorm, 
            activate_func=activate_func, lrelu_slope=lrelu_slope
        )

        self.res_node_func = SequentialBlock(
            in_dim=nodeEmb_in, out_dim=None, hidden_dim= msg_out,
            layer_type =layer_type, layer_bias=layer_bias,
            use_batchnorm=use_batchnorm, 
            activate_func=activate_func, lrelu_slope=lrelu_slope
        )

        self.update_func = SequentialBlock(
            in_dim=msg_out[-1], out_dim=update_out, hidden_dim= [],
            layer_type =layer_type,layer_bias=layer_bias,
            use_batchnorm=use_batchnorm, 
            activate_func=activate_func, lrelu_slope=lrelu_slope
        )


    def forward(self,node_emb :torch.Tensor,edge_index:torch.Tensor,edge_attr:torch.Tensor) -> torch.Tensor:
        # return self.lin(x) + self.propagate(edge_index,edge_attr=edge_attr,x=x)
        return self.propagate(edge_index,edge_attr=edge_attr,x=node_emb)
    
    def message(self, x_i:torch.Tensor, x_j:torch.Tensor,edge_attr:torch.Tensor) -> torch.Tensor:
        '''
        x_i : target nodes 
        x_j : source nodes
        '''
        return self.msg_func(torch.cat([edge_attr,x_j - x_i], dim=1))
    
    def update(self, msg:torch.Tensor,x:torch.Tensor) -> torch.Tensor:
        return self.update_func(self.res_node_func(x) + msg)
    

class DynamicGonv(MessagePassing):
    def __init__(self,edge_model_dict :dict,
                dims_list :list,aggr :str, layer_type :str,layer_bias:bool,
                use_batchnorm :bool,activate_func :str,lrelu_slope :float=0.0,
                bt_cosine :bool=False, bt_self_loop :bool=True,bt_directed :bool=True
                ):

        super().__init__(aggr=aggr)

        self.bt_cosine = bt_cosine
        self.bt_self_loop = bt_self_loop
        self.bt_directed = bt_directed
        
        msg_in , *msg_mid , msg_out = dims_list


        self.dg_edgeEncoder = EdgeEncoder(edge_model_dict)

        self.msg_func = SequentialBlock(
            in_dim=msg_in, out_dim=msg_out, hidden_dim= msg_mid, 
            layer_type = layer_type, layer_bias=layer_bias,
            use_batchnorm=use_batchnorm, 
            activate_func=activate_func, lrelu_slope=lrelu_slope
        )

    def forward(self,node_emb :torch.Tensor,graph :Union[Batch,Data],k:int) -> torch.Tensor:
        graph.x = node_emb
        # edge_index = EdgeEncoder.construct_edge_index(graph,k,bt_cosine=self.bt_cosine,bt_self_loop=self.bt_self_loop,bt_directed=self.bt_directed) 
        # edge_attr  = EdgeEncoder.compute_edge_attr(graph,k) 
        graph = self.dg_edgeEncoder(graph,k)

        node_emb = self.propagate(graph.edge_index,edge_attr=graph.edge_attr,x=node_emb)
        return node_emb
    
    def message(self, x_i:torch.Tensor,x_j:torch.Tensor,edge_attr:torch.Tensor) -> torch.Tensor:
        '''
        x_i : target nodes 
        x_j : source nodes
        '''
        return self.msg_func(torch.cat([x_i,edge_attr,x_j-x_i],dim=1))
    

class SDgraphConv(Module):
    def __init__(self,edge_Encoder_dict:dict,
                static_graph_model_dict :dict, dynamic_graph_model_dict :dict,fuse_model_dict :dict):

        super().__init__()

        self.sg1conv  = StaticConv(static_graph_model_dict['inputNodeSize'][0],static_graph_model_dict['layer'][0],
                        static_graph_model_dict['aggr'],static_graph_model_dict['layer_type'],static_graph_model_dict['layer_bias'],
                        static_graph_model_dict['use_batchnorm'],static_graph_model_dict['activate_func'],static_graph_model_dict['lrelu_slope'],)
        self.sg2conv  = StaticConv(static_graph_model_dict['inputNodeSize'][1],static_graph_model_dict['layer'][1],
                        static_graph_model_dict['aggr'],static_graph_model_dict['layer_type'],static_graph_model_dict['layer_bias'],
                        static_graph_model_dict['use_batchnorm'],static_graph_model_dict['activate_func'],static_graph_model_dict['lrelu_slope'])
        self.sg3conv  = StaticConv(static_graph_model_dict['inputNodeSize'][2],static_graph_model_dict['layer'][2],
                        static_graph_model_dict['aggr'],static_graph_model_dict['layer_type'],static_graph_model_dict['layer_bias'],
                        static_graph_model_dict['use_batchnorm'],static_graph_model_dict['activate_func'],static_graph_model_dict['lrelu_slope'])

        self.dg1conv  = DynamicGonv(edge_Encoder_dict,
                        dynamic_graph_model_dict['layer'][0],dynamic_graph_model_dict['aggr'],dynamic_graph_model_dict['layer_type'],dynamic_graph_model_dict['layer_bias'],
                        dynamic_graph_model_dict['use_batchnorm'],dynamic_graph_model_dict['activate_func'],dynamic_graph_model_dict['lrelu_slope'],
                        dynamic_graph_model_dict['bt_cosine'],dynamic_graph_model_dict['bt_self_loop'],dynamic_graph_model_dict['bt_directed'])
        
        self.dg2conv  = DynamicGonv(edge_Encoder_dict,
                        dynamic_graph_model_dict['layer'][1],dynamic_graph_model_dict['aggr'],dynamic_graph_model_dict['layer_type'],dynamic_graph_model_dict['layer_bias'],
                        dynamic_graph_model_dict['use_batchnorm'],dynamic_graph_model_dict['activate_func'],dynamic_graph_model_dict['lrelu_slope'],
                        dynamic_graph_model_dict['bt_cosine'],dynamic_graph_model_dict['bt_self_loop'],dynamic_graph_model_dict['bt_directed'])
        
        f1_in , *f1_mid, f1_out = fuse_model_dict['fuse1_dims']
        f2_in , *f2_mid, f2_out = fuse_model_dict['fuse2_dims']

        self.fuse1conv = SequentialBlock(f1_in,f1_out,f1_mid,fuse_model_dict['layer_type'],fuse_model_dict['layer_bias'],
                        fuse_model_dict['use_batchnorm'],fuse_model_dict['activate_func'],fuse_model_dict['lrelu_slope'])
        self.fuse2conv = SequentialBlock(f2_in,f2_out,f2_mid,fuse_model_dict['layer_type'],fuse_model_dict['layer_bias'],
                        fuse_model_dict['use_batchnorm'],fuse_model_dict['activate_func'],fuse_model_dict['lrelu_slope'],final_activation=False)


    def forward(self,graph:Union[Batch,Data],k:int) -> torch.Tensor:
        ''' graph in and node_embedding out '''
        assert graph.x is not None         and graph.edge_index is not None \
           and graph.edge_attr is not None and graph.location_info is not None
        
        # make a copy 
        if hasattr(graph,'num_graphs'): # Batch Type
            graph_copy = Batch(
                x = None,
                location_info=graph.location_info.clone(),
                batch=graph.batch.clone(),
                ptr=graph.ptr.clone(),
            )
        else: # Data type 
            graph_copy = Data( 
                x = None,
                location_info=graph.location_info.clone(),
            )
            
        node_embedding_sg1 = self.sg1conv(graph.x,graph.edge_index,graph.edge_attr)      
        node_embedding_sg2 = self.sg2conv(node_embedding_sg1,graph.edge_index,graph.edge_attr)  
        node_embedding_sg3 = self.sg3conv(node_embedding_sg2,graph.edge_index,graph.edge_attr)  


        node_embedding_dg1 = self.dg1conv(node_embedding_sg1,graph_copy,k)  
        node_embedding_dg2 = self.dg2conv(node_embedding_dg1,graph_copy,k)  

        node_embedding_cat1  = torch.cat([node_embedding_sg1,node_embedding_dg1,node_embedding_dg2,
                                          node_embedding_sg2,node_embedding_sg3],dim=1).unsqueeze(-1)   
        node_embedding_fuse1 = self.fuse1conv(node_embedding_cat1) 

        node_embedding_cat2  = torch.cat([node_embedding_fuse1,node_embedding_cat1],dim=1)  
        node_embedding_output= self.fuse2conv(node_embedding_cat2)
        
        return node_embedding_output.squeeze(-1)

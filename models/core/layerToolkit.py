#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     layerToolkit.py
@Time     :     2025/01/03 14:22:44
@Author   :     Louis Swift
@Desc     :     
'''


import torch.nn as nn
from typing import Union

__all__ = ['parse_layer_dimension','SequentialBlock']

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

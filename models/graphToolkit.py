#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
from torch import Tensor
from loguru import logger
import scipy.optimize as opt
import torch.nn.functional as F
from typing import Optional,Tuple

__all__ = ['knn','hungarian','sinkhorn_unrolled','Sinkhorn']


def knn(x: torch.tensor, k: int, bt_cosine: Optional[bool]=False,bt_self_loop: Optional[bool]=False,
        bt_edge_index:Optional[bool]=False, flow:Optional[str]='source_to_target') -> torch.Tensor:
    """
    Calculate K nearest neighbors, supporting Euclidean distance and cosine distance.
    
    Args:
        x (Tensor): Input point set, shape of (n, d), each row represents a d-dimensional feature vector.
        k (int): Number of neighbors.
        bt_cosine (bool): Whether to use cosine distance.
        bt_self_loop (bool): Whether to include self-loop (i.e., whether to consider itself as its own neighbor).
        bt_edge_index (bool): Whether to return the indices of neighbors, or the edge index of the graph.
        flow (str): the direction of the edge index ( reference pytorch-geometric docs) 
        
    Returns:
        Tensor: Indices of neighbors, shape of (2, n * k).
    """
    
    assert flow in ['source_to_target', 'target_to_source']
    num_node = x.shape[0]

    if num_node <= k :
        # raise ValueError("The number of points is less than k, please set k smaller than the number of points.")
        logger.warning(f"SPECIAL SITUATIONS: The number of points is less than k, set k to {x.shape[0] -1}")
        k = num_node - 1
    
    if bt_cosine:   # cosine distance
        x_normalized = F.normalize(x, p=2, dim=1)
        cosine_similarity_matrix = torch.mm(x_normalized, x_normalized.T)
        dist_matrix = 1 - cosine_similarity_matrix  
    else:           # Euclidean distance
        assert len(x.shape) == 2  
        dist_matrix = torch.cdist(x, x) 
        
    if not bt_self_loop:
        dist_matrix.fill_diagonal_(float('inf'))  # Remove self loop by setting diagonal to infinity
    else:
        k += 1 # Include self-loop
    
    _, indices1 = torch.topk(dist_matrix, k, largest=False, dim=1)
    
    if not bt_edge_index:
        return indices1 , k
    
    indices2 = torch.arange(0, num_node, device=x.device).repeat_interleave(k)
    if flow == 'source_to_target':
        return torch.stack([indices1.flatten(), indices2], dim=0)
    else:
        return torch.stack([indices2, indices1.flatten()], dim=0)

def hungarian(affinity_mtx: torch.Tensor,match_thresh: float=0.1  ):
    r"""
    Solve optimal LAP permutation by hungarian algorithm. The time cost is :math:`O(n^3)`.

    :param affinity_mtx: size - :math:`( n_tra \times n_det)`
    :param match_thresh: threshold for valid match

    :return  match_mtx: size - :math:`( n_tra \times n_det)`, match matrix
    :return  match_idx: size - :math:`( 2 \times n_match)`, match index
    :return  unmatch_tra: size - :math:`( n_unmatch_tra)`, unmatched trajectory index
    :return  unmatch_det: size - :math:`( n_unmatch_det)`, unmatched detection index
    """
    if affinity_mtx[0] is None: # frame_idx == 1 
        num_det = affinity_mtx[1]
        return np.array([]),np.array([]),np.array([]),np.arange(num_det)
    
    affinity_mtx = affinity_mtx[:-1,:-1].cpu().numpy()  # remove last row and column

    num_rows , num_cols = affinity_mtx.shape

    all_rows = np.arange(num_rows)
    all_cols = np.arange(num_cols)
    hungarian_mtx = np.zeros_like(affinity_mtx)

    cost_mtx  = affinity_mtx * -1
    row, col = opt.linear_sum_assignment(cost_mtx)
    
    hungarian_mtx[row, col] = 1
    valid_mask = (
        (hungarian_mtx == 1) &
        (affinity_mtx >= match_thresh)
    )
    
    match_mtx   = np.where(valid_mask,hungarian_mtx,0)
    valid_row,valid_col = np.where(valid_mask)

    match_idx   = np.vstack([valid_row,valid_col])
    unmatch_tra = np.setdiff1d(all_rows, valid_row,assume_unique=True)
    unmatch_det = np.setdiff1d(all_cols, valid_col,assume_unique=True)

    return match_mtx,match_idx,unmatch_tra,unmatch_det


'''
Extracted from https://github.com/marvin-eisenberger/implicit-sinkhorn, with minor modifications
And much thanks to their brilliant work~ :)
'''

def sinkhorn_unrolled(c, a, b, num_sink, lambd_sink):
    """
    An implementation of a Sinkhorn layer with Automatic Differentiation (AD).
    The format of input parameters and outputs is equivalent to the 'Sinkhorn' module below.
    """
    log_p = -c / lambd_sink
    log_a = torch.log(a).unsqueeze(dim=-1)
    log_b = torch.log(b).unsqueeze(dim=-2)
    for _ in range(num_sink):
        log_p = log_p - (torch.logsumexp(log_p, dim=-2, keepdim=True) - log_b)
        log_p = log_p - (torch.logsumexp(log_p, dim=-1, keepdim=True) - log_a)
    p = torch.exp(log_p)
    return p


class Sinkhorn(torch.autograd.Function):
    """
    An implementation of a Sinkhorn layer with our custom backward module, based on implicit differentiation
    :param c: input cost matrix, size [*,m,n], where * are arbitrarily many batch dimensions
    :param a: first input marginal, size [*,m]
    :param b: second input marginal, size [*,n]
    :param num_sink: number of Sinkhorn iterations
    :param lambd_sink: entropy regularization weight
    :return: optimized soft permutation matrix
    """

    @staticmethod
    def forward(ctx, c, a, b, num_sink, lambd_sink):
        log_p = -c / lambd_sink
        log_a = torch.log(a).unsqueeze(dim=-1)
        log_b = torch.log(b).unsqueeze(dim=-2)
        for _ in range(num_sink):
            log_p -= (torch.logsumexp(log_p, dim=-2, keepdim=True) - log_b)
            log_p -= (torch.logsumexp(log_p, dim=-1, keepdim=True) - log_a)
        p = torch.exp(log_p)

        ctx.save_for_backward(p, torch.sum(p, dim=-1), torch.sum(p, dim=-2))
        ctx.lambd_sink = lambd_sink
        return p

    @staticmethod
    def backward(ctx, grad_p):
        p, a, b = ctx.saved_tensors

        m, n = p.shape[-2:]
        batch_shape = list(p.shape[:-2])

        grad_p *= -1 / ctx.lambd_sink * p
        K = torch.cat((torch.cat((torch.diag_embed(a), p), dim=-1),
                       torch.cat((p.transpose(-2, -1), torch.diag_embed(b)), dim=-1)), dim=-2)[..., :-1, :-1]
        t = torch.cat((grad_p.sum(dim=-1), grad_p[..., :, :-1].sum(dim=-2)), dim=-1).unsqueeze(-1)
        grad_ab, _ = torch.solve(t, K)
        grad_a = grad_ab[..., :m, :]
        grad_b = torch.cat((grad_ab[..., m:, :], torch.zeros(batch_shape + [1, 1], device=grad_p.device, dtype=torch.float32)), dim=-2)
        U = grad_a + grad_b.transpose(-2, -1)
        grad_p -= p * U
        grad_a = -ctx.lambd_sink * grad_a.squeeze(dim=-1)
        grad_b = -ctx.lambd_sink * grad_b.squeeze(dim=-1)
        return grad_p, grad_a, grad_b, None, None, None


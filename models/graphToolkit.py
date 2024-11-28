#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
from torch import Tensor
from loguru import logger
from typing import Optional
import scipy.optimize as opt
import torch.nn.functional as F
from multiprocessing import Pool

__all__ = ['knn','sinkhorn_unrolled','Sinkhorn','hungarian']


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


'''
Extracted from https://github.com/Thinklab-SJTU/ThinkMatch/blob/master/src/lap_solvers/hungarian.py
And much thanks to their brilliant work :)
'''
def hungarian(s: Tensor, n1: Tensor=None, n2: Tensor=None, nproc: int=1) -> Tensor:
    r"""
    Solve optimal LAP permutation by hungarian algorithm. The time cost is :math:`O(n^3)`.

    :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
    :param n1: :math:`(b)` number of objects in dim1
    :param n2: :math:`(b)` number of objects in dim2
    :param nproc: number of parallel processes (default: ``nproc=1`` for no parallel)
    :return: :math:`(b\times n_1 \times n_2)` optimal permutation matrix

    .. note::
        We support batched instances with different number of nodes, therefore ``n1`` and ``n2`` are
        required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
        the batched matrices are not padded.
    """
    if len(s.shape) == 2:
        s = s.unsqueeze(0)
        matrix_input = True
    elif len(s.shape) == 3:
        matrix_input = False
    else:
        raise ValueError('input data shape not understood: {}'.format(s.shape))

    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy() * -1
    if n1 is not None:
        n1 = n1.cpu().numpy()
    else:
        n1 = [None] * batch_num
    if n2 is not None:
        n2 = n2.cpu().numpy()
    else:
        n2 = [None] * batch_num

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(_hung_kernel, zip(perm_mat, n1, n2))
            perm_mat = np.stack(mapresult.get())
    else:
        perm_mat = np.stack([_hung_kernel(perm_mat[b], n1[b], n2[b]) for b in range(batch_num)])

    perm_mat = torch.from_numpy(perm_mat).to(device)

    if matrix_input:
        perm_mat.squeeze_(0)

    return perm_mat

def _hung_kernel(s: torch.Tensor, n1=None, n2=None):
    if n1 is None:
        n1 = s.shape[0]
    if n2 is None:
        n2 = s.shape[1]
    row, col = opt.linear_sum_assignment(s[:n1, :n2])
    perm_mat = np.zeros_like(s)
    perm_mat[row, col] = 1
    return perm_mat
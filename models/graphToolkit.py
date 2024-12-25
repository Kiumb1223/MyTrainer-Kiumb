#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
from typing import Tuple
from loguru import logger
import scipy.optimize as opt
import torch.nn.functional as F

__all__ = ['knn','hungarian','box_iou','box_ciou','sinkhorn_unrolled','Sinkhorn','compute_f1_score']


def knn(x: torch.tensor, k: int, bt_cosine: bool=False,
        bt_self_loop: bool=False,bt_directed: bool=True) -> torch.Tensor:
    """
    Calculate K nearest neighbors, supporting Euclidean distance and cosine distance.
    
    Args:
        x (Tensor): Input point set, shape of (n, d), each row represents a d-dimensional feature vector.
        k (int): Number of neighbors.
        bt_cosine (bool): Whether to use cosine distance.
        bt_self_loop (bool): Whether to include self-loop (i.e., whether to consider itself as its own neighbor).
        bt_directed (bool): return the directed graph or the undirected one. 

    Returns:
        edge_index (tensor): the edge index of the graph, shape of (2, n * k).
    """
    
    num_node = x.shape[0]

    if num_node <= k :
        # raise ValueError("The number of points is less than k, please set k smaller than the number of points.")
        logger.warning(f"SPECIAL SITUATIONS: The number of points is less than k, set k to {x.shape[0] -1}")
        k = num_node - 1
    
    if k > 0:
        if bt_cosine:   # cosine distance
            x_normalized = F.normalize(x, p=2, dim=1)
            cosine_similarity_matrix = torch.mm(x_normalized, x_normalized.T)
            dist_matrix  = 1 - cosine_similarity_matrix  
        else:           # Euclidean distance
            assert len(x.shape) == 2  
            dist_matrix = torch.cdist(x, x) 
            
        dist_matrix.fill_diagonal_(float('inf'))  
    
        _, indices1 = torch.topk(dist_matrix, k, largest=False, dim=1)
        indices2 = torch.arange(0, num_node, device=x.device).repeat_interleave(k)
    else:
        indices1 = torch.tensor([],device=x.device)
        indices2 = torch.tensor([],device=x.device)
    
    if bt_self_loop:
        indices_self = torch.arange(0,num_node,device=x.device)
        if bt_directed:
            return torch.stack([  # flow: from source node to target node 
                torch.cat([indices1.flatten(),indices_self],dim=-1),
                torch.cat([indices2,indices_self],dim=-1),
            ]).to(x.device).to(torch.long)
        else:
            return torch.stack([  # flow: from source node to target node 
                torch.cat([indices1.flatten(),indices_self,indices2],dim=-1),
                torch.cat([indices2,indices_self,indices1.flatten()],dim=-1),
            ]).to(x.device).to(torch.long)
    else:
        if bt_directed:
            return torch.stack([indices1.flatten(),indices2]).to(x.device).to(torch.long)  # flow: from source node to target node 
        else:
            return torch.stack([  # flow: from source node to target node 
                torch.cat([indices1.flatten(),indices2],dim=-1),
                torch.cat([indices2,indices1.flatten()],dim=-1),
            ]).to(x.device).to(torch.long)


def hungarian(affinity_mtx: np.ndarray,match_thresh: float=0.1,is_iou_match:bool=False):
    r"""
    Solve optimal LAP permutation by hungarian algorithm. The time cost is :math:`O(n^3)`.

    :param affinity_mtx: size - :math:`( n_tra \times n_det )`
    :param match_thresh: threshold for valid match
    :param is_iou_match: flag to switch between affinity and IOU-based matching

    :return  match_mtx: size - :math:`( n_tra \times n_det )`, match matrix
    :return  match_idx: size - :math:`( 2 \times n_match )`, match index
    :return  unmatch_tra: size - :math:`( n_unmatch_tra )`, unmatched trajectory index
    :return  unmatch_det: size - :math:`( n_unmatch_det )`, unmatched detection index
    """
    if affinity_mtx.size == 0 : # frame_idx == 1 
        return np.array([]),[],list(range(affinity_mtx.shape[0])),list(range(affinity_mtx.shape[1]))
    
    num_rows , num_cols = affinity_mtx.shape

    all_rows = np.arange(num_rows)
    all_cols = np.arange(num_cols)
    hungarian_mtx = np.zeros_like(affinity_mtx)

    cost_mtx = 1 - affinity_mtx
    
    row, col = opt.linear_sum_assignment(cost_mtx)
    
    hungarian_mtx[row, col] = 1
    valid_mask = (
        (hungarian_mtx == 1) &
        (affinity_mtx >= match_thresh)
    )
    
    match_mtx   = np.where(valid_mask,hungarian_mtx,0)
    valid_row,valid_col = np.where(valid_mask)

    match_idx   = np.vstack([valid_row,valid_col]).tolist()
    unmatch_tra = np.setdiff1d(all_rows, valid_row,assume_unique=True).tolist()
    unmatch_det = np.setdiff1d(all_cols, valid_col,assume_unique=True).tolist()

    return match_mtx,match_idx,unmatch_tra,unmatch_det



def box_iou(boxes1:np.ndarray, boxes2:np.ndarray) -> np.ndarray:
    ''' Return intersection-over-union (Jaccard index) of boxes.
     
    Args:
        boxes1 (np.ndarray): shape (n1, 4)  || (min x, min y, max x, max y)  
        boxes2 (np.ndarray): shape (n2, 4)  || (min x, min y, max x, max y)
    Returns:
        iou (np.ndarray): shape (n1, n2)
    '''
    iou = np.zeros((len(boxes1),len(boxes2)),dtype=np.float32)
    if iou.size == 0 :
        return iou
    # cal the box's area of boxes1 and boxess
    boxes1Area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # cal Intersection
    lt = np.maximum(boxes1[:,None, :2], boxes2[..., :2])
    rb = np.minimum(boxes1[:,None, 2:], boxes2[..., 2:])
    inter = np.maximum(rb - lt, 0)
    inter_area = inter[:,:, 0] * inter[:,:, 1]

    # cal Union
    union_area = boxes1Area[:,None] + boxes2Area - inter_area

    # cal IoU
    iou = inter_area / union_area

    return iou



def box_ciou(boxes1:torch.Tensor, boxes2:torch.Tensor, GIOU=False, CIOU=True, DIOU=False) -> torch.Tensor:
    """
    Return complete IOU between two sets of boxes

    Args:
        b1 Tensor[M, 4]: first set of boxes
        b2 Tensor[N, 4]: second set of boxes

    Returns:
        Tensor[M, N]:
    """
    # cal the box's area of boxes1 and boxess
    boxes1Area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = torch.clamp(rb - lt, min=0)  # [N,M,2]
    inter_area = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union_area = boxes1Area[:, None] + boxes2Area - inter_area
    iou = inter_area / union_area

    w1, h1 = boxes1[:, 2] - boxes1[:, 0], boxes1[:, 3] - boxes1[:, 1]
    w2, h2 = boxes2[:, 2] - boxes2[:, 0], boxes2[:, 3] - boxes2[:, 1]
    center1 = (boxes1[:, 2:] + boxes1[:, :1]) / 2
    center2 = (boxes2[:, 2:] + boxes2[:, :1]) / 2
    out_max_xy = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    out_min_xy = torch.min(boxes1[:, None, :2], boxes2[:, :2])# [N, M, 2]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    if GIOU:
        c_area = outer[:, :, 0] * outer[:, :, 1] + 1e-16
        return iou - (c_area - union_area) / c_area
    if DIOU or CIOU:
        outer_diag = (outer[:, :, 0] ** 2) + (outer[:, :, 1] ** 2)
        inter_diag = (center1[:, None, 0] - center2[:, 0]) ** 2 + (center1[:, None, 1] - center2[:, 1]) ** 2
        if DIOU:
            return iou - inter_diag / outer_diag
        elif CIOU:
            arctan = torch.atan(w2 / h2)[:] - torch.atan(w1 / h1)[:, None]
            v = (4 / (torch.pi ** 2)) * torch.pow(arctan, 2)
            S = 1 - iou
            with torch.no_grad():
                alpha = v / (S + v)
                u = inter_diag / outer_diag
                ciou = iou - (u + v * alpha)
                ciou = torch.clamp(ciou, min=-1.0, max=1.0)
            return ciou


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



def compute_f1_score(pred_mtx:np.ndarray,gt_mtx:np.ndarray) -> int:
    '''for evaluation phase'''
    TP = np.sum(np.logical_and(pred_mtx == 1, gt_mtx == 1))
    FP = np.sum(np.logical_and(pred_mtx == 1, gt_mtx == 0))
    FN = np.sum(np.logical_and(pred_mtx == 0, gt_mtx == 1))


    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score
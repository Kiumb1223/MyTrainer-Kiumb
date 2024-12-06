#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.graphToolkit import hungarian

__all__ = ['GraphLoss', 'PermutationLossHung', 'CombinedMatchingLoss']


class GraphLoss(nn.Module):
    """
    Improved graph matching loss with better numerical stability and masking
    """

    def __init__(self, use_focal_loss: bool = True, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        # 添加Focal Loss以更好地处理类别不平衡问题
        """
        # eps = 1e-6  # numerical stability
        # pred = torch.clamp(pred, eps, 1 - eps)

        ce_loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** self.gamma

        loss = self.alpha * focal_weight * ce_loss

        return loss.sum()

    def forward(self, pred_mtx_list: list, gt_mtx_list: list) -> Tensor:
        num_graphs = len(pred_mtx_list)
        assert num_graphs == len(gt_mtx_list)

        loss = torch.zeros(1).to(pred_mtx_list[0])

        for i in range(num_graphs):

            gt_mtx = gt_mtx_list[i]
            pred_mtx = pred_mtx_list[i][:-1,:-1] 

            loss += self.focal_loss(pred_mtx, gt_mtx)

        return loss / num_graphs


class PermutationLossHung(nn.Module):
    """
    改进的带Hungarian注意力的排列损失
    """

    def __init__(self, lambda_reg: float = 0.1):
        super().__init__()
        # 添加正则化参数
        self.lambda_reg = lambda_reg

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        batch_num = pred_dsmat.shape[0]

        # 添加数值稳定性检查
        assert torch.all((pred_dsmat >= -1e-6) * (pred_dsmat <= 1 + 1e-6)), "Pred matrix out of valid range"
        assert torch.all((gt_perm >= -1e-6) * (gt_perm <= 1 + 1e-6)), "GT matrix out of valid range"

        # 确保预测矩阵在有效范围内
        pred_dsmat = torch.clamp(pred_dsmat, 0, 1)

        # 计算Hungarian匹配
        dis_pred = hungarian(pred_dsmat, src_ns, tgt_ns)

        # 合并预测和真实标签的注意力
        ali_perm = dis_pred + gt_perm
        ali_perm = torch.clamp(ali_perm, 0, 1)

        # 应用注意力机制
        pred_dsmat = torch.mul(ali_perm, pred_dsmat)
        gt_perm = torch.mul(ali_perm, gt_perm)

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)

        for b in range(batch_num):
            # 提取有效区域
            valid_pred = pred_dsmat[b, :src_ns[b], :tgt_ns[b]]
            valid_gt = gt_perm[b, :src_ns[b], :tgt_ns[b]]

            # 计算主损失
            main_loss = F.binary_cross_entropy(valid_pred, valid_gt, reduction='sum')

            # 添加正则化项
            reg_loss = self.lambda_reg * torch.norm(valid_pred - valid_gt, p='fro')

            loss += main_loss + reg_loss
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / (n_sum + 1e-6)


class CombinedMatchingLoss(nn.Module):
    """
    # 新增组合损失函数，结合多个损失项
    """

    def __init__(self,
                 lambda_graph: float = 1.0,
                 lambda_hung: float = 1.0,
                 use_focal: bool = True):
        super().__init__()
        self.graph_loss = GraphLoss(use_focal_loss=use_focal)
        self.hung_loss = PermutationLossHung()
        self.lambda_graph = lambda_graph
        self.lambda_hung = lambda_hung

    def forward(self, pred_mtx_list: list, gt_mtx_list: list,
                pred_dsmat: Tensor = None, gt_perm: Tensor = None,
                src_ns: Tensor = None, tgt_ns: Tensor = None) -> Tensor:
        """
        组合损失计算
        """
        # 计算基础图匹配损失
        loss = self.lambda_graph * self.graph_loss(pred_mtx_list, gt_mtx_list)

        # 如果提供了置换矩阵相关参数，计算Hungarian损失
        if pred_dsmat is not None and gt_perm is not None:
            loss += self.lambda_hung * self.hung_loss(pred_dsmat, gt_perm, src_ns, tgt_ns)

        return loss
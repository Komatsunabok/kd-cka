from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cka.LinearCKA import linear_CKA

class CKADistillLoss(nn.Module):
    """
    CKAベースの蒸留損失関数
    ours
    """
    def __init__(self, group_num=4, method='mean', reduction='sum'):
        super().__init__()
        self.group_num = group_num
        self.method = method
        self.reduction = reduction

    def forward(self, s_group_feats, t_group_feats):
        # 各グループごとにCKA平均を計算し、(1-平均CKA)を損失として合計
        total_loss = 0.0
        for s_feats, t_feats in zip(s_group_feats, t_group_feats):
            cka_vals = []
            for s, t in zip(s_feats, t_feats):
                cka = linear_CKA(s, t)  # 外部CKA関数（要import or globalで用意）
                cka_vals.append(cka)
            mean_cka = torch.mean(torch.stack(cka_vals))
            total_loss += (1.0 - mean_cka)
        if self.reduction == 'mean':
            total_loss = total_loss / self.group_num
        return total_loss


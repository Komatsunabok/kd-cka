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
    ckadと名付けた、とりあえず
    """
    def __init__(self, group_num=4, method_inner_group='mean', method_inter_group='mean'):
        super().__init__()
        self.group_num = group_num
        self.method_inner_group = method_inner_group # グループ内のCKA計算方法
        self.method_inter_group = method_inter_group # グループ間のCKA計算方法

    def forward(self, s_group_feats, t_group_feats):
        total_loss = 0.0
        
        # グループ内のCKAを計算
        cka_vals_inter = []
        for s_feats, t_feats in zip(s_group_feats, t_group_feats): # 同じグループの特徴量をペアで取得
            # あるグループについて
            if self.method_inner_group == 'mean':
                cka_vals_inner = []
                # 各グループの特徴量のCKAを計算
                for i, s in enumerate(s_feats):
                    for j, t in enumerate(t_feats):
                        cka = linear_CKA(s, t)
                        loss = 1 - cka  # CKAは0から1の範囲なので、1から引くことで損失に変換
                        cka_vals_inner.append(loss)
                # グループ内のCKAを平均
                mean_loss = torch.mean(torch.stack(cka_vals_inner))
                cka_vals_inter.append(mean_loss)

        # グループ間のCKAを計算
        if self.method_inter_group == 'mean':
            total_loss = torch.mean(torch.stack(cka_vals_inter))
        return total_loss


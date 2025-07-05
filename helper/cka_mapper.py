import torch
import torch.nn as nn
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cka.LinearCKA import linear_CKA

class CKAMapper(nn.Module):
    """
    教師・生徒の特徴マップをグループ分けし、対応付けを管理するモジュール
    """
    def __init__(self, s_shapes, t_shapes, feat_t, group_num=4):
        super().__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes
        self.group_num = group_num

        # 与えられたデータすべてを使用
        # =ここに与えられるデータは計算されるべきものだけにする

        # 教師特徴マップとCKA計算関数が与えられた場合のみ、CKAベースのグループ分け
        # 層インデックスをグループ数で分割したリスト
        # 例: s_shapes の長さが8（生徒の対象層が8個）、group_num=4 の場合
        # 　  self.s_groups == [[0, 1], [2, 3], [4, 5], [6, 7]]
        self.t_groups = self._split_groups_by_cka(feat_t, group_num)
        self.s_groups = self._split_groups(len(s_shapes), group_num)
    
    def _split_groups_by_cka(self, feat_t, group_num):
        # 1. CKA行列を計算（隣接層のみ）
        n = len(feat_t)
        cka_mat = np.zeros((n-1,))
        for i in range(n-1):
            cka_mat[i] = linear_CKA(feat_t[i], feat_t[i+1]).item()  # 隣接層間のCKA値
            # 隣接した層でないと同じグループにはならないので、隣接層間のCKA値のみで十分
            # LinearCKAはtorch.Tensorを受け取るので、feat_t[i]とfeat_t[i+1]はtorch.Tensorである必要がある
            # LinearCKAでCKA計算のために２次元テンソルに変形されるので、ここでは４次元テンソルをそのまま渡してもよい

        # 2. CKA値が高い順にグループ化（貪欲法）
        # まず全層を1つずつグループに分ける
        groups = [[i] for i in range(n)]
        # グループ数が指定数になるまで、CKA値が最大の隣接グループをマージ
        while len(groups) > group_num:
            # 隣接グループ間のCKA値を取得
            merge_scores = [cka_mat[groups[i][-1]] for i in range(len(groups)-1)]
            # 最高のCKA値を持つ隣接グループを見つける
            # ある層とその次の層のCKA値を用いて、隣接グループ間のCKA値を計算
            max_idx = np.argmax(merge_scores)
            # マージ
            groups[max_idx] = groups[max_idx] + groups[max_idx+1] # 隣接グループmax_ind+1をmax_idxにマージ
            del groups[max_idx+1] # マージして必要なくなったグループ(max_ind+1)を削除
        return groups

    def forward(self, feat_s, feat_t):
        # 各グループごとに特徴マップリストを返す
        # s_group_feats = [
        #     [feat_s[0], feat_s[1]],  # グループ1
        #     [feat_s[2], feat_s[3]],  # グループ2
        #     [feat_s[4], feat_s[5]],  # グループ3
        #     [feat_s[6], feat_s[7]],  # グループ4
        # ]
        s_group_feats = [[feat_s[i] for i in idxs] for idxs in self.s_groups]
        t_group_feats = [[feat_t[i] for i in idxs] for idxs in self.t_groups]
        return s_group_feats, t_group_feats
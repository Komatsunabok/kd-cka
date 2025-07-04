import torch
import torch.nn as nn

# 線形CKAの計算
# 2つの層のCKA（Centered Kernel Alignment）を計算する関数
# HSICを用いたCKAの実装よりも、線形CKAを用いた方が計算が高速であるため、
# CKAの代わりに線形CKAを用いることが多い
# torchで計算するためpytorchで実装した知識蒸留の損失関数として使える

def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    入力
    X: Tensor of shape [n, p1] （n個のサンプル・p1次元の特徴量、例：教師のある層）
    Y: Tensor of shape [n, p2] （n個のサンプル・p2次元の特徴量、例：生徒の対応層）
    ※ p1 ≠ p2 でもOK
    ※ バッチサイズ n は一致している必要があります（＝同じデータに対する層出力）

    出力
    cka_value: torch.Tensor（0次元、スカラー）
    → CKA 類似度（範囲は理論的には 0～1に近い）
    """
    # 次元が4次元以上の場合、2次元に変形
    # 4次元なら [B, C, H, W] → [B, C*H*W]
    if X.dim() > 2:
        X = X.view(X.size(0), -1)
    if Y.dim() > 2:
        Y = Y.view(Y.size(0), -1)

    # Centering the data
    # データを中心化（平均を0にする）
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # ノルム計算（Frobenius）
    dot_product_similarity = torch.norm(Y.T @ X, p="fro") ** 2
    normalization_x = torch.norm(X.T @ X, p="fro")
    normalization_y = torch.norm(Y.T @ Y, p="fro")

    return dot_product_similarity / (normalization_x * normalization_y)

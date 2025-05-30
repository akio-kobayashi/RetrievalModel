# soft_dtw.py

import torch
import torch.nn as nn

class SoftDTW(nn.Module):
    """
    Pure-PyTorch implementation of Soft‐DTW (Cuturi & Blondel, 2017).
    Usage:
        sdtw = SoftDTW(gamma=0.1)
        dist = sdtw(x, y)  # x: (N, D), y: (M, D) → scalar
    """

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor[N, D], y: Tensor[M, D]
        returns: scalar Soft-DTW distance
        """
        # 1) ペアワイズ距離行列の計算（ここでは二乗ユークリッド距離）
        D = self._pairwise_dist(x, y)  # shape: (N, M)
        # 2) Soft-DTW の動的計画法
        return self._soft_dtw(D)

    def _pairwise_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # ||x_i - y_j||^2 = x_i^2 + y_j^2 - 2 x_i·y_j
        x2 = x.pow(2).sum(dim=1, keepdim=True)        # (N,1)
        y2 = y.pow(2).sum(dim=1, keepdim=True).t()    # (1,M)
        xy = x @ y.t()                                 # (N,M)
        return x2 + y2 - 2*xy

    def _soft_dtw(self, D: torch.Tensor) -> torch.Tensor:
        N, M = D.size()
        gamma = self.gamma
        # 境界を∞で埋めたコスト行列 R を作成
        R = torch.full((N+1, M+1), float('inf'), device=D.device, dtype=D.dtype)
        R[0, 0] = 0.0

        # 動的計画法のループ
        for i in range(1, N+1):
            for j in range(1, M+1):
                r0 = R[i-1, j-1]
                r1 = R[i-1, j]
                r2 = R[i, j-1]
                # soft-min 演算: -γ·logsumexp( -r_k/γ )
                r_stack = torch.stack([-r0/gamma, -r1/gamma, -r2/gamma], dim=0)
                softmin = -gamma * torch.logsumexp(r_stack, dim=0)
                R[i, j] = D[i-1, j-1] + softmin

        return R[N, M]

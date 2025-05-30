import torch
import torch.nn as nn

class SoftDTW(nn.Module):
    """
    Pure-PyTorch Soft-DTW (Cuturi & Blondel, 2017) with batch support.
    Usage:
        sdtw = SoftDTW(gamma=0.1)
        dist = sdtw(x, y)  # x,y: (N, D) or (B, N, D)
    Returns: scalar or (B,) tensor
    """
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Handle batched inputs
        if x.dim() == 3 and y.dim() == 3:
            # x, y: (B, N, D), (B, M, D)
            distances = []
            for xi, yi in zip(x, y):
                D = self._pairwise_dist(xi, yi)
                distances.append(self._soft_dtw(D))
            return torch.stack(distances)
        # single example: x,y are (N, D) and (M, D)
        D = self._pairwise_dist(x, y)
        return self._soft_dtw(D)

    def _pairwise_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (..., N, D), y: (..., M, D) or 2D
        # Flatten possible leading dims for simplicity
        original_shape = x.shape
        if x.dim() == 3:
            # Should not reach here due to batch handling
            x = x
        # Compute pairwise squared euclidean distances
        x2 = x.pow(2).sum(dim=-1, keepdim=True)  # (..., N, 1)
        y2 = y.pow(2).sum(dim=-1, keepdim=True).transpose(-2, -1)  # (..., 1, M)
        xy = torch.matmul(x, y.transpose(-2, -1))  # (..., N, M)
        return x2 + y2 - 2 * xy

    def _soft_dtw(self, D: torch.Tensor) -> torch.Tensor:
        # D: (N, M)
        N, M = D.size()
        gamma = self.gamma
        device = D.device
        dtype = D.dtype
        R = torch.full((N+1, M+1), float('inf'), device=device, dtype=dtype)
        R[0, 0] = 0.0
        for i in range(1, N+1):
            for j in range(1, M+1):
                r0 = R[i-1, j-1]
                r1 = R[i-1, j]
                r2 = R[i, j-1]
                softmin = -gamma * torch.logsumexp(
                    torch.stack([-r0/gamma, -r1/gamma, -r2/gamma]), dim=0
                )
                R[i, j] = D[i-1, j-1] + softmin
        return R[N, M]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from melmodel import PosteriorEncoder, ConformerGenerator, CrossAttnFusion, ConformerBlock

# -----------------------
# LoRA for Linear
# -----------------------
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(linear.out_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, linear.in_features) * 0.01)
        self.scale = alpha / rank
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x) + F.linear(x, self.scale * self.A @ self.B)

# -----------------------
# LoRA wrapper for MHA
# -----------------------
class LoRAMultiheadAttention(nn.Module):
    def __init__(self, mha: MultiheadAttention, rank=4, alpha=1.0):
        super().__init__()
        self.base = mha
        self.q_lora = LoRALinear(nn.Linear(mha.embed_dim, mha.embed_dim), rank, alpha)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, query, key, value, **kwargs):
        query = self.q_lora(query)
        return self.base(query, key, value, **kwargs)

# -----------------------
# CrossAttnFusion + LoRA
# -----------------------
class CrossAttnFusionWithLoRA(CrossAttnFusion):
    def __init__(self, *args, rank=4, alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_attn = LoRAMultiheadAttention(self.cross_attn, rank, alpha)

# -----------------------
# ConformerBlock + LoRA
# -----------------------
class ConformerBlockWithLoRA(ConformerBlock):
    def __init__(self, *args, rank=4, alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attn = LoRAMultiheadAttention(self.self_attn, rank, alpha)

# -----------------------
# Full model with LoRA
# -----------------------
class RVCStyleVC_LoRA(nn.Module):
    def __init__(self, latent_ch=256, d_model=256, n_conformer=8, nhead=8, rank=4, alpha=1.0):
        super().__init__()
        self.encoder = PosteriorEncoder()
        self.encoder.fusion = CrossAttnFusionWithLoRA(768, nhead, rank=rank, alpha=alpha)

        self.generator = ConformerGenerator(in_ch=latent_ch, d_model=d_model,
                                            n_conformer=n_conformer, nhead=nhead)
        for i, blk in enumerate(self.generator.blocks):
            self.generator.blocks[i] = ConformerBlockWithLoRA(
                blk.self_attn.embed_dim,
                blk.self_attn.num_heads,
                dropout=blk.self_attn.dropout,
                rank=rank,
                alpha=alpha
            )

        # freeze everything except LoRA
        for n, p in self.named_parameters():
            p.requires_grad = ('A' in n or 'B' in n)

    def forward(self, hubert, pitch, target_length: int | None = None, mask=None):
        z = self.encoder(hubert, pitch, mask)
        return self.generator(z, target_length)

import torch
import torch.nn as nn
import torch.nn.functional as F

class HeadwiseLoRA(nn.Module):
    def __init__(self, mha: nn.MultiheadAttention, rank=2, alpha=1.0):
        super().__init__()
        self.mha = mha
        self.num_heads = mha.num_heads
        self.head_dim = mha.embed_dim // self.num_heads
        self.scale = alpha / rank
        # 各ヘッドごとに A,B を持つ
        self.A = nn.Parameter(torch.randn(self.num_heads, self.head_dim, rank) * 0.02)
        self.B = nn.Parameter(torch.zeros(self.num_heads, rank, self.head_dim))
        for p in self.mha.parameters():
            p.requires_grad = False

    def forward(self, query, key, value, **kwargs):
        # 1) 通常の MHA 出力
        out, attn_w = self.mha(query, key, value, **kwargs)
        # 2) LoRA 補正
        B, T, C = query.shape
        q = query.view(B, T, self.num_heads, self.head_dim).transpose(1,2)  # (B, H, T, hd)
        delta_heads = []
        for h in range(self.num_heads):
            # head h にだけ補正
            Ah = self.A[h]    # (hd, r)
            Bh = self.B[h]    # (r, hd)
            # Q_proj 直後を補正すると仮定
            dq = (q[:,h] @ Ah) @ Bh * self.scale  # (B, T, hd)
            delta_heads.append(dq)
        # 再結合
        delta = torch.stack(delta_heads, dim=1)               # (B,H,T,hd)
        delta = delta.transpose(1,2).reshape(B, T, C)        # (B,T,C)
        return out + delta, attn_w

    
class LoRAFFN(nn.Module):
    def __init__(self, ffn: nn.Sequential, rank=4, alpha=1.0):
        super().__init__()
        # ffn = nn.Sequential(Linear(in→hidden), Activation, Linear(hidden→in))
        self.ffn = ffn
        # LoRA を掛けるのは出力側の Linear
        linear_out: nn.Linear = ffn[-1]
        in_ch, out_ch = linear_out.in_features, linear_out.out_features
        self.A = nn.Parameter(torch.randn(in_ch, rank) * 0.02)
        self.B = nn.Parameter(torch.zeros(rank, out_ch))
        self.scale = alpha / rank
        for p in self.ffn.parameters():
            p.requires_grad = False

    def forward(self, x):
        h = self.ffn[0](x)
        h = self.ffn[1](h)
        out = self.ffn[2](h)
        # LoRA 部分
        delta = (h @ (self.A @ self.B).T) * self.scale  # (B, T, out_ch)
        return out + delta

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
    def __init__(self, mha: nn.MultiheadAttention, rank=4, alpha=1.0):
        super().__init__()
        self.base = mha
        self.q_lora = LoRALinear(nn.Linear(mha.embed_dim, mha.embed_dim), rank, alpha)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, query, key, value, **kwargs):
        query = self.q_lora(query)
        return self.base(query, key, value, **kwargs)

# -----------------------
# LoRA utility for TransformerAligner
# -----------------------
def apply_lora_to_transformeraligner(model, rank=4, alpha=1.0):
    for layer in model.encoder_layers:
        layer['self_attn'] = HeadwiseLoRA(layer['self_attn'], rank, alpha)
        layer['pitch_attn'] = HeadwiseLoRA(layer['pitch_attn'], rank, alpha)
        layer['ffn']        = LoRAFFN(layer['ffn'], rank, alpha)
    for layer in model.decoder_layers:
        layer['self_attn'] = HeadwiseLoRA(layer['self_attn'], rank, alpha)
        layer['pitch_attn'] = HeadwiseLoRA(layer['pitch_attn'], rank, alpha)
        layer['cross_attn'] = HeadwiseLoRA(layer['cross_attn'], rank, alpha)
        layer['ffn']        = LoRAFFN(layer['ffn'], rank, alpha)
    return model

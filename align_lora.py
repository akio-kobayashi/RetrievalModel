import torch
import torch.nn as nn
import torch.nn.functional as F

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
        layer['self_attn'] = LoRAMultiheadAttention(layer['self_attn'], rank, alpha)
        layer['pitch_attn'] = LoRAMultiheadAttention(layer['pitch_attn'], rank, alpha)
    for layer in model.decoder_layers:
        layer['self_attn'] = LoRAMultiheadAttention(layer['self_attn'], rank, alpha)
        layer['pitch_attn'] = LoRAMultiheadAttention(layer['pitch_attn'], rank, alpha)
        layer['cross_attn'] = LoRAMultiheadAttention(layer['cross_attn'], rank, alpha)
    return model

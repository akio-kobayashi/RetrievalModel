import torch, math
import torch.nn as nn
import torch.nn.utils.parametrize as P
from melmodel import PosteriorEncoder, ConformerGenerator, CrossAttnFusion

class LoRAParam(nn.Module):
    def __init__(self, w, rank=4, alpha=1.0):
        super().__init__()
        in_ch, out_ch = w.size(1), w.size(0)          # (in, out)
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.randn(in_ch, rank) * 0.02)   # (in , r)
        self.B = nn.Parameter(torch.zeros(rank, out_ch))         # (r  , out)

    def forward(self, w_orig):                                    # w_orig: (out, in)
        # ▲ A @ B → (in, out) を転置して (out, in) に合わせる
        delta = (self.A @ self.B).T * self.scale                  # (out, in)
        return w_orig + delta
   
def inject_lora(module: nn.Module, rank: int = 4, alpha: float = 1.0):
    """
    再帰的に走査し、以下を LoRA 化していく
      • nn.Linear.weight
      • nn.MultiheadAttention.in_proj_weight
      • nn.MultiheadAttention.out_proj.weight
    """
    for child in module.children():
        # 1) Linear
        if isinstance(child, nn.Linear):
            P.register_parametrization(child, "weight",
                LoRAParam(child.weight, rank, alpha))
        # 2) MHA
        elif isinstance(child, nn.MultiheadAttention):
            # in_proj_weight は Parameter
            P.register_parametrization(child, "in_proj_weight",
                LoRAParam(child.in_proj_weight, rank, alpha))
            # out_proj は nn.Linear
            P.register_parametrization(child.out_proj, "weight",
                LoRAParam(child.out_proj.weight, rank, alpha))
        # dive deeper
        inject_lora(child, rank, alpha)

# ------------------------------------------------------
class RVCStyleVC_LoRA(nn.Module):
    def __init__(self, ckpt_path, *, latent_ch=256, d_model=256,
                 n_conformer=8, nhead=8, rank=4, alpha=1.):
        super().__init__()
        # 1) ― ベースモデル（LoRA 無し）
        self.encoder = PosteriorEncoder()
        self.encoder.fusion = CrossAttnFusion(768, nhead, dropout=0.1)
        self.generator = ConformerGenerator(latent_ch, d_model,
                                            n_conformer=n_conformer, nhead=nhead)

        # 2) ― ckpt 読み込み（`gen.` prefix を剥いでフィルタ）
        raw = torch.load(ckpt_path, map_location="cpu")
        raw = raw.get("state_dict", raw)          # Lightning / plain 両対応

        gen_only = {k[4:]: v for k, v in raw.items()           # `gen.` を削る
                    if k.startswith("gen.")}                  # それ以外は無視
        missing, unexpected = self.load_state_dict(gen_only, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"← まだ不一致があります: "
                               f"missing={len(missing)}, unexpected={len(unexpected)}")

        # 3) ― LoRA 注入 & それ以外を凍結
        inject_lora(self, rank, alpha)
        for n, p in self.named_parameters():
            p.requires_grad = ("parametrizations" in n)   # LoRA 行列だけ学習

    def forward(self, hubert, pitch, target_length=None, mask=None):
        z = self.encoder(hubert, pitch, mask)
        return self.generator(z, target_length)

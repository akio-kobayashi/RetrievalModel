# lora_rvc.py
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from melmodel import PosteriorEncoder, ConformerGenerator, CrossAttnFusion

# ————————— AdapterParametrization —————————
class AdapterParametrization(nn.Module):
    def __init__(self, W_orig: nn.Parameter, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        out_ch, in_ch = W_orig.shape
        self.scale = alpha / rank
        # LoRA 行列
        self.A = nn.Parameter(torch.randn(in_ch, rank) * 0.02)
        self.B = nn.Parameter(torch.zeros(rank, out_ch))

    def forward(self, W):
        # W: (out, in)
        delta = (self.A @ self.B).T * self.scale  # (out, in)
        return W + delta

# ————————— インジェクト関数 —————————
def inject_lora(module: nn.Module, rank: int = 4, alpha: float = 1.0):
    """
    module の子モジュールを再帰的にたどって、
      • nn.Linear.weight
      • nn.MultiheadAttention.in_proj_weight
      • nn.MultiheadAttention.out_proj.weight
    に AdapterParametrization を登録します。
    """
    for child in module.children():
        if isinstance(child, nn.Linear):
            P.register_parametrization(child, "weight",
                AdapterParametrization(child.weight, rank, alpha))
        elif isinstance(child, nn.MultiheadAttention):
            # QKV 投影
            P.register_parametrization(child, "in_proj_weight",
                AdapterParametrization(child.in_proj_weight, rank, alpha))
            # 出力投影
            P.register_parametrization(child.out_proj, "weight",
                AdapterParametrization(child.out_proj.weight, rank, alpha))
        inject_lora(child, rank, alpha)

# ————————— RVCStyleVC_LoRA クラス —————————
class RVCStyleVC_LoRA(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        *,
        latent_ch: int = 256,
        d_model: int = 256,
        n_conformer: int = 4,
        nhead: int = 4,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        # 1) 元モデルインスタンス生成（コードは一切変更なし）
        self.encoder = PosteriorEncoder()
        # fusion だけ差し替え済みの CrossAttnFusion を使う
        self.encoder.fusion = CrossAttnFusion(768, nhead, dropout=0.1)

        self.generator = ConformerGenerator(
            in_ch=latent_ch,
            d_model=d_model,
            n_conformer=n_conformer,
            nhead=nhead,
            dropout=0.1,
        )

        # ------------------------------------------------------------
        # 2) ― ckpt 読み込み（Lightning / plain どちらでも OK）
        # ------------------------------------------------------------
        raw = torch.load(ckpt_path, map_location="cpu")
        raw = raw.get("state_dict", raw)           # ← Lightning の場合は state_dict を取り出す

        # -------- ① キー一覧をざっと確認（デバッグ用：必要なら一時的に） --------
        # print("\n".join(list(raw.keys())[:40]))
        
        # -------- ② encoder / generator 用に prefix を切り分ける -------------

        enc_state, gen_state = {}, {}
        for k, v in raw.items():
            if k.startswith("encoder."):
                enc_state[k[len("encoder."):]] = v      # "encoder." を削除して格納
            elif k.startswith("generator."):
                gen_state[k[len("generator."):]] = v    # 同上

        # -------- ③ 想定外のキーが残っていないか軽くチェック -----------------
        unused = [k for k in raw.keys() if not (k.startswith("encoder.") or k.startswith("generator."))]
        if unused:
            print(f"[WARN] {len(unused)} keys were ignored (not encoder./generator.*): e.g. {unused[:5]}")

        # -------- ④ それぞれロード（strict=True で形状もチェック） ------------
        self.encoder.load_state_dict(enc_state, strict=True)
        self.generator.load_state_dict(gen_state, strict=True)
        
        # 3) LoRA 注入 & 重み凍結
        inject_lora(self.encoder,   rank, alpha)
        inject_lora(self.generator, rank, alpha)
        # parametrization（=LoRA部）だけ学習
        for name, p in self.named_parameters():
            p.requires_grad = "parametrizations" in name

    def forward(self, hubert, pitch, target_length=None, mask=None):
        z = self.encoder(hubert, pitch, mask)           # (B, latent_ch, T)
        return self.generator(z, target_length)         # (B, T, mel_dim)

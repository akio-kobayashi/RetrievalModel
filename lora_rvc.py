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
        raw = raw.get("state_dict", raw)

        enc_state, gen_state = {}, {}

        # ------- ① prefix テーブルを用意してループ１回で振り分け ------------
        prefix_map = {
            "gen.encoder.":    ("enc",  len("gen.encoder.")),
            "gen.generator.":  ("gen", len("gen.generator.")),
            "encoder.":        ("enc",  len("encoder.")),
            "generator.":      ("gen", len("generator.")),
        }

        for k, v in raw.items():
            for pre, (which, cut) in prefix_map.items():
                if k.startswith(pre):
                    if which == "enc":
                        enc_state[k[cut:]] = v
                    else:  # "gen"
                        gen_state[k[cut:]] = v
                        break   # マッチしたら次のキーへ
                # else 節は不要（マッチしなければ何もしない）

        # ------- ② 使わなかったキーを警告（確認用） -------------------------
        unused = [k for k in raw.keys() if k not in
                  {pre+k2 for pre in prefix_map for k2 in
                   enc_state.keys() | gen_state.keys()}]
        if unused:
            print(f"[WARN] {len(unused)} keys remain unused, e.g. {unused[:5]}")


        # ------- ③ strict=True でロード --------------------------------------
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

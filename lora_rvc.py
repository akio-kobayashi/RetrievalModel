# lora_rvc.py
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from melmodel import PosteriorEncoder, ConformerGenerator, CrossAttnFusion

def _infer_generator_hparams(sd):
    """
    ckpt の state_dict から
      latent_ch (= input_proj.in_channels),
      d_model   (= input_proj.out_channels),
      n_blocks  (= generator.blocks の数)
    を推定して返す
    """
    # ① input_proj の shape で in/out ch を取得
    w_inproj = sd['gen.generator.input_proj.weight'] if 'gen.generator.input_proj.weight' in sd \
               else sd['generator.input_proj.weight']           # Lightning / plain 両対応
    d_model_ckpt, latent_ch_ckpt, *_ = w_inproj.shape           # (out_ch, in_ch, k)
    
    # ② blocks.* から最大 id を取って層数推定
    pat = re.compile(r'(?:gen\.)?generator\.blocks\.(\d+)\.')
    n_blocks_ckpt = max(int(m.group(1)) for k in sd.keys() if (m := pat.match(k))) + 1
    
    return latent_ch_ckpt, d_model_ckpt, n_blocks_ckpt

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
        *,                       # 昔からある手動引数（ None なら自動推定）
        latent_ch: int | None = None,
        d_model:   int | None = None,
        n_conformer: int | None = None,
        nhead: int = 8,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()

        # 1) ckpt 読み込み ------------------------------------------------------
        raw = torch.load(ckpt_path, map_location="cpu")
        sd  = raw.get("state_dict", raw)             # Lightning / plain 両対応

        # 2) ★ 必要なら ckpt から自動推定 -------------------------------------
        if None in (latent_ch, d_model, n_conformer):
            latent_ch_ckpt, d_model_ckpt, n_blocks_ckpt = _infer_from_ckpt(sd)
            latent_ch    = latent_ch    or latent_ch_ckpt
            d_model      = d_model      or d_model_ckpt
            n_conformer  = n_conformer  or n_blocks_ckpt
            print(f"[INFO] auto-infer → latent={latent_ch}, d_model={d_model}, "
                  f"n_blocks={n_conformer}")

        # 3) モデル生成（コードはオリジナルのまま） ----------------------------
        self.encoder = PosteriorEncoder(latent_ch=latent_ch)
        self.encoder.fusion = CrossAttnFusion(768, nhead, dropout=0.1)

        self.generator = ConformerGenerator(
            in_ch=latent_ch,
            d_model=d_model,
            n_conformer=n_conformer,
            nhead=nhead,
            dropout=0.1,
        )

        # 4) state_dict を prefix で振り分け -----------------------------------
        enc_state, gen_state = {}, {}
        for k, v in sd.items():
            if   k.startswith("gen.encoder."): enc_state[k[12:]] = v
            elif k.startswith("encoder.")    : enc_state[k[8:]]  = v
            elif k.startswith("gen.generator."): gen_state[k[14:]] = v
            elif k.startswith("generator.")    : gen_state[k[10:]] = v

        # 5) ロード（strict=True でズレがあれば即エラー） ----------------------
        self.encoder.load_state_dict(enc_state, strict=True)
        self.generator.load_state_dict(gen_state, strict=True)

        # 6) LoRA 注入 & パラメータ凍結 ----------------------------------------
        inject_lora(self, rank, alpha)
        for n, p in self.named_parameters():
            p.requires_grad = "parametrizations" in n    # A/B だけ学習

    def forward(self, hubert, pitch, *, target_length=None, mask=None):
        z = self.encoder(hubert, pitch, mask)
        return self.generator(z, target_length)

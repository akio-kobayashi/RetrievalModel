import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from typing import List
from einops.layers.torch import Rearrange

# --------------------------------------------------
# HuBERT-pitch Cross-Attention Fusion
# --------------------------------------------------
class CrossAttnFusion(nn.Module):
    def __init__(self, hub_dim: int = 768, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        # pitch(1) → hub_dim へ射影
        self.pitch_proj = nn.Linear(1, hub_dim)
        # HuBERT(Q) × pitch(K,V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hub_dim, num_heads=nhead,
            dropout=dropout, batch_first=True
        )
        # 残差後を正規化
        self.ln = nn.LayerNorm(hub_dim)

    def forward(self, hubert: torch.Tensor, pitch: torch.Tensor, mask=None):
        """
        hubert : (B, T, 768)
        pitch  : (B, T)     → 内部で (B, T, 1) に unsqueeze
        mask   : (B, T) or None
        """
        p_emb = self.pitch_proj(pitch.unsqueeze(-1))      # (B, T, 768)
        attn_out, _ = self.cross_attn(
            query=hubert,            # Q
            key=p_emb, value=p_emb,  # K,V
            key_padding_mask=mask
        )
        # HuBERT へ pitch 情報を注入（residual）
        return self.ln(hubert + attn_out)                 # (B, T, 768)

# --------------------------------------------------
# Conformer Block
# --------------------------------------------------
class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead, ff_mult=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_mult * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.conv = nn.Sequential(
            Rearrange("b c t -> b t c"),
            nn.LayerNorm(d_model),
            Rearrange("b t c -> b c t"),
            nn.Conv1d(d_model, d_model, 5, padding=2, groups=d_model),
            nn.GLU(dim=1),
            nn.Conv1d(d_model // 2, d_model, 1),
            nn.Dropout(dropout),
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_mult * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):  # x: (B, T, D)
        x = x + self.self_attn(x, x, x, key_padding_mask=mask)[0]
        x = x + self.ffn1(x)
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.ffn2(x)
        return self.norm(x)

# --------------------------------------------------
# Posterior Encoder  (HuBERT × pitch → latent)
# --------------------------------------------------
class PosteriorEncoder(nn.Module):
    def __init__(self, latent_ch: int = 256, n_layers: int = 4,
                 nhead: int = 8, dropout: float = 0.1):
        super().__init__()

        self.fusion = CrossAttnFusion(768, nhead, dropout)  # ★ 追加部分 ★

        # 以下は元コードとほぼ同じ ― conv-GLU × n_layers
        ch_in = 768
        layers = []
        for _ in range(n_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(ch_in, latent_ch * 2, 5, padding=2),
                    nn.GLU(dim=1),
                    nn.BatchNorm1d(latent_ch),
                )
            )
            ch_in = latent_ch
        self.net = nn.Sequential(*layers)

    def forward(self, hubert, pitch, mask=None):           # hubert:(B,T,768) pitch:(B,T)
        # ① pitch をクロスアテンションで注入
        x = self.fusion(hubert, pitch, mask)               # (B,T,768)

        # ② conv スタックへ (B,C,T) 形状で渡す
        x = self.net(x.transpose(1, 2))                    # (B, latent_ch, T)
        return x


# --------------------------------------------------
# Conformer-based Generator (latent_C, upsample -> mel)
# --------------------------------------------------
class ConformerGenerator(nn.Module):
    def __init__(
        self,
        in_ch=256,
        d_model=256,
        mel_dim=80,
        n_conformer=4,
        nhead=4,
        dropout=0.1,
        upsample_factor=1.72,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_ch, d_model, 1)
        self.blocks = nn.ModuleList([ConformerBlock(d_model, nhead, dropout=dropout) for _ in range(n_conformer)])
        self.out_proj = nn.Linear(d_model, mel_dim)
        self.upsample_factor = upsample_factor

    def forward(self, x, target_length: int | None = None):  # x: (B, C, T)
        x = self.input_proj(x).transpose(1, 2)  # (B,T,C)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2)  # (B,C,T)
        if target_length is not None:
            x = F.interpolate(x, size=target_length, mode="linear", align_corners=True)
        elif self.upsample_factor != 1.0:
            x = F.interpolate(x, scale_factor=self.upsample_factor, mode="linear", align_corners=True)
        return self.out_proj(x.transpose(1, 2))  # (B, T, 80)

# --------------------------------------------------
# Mel‑space Discriminators (MPD / MSD)
# --------------------------------------------------
class MelPeriodDiscriminator(nn.Module):
    def __init__(self, period: int, mel_dim: int = 80):
        super().__init__()
        self.period = period
        chs = [mel_dim, 32, 128, 512, 1024, 1024]
        ks = [5, 3, 3, 3, 3]
        seq: List[nn.Module] = []
        for i in range(len(ks)):
            seq += [
                spectral_norm(nn.Conv2d(chs[i], chs[i + 1], (ks[i], 1), stride=(3 if i == 0 else 1, 1), padding=((ks[i] - 1) // 2, 0))),
                nn.LeakyReLU(0.1),
            ]
        seq.append(spectral_norm(nn.Conv2d(chs[-1], 1, (3, 1), padding=(1, 0))))
        self.convs = nn.ModuleList(seq)

    def forward(self, x):  # x: (B, M, T)
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            t += pad
        x = x.view(b, c, self.period, t // self.period)
        feats = []
        for l in self.convs[:-1]:
            x = l(x)
            feats.append(x)
        out = self.convs[-1](x)
        feats.append(out)
        return out.flatten(1, -1), feats

class MelMultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), mel_dim=80):
        super().__init__()
        self.discs = nn.ModuleList([MelPeriodDiscriminator(p, mel_dim) for p in periods])

    def forward(self, x):
        logits, feats = [], []
        for d in self.discs:
            l, f = d(x)
            logits.append(l); feats.append(f)
        return logits, feats

class MelScaleDiscriminator(nn.Module):
    def __init__(self, mel_dim=80):
        super().__init__()
        chs = [mel_dim, 128, 128, 256, 512, 1024, 1024]
        ks = [15, 41, 41, 41, 41, 5]
        ss = [1, 4, 4, 4, 4, 1]
        groups = [1, 4, 16, 16, 16, 1]
        seq: List[nn.Module] = []
        for i in range(len(ks)):
            seq += [
                spectral_norm(nn.Conv1d(chs[i], chs[i + 1], ks[i], stride=ss[i], padding=(ks[i] - 1) // 2, groups=groups[i])),
                nn.LeakyReLU(0.1),
            ]
        seq.append(spectral_norm(nn.Conv1d(chs[-1], 1, 3, padding=1)))
        self.convs = nn.ModuleList(seq)

    def forward(self, x):  # x: (B, M, T)
        feats = []
        for l in self.convs[:-1]:
            x = l(x)
            feats.append(x)
        out = self.convs[-1](x)
        feats.append(out)
        return out.flatten(1, -1), feats

class MelMultiScaleDiscriminator(nn.Module):
    def __init__(self, mel_dim=80):
        super().__init__()
        self.discs = nn.ModuleList([MelScaleDiscriminator(mel_dim) for _ in range(3)])
        self.pools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)])

    def forward(self, x):
        logits, feats = [], []
        for i, d in enumerate(self.discs):
            y = x if i == 0 else self.pools[i - 1](x)
            l, f = d(y)
            logits.append(l); feats.append(f)
        return logits, feats

# --------------------------------------------------
# Full RVC model (Encoder + Generator)
# --------------------------------------------------
class RVCStyleVC(nn.Module):
    def __init__(self, latent_ch=256, d_model=256, n_conformer=8, nhead=8):
        super().__init__()
        self.encoder = PosteriorEncoder(latent_ch, nhead=nhead)  # ← new encoder
        self.generator = ConformerGenerator(
            in_ch=latent_ch, d_model=d_model,
            n_conformer=n_conformer, nhead=nhead
        )

    def forward(self, hubert, pitch, target_length: int | None = None, mask=None):
        z = self.encoder(hubert, pitch, mask)              # (B,C,T)
        mel = self.generator(z, target_length)             # (B,T,80)
        return mel

# --------------------------------------------------
if __name__ == "__main__":
    B, T = 2, 100
    hub, f0 = torch.randn(B, T, 768), torch.rand(B, T)
    model = RVCStyleVC()
    mel = model(hub, f0)                     # (B,T,80)
    print("Generator output", mel.shape)

    mpd = MelMultiPeriodDiscriminator()
    msd = MelMultiScaleDiscriminator()
    logits_mpd, _ = mpd(mel.transpose(1, 2))
    logits_msd, _ = msd(mel.transpose(1, 2))
    print("MPD logits shapes", [l.shape for l in logits_mpd])
    print("MSD logits shapes", [l.shape for l in logits_msd])

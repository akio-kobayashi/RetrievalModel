import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from typing import List, Tuple, Optional

# ---- Conformer Block -------------------------------------------------------
# PyTorch公式やespnetのConformerBlockと互換あり
class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead, ff_mult=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_mult*d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult*d_model, d_model),
            nn.Dropout(dropout)
        )
        self.conv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Conv1d(d_model, d_model, 5, padding=2, groups=d_model),  # Depthwise Conv
            nn.GLU(dim=1),  # gating
            nn.Conv1d(d_model, d_model, 1),  # pointwise
            nn.Dropout(dropout)
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_mult*d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult*d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: (B, T, D)
        x_attn, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = x + x_attn
        x = x + self.ffn1(x)
        # Conv expects (B,D,T)
        x_conv = x.transpose(1,2)
        x_conv = self.conv(x_conv)
        x = x + x_conv.transpose(1,2)
        x = x + self.ffn2(x)
        x = self.norm(x)
        return x

# ---- Posterior Encoder (Conv stack) ----------------------------------------
class PosteriorEncoder(nn.Module):
    """Compress HuBERT+pitch to latent channels."""
    def __init__(self, latent_ch=256, n_layers=4):
        super().__init__()
        layers = []
        ch_in = 768 + 1  # HuBERT(768)+pitch
        for _ in range(n_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(ch_in, latent_ch, 5, padding=2),
                    nn.GLU(dim=1),
                    nn.BatchNorm1d(latent_ch)
                )
            )
            ch_in = latent_ch
        self.net = nn.Sequential(*layers)

    def forward(self, hubert, pitch):
        # hubert: (B,T,768), pitch: (B,T)
        hubert = hubert.transpose(1,2)  # (B,768,T)
        pitch = pitch.unsqueeze(1)      # (B,1,T)
        x = torch.cat([hubert, pitch], 1)
        return self.net(x)  # (B,latent_ch,T)

# ---- Conformer-based Generator --------------------------------------------
class ConformerGenerator(nn.Module):
    def __init__(self, in_ch=256, d_model=256, mel_dim=80, n_conformer=4, nhead=4, upsample_rates=[5,2,4,2,2]):
        super().__init__()
        self.input_proj = nn.Conv1d(in_ch, d_model, 1)
        self.conformers = nn.ModuleList([
            ConformerBlock(d_model, nhead) for _ in range(n_conformer)
        ])
        self.upsample_blocks = nn.ModuleList()
        ch = d_model
        for r in upsample_rates:
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Upsample(scale_factor=r, mode='nearest'),
                    weight_norm(nn.Conv1d(ch, ch//2, 5, padding=2)),
                    nn.LeakyReLU(0.1),
                    weight_norm(nn.Conv1d(ch//2, ch//2, 1)),  # Depthwise lowpass
                )
            )
            ch //= 2
        self.post = nn.Sequential(
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(ch, 1, 7, padding=3)),  # 波形生成
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.input_proj(x)
        x = x.transpose(1,2)  # (B,T,C)
        for block in self.conformers:
            x = block(x)
        x = x.transpose(1,2)  # (B,C,T)
        for up in self.upsample_blocks:
            x = up(x)
        return self.post(x)  # (B,1,T_wav)

# ---- Discriminators (MPD/MSD, 波形判定, 可変長対応) ------------------------
class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        seq = []
        chs = [1, 32, 128, 512, 1024, 1024]
        ks = [5, 3, 3, 3, 3]
        for i in range(len(ks)):
            seq.append(spectral_norm(
                nn.Conv2d(
                    chs[i], chs[i+1],
                    (ks[i], 1),
                    stride=(3 if i ==0 else 1, 1),
                    padding=((ks[i]-1)//2, 0)
                )
            ))
            seq.append(nn.LeakyReLU(0.1))
        seq.append(spectral_norm(
            nn.Conv2d(chs[-1], 1, (3, 1), padding=(1,0))
        ))
        self.convs = nn.ModuleList(seq)

    def forward(self, x):
        b, _, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            t = t + pad
        x = x.view(b, 1, self.period, t // self.period)
        feats = []
        for l in self.convs[:-1]:
            x = l(x)
            feats.append(x)
        out = self.convs[-1](x)
        feats.append(out)
        return out.flatten(1, -1), feats

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=[2,3,5,7,11]):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x):
        logits, feats = [], []
        for d in self.discriminators:
            l, f = d(x)
            logits.append(l)
            feats.append(f)
        return logits, feats

class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        chs = [1, 128, 128, 256, 512, 1024, 1024]
        ks = [15, 41, 41, 41, 41, 5]
        ss = [1, 4, 4, 4, 4, 1]
        groups = [1, 4, 16, 16, 16, 1]
        seq = []
        for i in range(len(ks)):
            seq.append(spectral_norm(
                nn.Conv1d(
                    chs[i], chs[i+1], ks[i], stride=ss[i], padding=(ks[i]-1)//2, groups=groups[i]
                )
            ))
            seq.append(nn.LeakyReLU(0.1))
        seq.append(spectral_norm(nn.Conv1d(chs[-1], 1, 3, padding=1)))
        self.convs = nn.ModuleList(seq)

    def forward(self, x):
        feats = []
        for l in self.convs[:-1]:
            x = l(x)
            feats.append(x)
        out = self.convs[-1](x)
        feats.append(out)
        return out.flatten(1, -1), feats

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([ScaleDiscriminator() for _ in range(3)])
        self.avgpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, x):
        logits, feats = [], []
        for i, d in enumerate(self.discriminators):
            if i == 0:
                y = x
            elif i == 1:
                y = self.avgpools[0](x)
            else:
                y = self.avgpools[1](self.avgpools[0](x))
            l, f = d(y)
            logits.append(l)
            feats.append(f)
        return logits, feats

# ---- RVC VCモデル本体 (ConformerGen) ---------------------------------------
class RVCStyleVC(nn.Module):
    def __init__(self, latent_ch=256, d_model=256, n_conformer=4, nhead=4, upsample_rates=[5,2,4,2,2]):
        super().__init__()
        self.encoder = PosteriorEncoder(latent_ch=latent_ch)
        self.generator = ConformerGenerator(
            in_ch=latent_ch, d_model=d_model, n_conformer=n_conformer, nhead=nhead, upsample_rates=upsample_rates
        )

    def forward(self, hubert, pitch):
        z = self.encoder(hubert, pitch)     # (B, latent_ch, T)
        wav = self.generator(z)             # (B, 1, T_wav)
        return wav.squeeze(1)               # (B, T_wav)

# ---- 損失関数側: 切り詰め/truncateの例 ------------------------------------
def gan_loss_fn(gen_wav, real_wav):
    min_len = min(gen_wav.size(-1), real_wav.size(-1))
    gen = gen_wav[..., :min_len]
    real = real_wav[..., :min_len]
    return F.l1_loss(gen, real)  # + STFT Loss等も併用可

# ---- テスト ---------------------------------------------------------------
if __name__ == "__main__":
    B, T = 2, 100
    hub = torch.randn(B, T, 768)
    f0 = torch.rand(B, T)
    model = RVCStyleVC()
    wav = model(hub, f0)
    print("Generator output:", wav.shape)  # e.g. (B, T_wav)

    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    logits_mpd, feats_mpd = mpd(wav.unsqueeze(1))
    logits_msd, feats_msd = msd(wav.unsqueeze(1))
    print("MPD logits:", [l.shape for l in logits_mpd])
    print("MSD logits:", [l.shape for l in logits_msd])

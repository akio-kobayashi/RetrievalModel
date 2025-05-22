import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm  # ← 追加
from torch.nn.utils.parametrizations import spectral_norm

###############################################################################
#  Posterior Encoder – HuBERT (+ Pitch) → latent z                           #
###############################################################################

class ConvGLU(nn.Module):
    """1‑D Conv + Gated Linear Unit."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1):
        super().__init__()
        p = (k - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch * 2, k, stride=s, padding=p)
        self.bn = nn.BatchNorm1d(out_ch * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        a, b = x.chunk(2, 1)
        return a * torch.sigmoid(b)


class PosteriorEncoder(nn.Module):
    """Compress HuBERT (768) + pitch (1) to latent channels."""

    def __init__(self, latent_ch: int = 256, n_layers: int = 4):
        super().__init__()
        layers = []
        ch_in = 769
        for _ in range(n_layers):
            layers.append(ConvGLU(ch_in, latent_ch, k=5))
            ch_in = latent_ch
        self.net = nn.Sequential(*layers)

    def forward(self, hubert: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        hubert = hubert.transpose(1, 2)        # (B,768,T)
        pitch = pitch.unsqueeze(1)             # (B,1,T)
        x = torch.cat([hubert, pitch], 1)
        return self.net(x)                     # (B,C,T)

###############################################################################
#  Generator – HiFi‑GAN style                                                #
###############################################################################

class ResBlock(nn.Module):
    def __init__(self, ch: int, kernels: List[int] = [3, 7, 11]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv1d(ch, ch, k, padding=(k - 1) // 2)),
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv1d(ch, ch, k, padding=(k - 1) // 2)),
            )
            for k in kernels
        ])

    def forward(self, x):
        return sum(c(x) for c in self.convs) / len(self.convs) + x


class Generator(nn.Module):
    """HiFi‑GAN like Generator (RVC v2)."""

    def __init__(self, in_ch: int = 256, upsample_rates: List[int] = [10, 8, 2, 2]):
        super().__init__()
        self.pre = weight_norm(nn.Conv1d(in_ch, 512, 7, padding=3))
        ch = 512
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for r in upsample_rates:
            self.ups.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    weight_norm(
                        nn.ConvTranspose1d(
                            ch,
                            ch // 2,
                            kernel_size=r * 2,
                            stride=r,
                            padding=r // 2 + r % 2,
                            output_padding=r % 2,
                        )
                    ),
                )
            )
            ch //= 2
            self.resblocks.append(nn.ModuleList([ResBlock(ch) for _ in range(3)]))
        self.post = nn.Sequential(
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(ch, 1, 7, padding=3)),
            #nn.Tanh(),
        )

    def forward(self, x):
        x = self.pre(x)
        for up, rbs in zip(self.ups, self.resblocks):
            x = up(x)
            x = sum(rb(x) for rb in rbs) / len(rbs)
        return self.post(x)

###############################################################################
#  Discriminators (MPD + MSD)                                                #
###############################################################################

class PeriodDiscriminator(nn.Module):
    """Sub‑discriminator for a single period P in MPD."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        seq = []
        chs = [1, 32, 128, 512, 1024, 1024]
        ks = [5, 3, 3, 3, 3]
        for i in range(len(ks)):
            seq.append(
                #weight_norm(
                #    nn.Conv2d(
                #        chs[i], chs[i + 1], (ks[i], 1), stride=(3 if i == 0 else 1, 1), padding=((ks[i] - 1) // 2, 0)
                #    )
                #)
                spectral_norm(
                    nn.Conv2d(
                        chs[i], chs[i+1],
                        (ks[i], 1),
                        stride=(3 if i ==0 else 1, 1),
                        padding=((ks[i]-1)//2, 0), 
                    )
                )
            )
            seq.append(nn.LeakyReLU(0.1))
        #seq.append(weight_norm(nn.Conv2d(chs[-1], 1, (3, 1), padding=(1, 0))))
        seq.append(spectral_norm(
            nn.Conv2d(chs[-1], 1, (3, 1), padding=(1, 0))
        ))
        self.convs = nn.ModuleList(seq)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x: (B,1,T) → reshape to (B,1,P,T//P)
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
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x: torch.Tensor):
        logits, feats = [], []
        for d in self.discriminators:
            l, f = d(x)
            logits.append(l)
            feats.append(f)
        return logits, feats


class ScaleDiscriminator(nn.Module):
    """Conv1d discriminator used at a single temporal scale."""

    def __init__(self):
        super().__init__()
        chs = [1, 128, 128, 256, 512, 1024, 1024]
        ks = [15, 41, 41, 41, 41, 5]
        ss = [1, 4, 4, 4, 4, 1]
        groups = [1, 4, 16, 16, 16, 1]
        seq = []
        for i in range(len(ks)):
            seq.append(
                nn.utils.spectral_norm(
                    nn.Conv1d(
                        chs[i], chs[i + 1], ks[i], stride=ss[i], padding=(ks[i] - 1) // 2, groups=groups[i]
                    )
                )
            )
            seq.append(nn.LeakyReLU(0.1))
        seq.append(nn.utils.spectral_norm(nn.Conv1d(chs[-1], 1, 3, padding=1)))
        self.convs = nn.ModuleList(seq)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
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
        self.avgpools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)])

    def forward(self, x: torch.Tensor):
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

###############################################################################
#  Full VC Model (Encoder + Generator)                                      #
###############################################################################

class RVCStyleVC(nn.Module):
    def __init__(self, latent_ch: int = 256, upsample_rates: List[int] = [10, 8, 2, 2]):
        super().__init__()
        self.encoder = PosteriorEncoder(latent_ch=latent_ch)
        self.generator = Generator(in_ch=latent_ch, upsample_rates=upsample_rates)

    def forward(self, hubert: torch.Tensor, pitch: torch.Tensor):
        z = self.encoder(hubert, pitch)
        wav = self.generator(z)
        return wav.squeeze(1)

###############################################################################
#  Smoke test                                                               #
###############################################################################

if __name__ == "__main__":
    B, T = 2, 100
    hub = torch.randn(B, T, 768)
    f0 = torch.rand(B, T) * 300
    model = RVCStyleVC()
    wav = model(hub, f0)
    print("Generator output:", wav.shape)

    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    logits_mpd, feats_mpd = mpd(wav.unsqueeze(1))
    logits_msd, feats_msd = msd(wav.unsqueeze(1))
    print("MPD logits shapes:", [l.shape for l in logits_mpd])
    print("MSD logits shapes:", [l.shape for l in logits_msd])

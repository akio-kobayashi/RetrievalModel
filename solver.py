from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model import RVCStyleVC, MultiPeriodDiscriminator, MultiScaleDiscriminator

# ------------------------------------------------------------
#  Multi‑Resolution STFT Loss                                 
# ------------------------------------------------------------
class MRSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(1024, 2048, 512), hop_sizes=(120, 240, 50), win_lengths=(600, 1200, 240)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.register_buffer("window", torch.hann_window(2048), persistent=False)

    def stft(self, x, fft, hop, win):
        w = self.window[:win].to(x.device)
        return torch.stft(x, fft, hop, win, window=w, return_complex=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mag_loss = sc_loss = 0.0
        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            X, Y = self.stft(x, fft, hop, win), self.stft(y, fft, hop, win)
            magX, magY = torch.abs(X), torch.abs(Y)
            mag_loss += F.l1_loss(magX, magY)
            sc_loss  += torch.mean((magY - magX) ** 2 / (magY ** 2 + 1e-7))
        n = len(self.fft_sizes)
        return mag_loss / n, sc_loss / n

# ------------------------------------------------------------
#  LightningModule                                            
# ------------------------------------------------------------
class VCSystem(pl.LightningModule):
    """RVC‑style GAN with manual optimization and length‑safe losses."""

    def __init__(
        self,
        sr: int = 16000,
        hop: int = 320,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        lambda_fm: float = 2.0,
        lambda_mag: float = 1.0,
        lambda_sc: float = 1.0,
        sched_gamma: float = 0.5,
        sched_step: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.gen = RVCStyleVC()
        self.disc_mpd = MultiPeriodDiscriminator()
        self.disc_msd = MultiScaleDiscriminator()
        self.stft_loss = MRSTFTLoss()

    # ---------------- helper losses ----------------
    @staticmethod
    def _adv_d(real, fake):
        return sum(F.mse_loss(r, torch.ones_like(r)) + F.mse_loss(f, torch.zeros_like(f)) for r, f in zip(real, fake)) / len(real)

    @staticmethod
    def _adv_g(fake):
        return sum(F.mse_loss(f, torch.ones_like(f)) for f in fake) / len(fake)

    @staticmethod
    def _feat_match(real_feats, fake_feats):
        loss, cnt = 0.0, 0
        for r_list, f_list in zip(real_feats, fake_feats):
            for r, f in zip(r_list, f_list):
                L = min(r.size(-1), f.size(-1))
                loss += F.l1_loss(f[..., :L], r[..., :L])
                cnt += 1
        return loss / max(cnt, 1)

    # ---------------- inference ----------------
    def forward(self, hubert, pitch):
        return self.gen(hubert, pitch)

    # ---------------- training ----------------
    def training_step(self, batch, batch_idx):
        hub, pit, wav_real = batch
        opt_g, opt_d = self.optimizers()

        # ---- Discriminator update ----
        wav_fake = self.gen(hub, pit).detach()
        cut_len = min(wav_real.size(-1), wav_fake.size(-1))
        wav_real_c = wav_real[..., :cut_len]
        wav_fake_c = wav_fake[..., :cut_len]

        opt_d.zero_grad()
        rl_mpd, _ = self.disc_mpd(wav_real_c.unsqueeze(1))
        rl_msd, _ = self.disc_msd(wav_real_c.unsqueeze(1))
        fk_mpd, _ = self.disc_mpd(wav_fake_c.unsqueeze(1))
        fk_msd, _ = self.disc_msd(wav_fake_c.unsqueeze(1))
        loss_d = self._adv_d(rl_mpd, fk_mpd) + self._adv_d(rl_msd, fk_msd)
        self.manual_backward(loss_d)
        opt_d.step()

        # ---- Generator update ----
        wav_fake = self.gen(hub, pit)
        wav_fake_c = wav_fake[..., :cut_len]
        fk_mpd, fk_feat_mpd = self.disc_mpd(wav_fake_c.unsqueeze(1))
        fk_msd, fk_feat_msd = self.disc_msd(wav_fake_c.unsqueeze(1))
        loss_adv = self._adv_g(fk_mpd) + self._adv_g(fk_msd)

        _, rl_feat_mpd = self.disc_mpd(wav_real_c.unsqueeze(1).detach())
        _, rl_feat_msd = self.disc_msd(wav_real_c.unsqueeze(1).detach())
        loss_fm = self._feat_match(rl_feat_mpd, fk_feat_mpd) + self._feat_match(rl_feat_msd, fk_feat_msd)
        loss_mag, loss_sc = self.stft_loss(wav_real_c, wav_fake_c)

        loss_g = loss_adv + self.hparams.lambda_fm * loss_fm + self.hparams.lambda_mag * loss_mag + self.hparams.lambda_sc * loss_sc

        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()

        self.log_dict({
            "loss_d": loss_d,
            "loss_g": loss_g,
            "loss_adv": loss_adv,
            "loss_fm": loss_fm,
            "loss_mag": loss_mag,
            "loss_sc": loss_sc,
        }, prog_bar=True, on_step=True)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        hub, pit, wav_real = batch
        wav_fake = self.gen(hub, pit)
        cut_len = min(wav_real.size(-1), wav_fake.size(-1))
        wav_real = wav_real[..., :cut_len]
        wav_fake = wav_fake[..., :cut_len]
        mag, sc = self.stft_loss(wav_real, wav_fake)
        self.log_dict({"val_mag": mag, "val_sc": sc}, prog_bar=True)

    # ---------------- optimizers & schedulers ----------------
    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.gen.parameters(), lr=self.hparams.lr_g, betas=(0.8, 0.99))
        opt_d = torch.optim.AdamW(list(self.disc_mpd.parameters()) + list(self.disc_msd.parameters()), lr=self.hparams.lr_d, betas=(0.8, 0.99))

        sched_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=self.hparams.sched_step, gamma=self.hparams.sched_gamma)
        sched_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=self.hparams.sched_step, gamma=self.hparams.sched_gamma)

        return ([opt_g, opt_d], [
            {"scheduler": sched_g, "interval": "epoch"},
            {"scheduler": sched_d, "interval": "epoch"},
        ])

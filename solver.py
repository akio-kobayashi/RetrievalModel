from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

#--------------------------------------------------------------------------------------------------
#  Import generator / discriminators (assumes model.py is in PYTHONPATH or same dir)        
#--------------------------------------------------------------------------------------------------
from model import (
    RVCStyleVC,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)

#--------------------------------------------------------------------------------------------------
#  Multi‑Resolution STFT Loss (minimal implementation)                                            
#--------------------------------------------------------------------------------------------------
class MRSTFTLoss(nn.Module):
    """Multi‑resolution STFT loss used in HiFi‑GAN / RVC."""

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
        loss_mag = 0.0
        loss_sc = 0.0
        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            X = self.stft(x, fft, hop, win)
            Y = self.stft(y, fft, hop, win)
            magX, magY = torch.abs(X), torch.abs(Y)
            loss_mag += F.l1_loss(magX, magY)
            loss_sc += torch.mean((magY - magX) ** 2 / (magY ** 2 + 1e-7))
        return loss_mag / len(self.fft_sizes), loss_sc / len(self.fft_sizes)

#--------------------------------------------------------------------------------------------------
#  LightningModule                                                                                
#--------------------------------------------------------------------------------------------------
class VCSystem(pl.LightningModule):
    """RVC‑style voice conversion system with GAN training."""

    def __init__(
        self,
        sr: int = 16000, # 出力サンプリングレート
        hop: int = 320,  # HuBertのフレームレートが20msなので，16kHzの場合は16000*20/1000=320
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        lambda_fm: float = 2.0,
        lambda_mag: float = 1.0,
        lambda_sc: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.gen = RVCStyleVC()
        self.disc_mpd = MultiPeriodDiscriminator()
        self.disc_msd = MultiScaleDiscriminator()

        self.stft_loss = MRSTFTLoss()

    #----------------------------------------------------------------------------
    #  helper functions for GAN losses                                           
    #----------------------------------------------------------------------------
    @staticmethod
    def _adv_d(real_logits, fake_logits):
        loss = 0.0
        for r, f in zip(real_logits, fake_logits):
            loss += F.mse_loss(r, torch.ones_like(r)) + F.mse_loss(f, torch.zeros_like(f))
        return loss / len(real_logits)

    @staticmethod
    def _adv_g(fake_logits):
        loss = 0.0
        for f in fake_logits:
            loss += F.mse_loss(f, torch.ones_like(f))
        return loss / len(fake_logits)

    @staticmethod
    def _feature_matching(real_feats, fake_feats):
        loss = 0.0
        for rf_list, ff_list in zip(real_feats, fake_feats):
            for rf, ff in zip(rf_list, ff_list):
                loss += F.l1_loss(ff, rf)
        return loss / sum(len(r) for r in real_feats)

    #----------------------------------------------------------------------------
    #  forward only for inference                                                
    #----------------------------------------------------------------------------
    def forward(self, hubert, pitch):
        return self.gen(hubert, pitch)

    #----------------------------------------------------------------------------
    #  training_step with manual optimisation                                   
    #----------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        hubert, pitch, wav_real = batch  # shapes: (B,T,768), (B,T), (B,N)
        opt_g, opt_d = self.optimizers()

        # -------------------- Generator forward --------------------
        wav_fake = self.gen(hubert, pitch).detach()  # detach for D first

        # 1) Train Discriminators
        opt_d.zero_grad()
        real_logits_mpd, real_feats_mpd = self.disc_mpd(wav_real.unsqueeze(1))
        real_logits_msd, real_feats_msd = self.disc_msd(wav_real.unsqueeze(1))
        fake_logits_mpd, _ = self.disc_mpd(wav_fake.unsqueeze(1))
        fake_logits_msd, _ = self.disc_msd(wav_fake.unsqueeze(1))

        loss_d = self._adv_d(real_logits_mpd, fake_logits_mpd) + self._adv_d(real_logits_msd, fake_logits_msd)
        self.manual_backward(loss_d)
        opt_d.step()

        # -------------------- Generator update --------------------
        wav_fake = self.gen(hubert, pitch)  # forward again for G gradient
        fake_logits_mpd, fake_feats_mpd = self.disc_mpd(wav_fake.unsqueeze(1))
        fake_logits_msd, fake_feats_msd = self.disc_msd(wav_fake.unsqueeze(1))

        # adversarial
        loss_adv = self._adv_g(fake_logits_mpd) + self._adv_g(fake_logits_msd)

        # feature matching
        _, real_feats_mpd = self.disc_mpd(wav_real.unsqueeze(1).detach())
        _, real_feats_msd = self.disc_msd(wav_real.unsqueeze(1).detach())
        loss_fm = self._feature_matching(real_feats_mpd, fake_feats_mpd) + self._feature_matching(real_feats_msd, fake_feats_msd)

        # STFT losses
        loss_mag, loss_sc = self.stft_loss(wav_real, wav_fake)

        loss_g = loss_adv + self.hparams.lambda_fm * loss_fm + self.hparams.lambda_mag * loss_mag + self.hparams.lambda_sc * loss_sc

        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()

        # logging
        self.log_dict(
            {
                "loss_d": loss_d,
                "loss_g": loss_g,
                "loss_adv": loss_adv,
                "loss_fm": loss_fm,
                "loss_mag": loss_mag,
                "loss_sc": loss_sc,
            },
            prog_bar=True,
            on_step=True,
            logger=True,
        )

    #----------------------------------------------------------------------------
    #  configure optimizers                                                     
    #----------------------------------------------------------------------------
    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.gen.parameters(), lr=self.hparams.lr_g, betas=(0.8, 0.99))
        opt_d = torch.optim.AdamW(
            list(self.disc_mpd.parameters()) + list(self.disc_msd.parameters()),
            lr=self.hparams.lr_d,
            betas=(0.8, 0.99),
        )
        return opt_g, opt_d

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from melmodel import (
    RVCStyleVC,
    MelMultiPeriodDiscriminator,
    MelMultiScaleDiscriminator,
)

class MelVCSystem(pl.LightningModule):
    """Full‑length Conformer‑based Mel GAN (Warm‑up L1  →  GAN+FM)."""

    def __init__(
        self,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        lambda_fm: float = 2.0,
        lambda_mel: float = 1.0,
        lambda_adv: float = 1.0,
        sched_gamma: float = 0.5,
        sched_step: int = 200,
        grad_accum: int = 1,
        warmup_epochs: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.grad_accum = max(1, grad_accum)
        self.warmup_epochs = warmup_epochs

        self.gen = RVCStyleVC()
        self.disc_mpd = MelMultiPeriodDiscriminator()
        self.disc_msd = MelMultiScaleDiscriminator()

        self.val_losses = []

    # --------------------------------------------------
    # helper losses
    # --------------------------------------------------
    @staticmethod
    def _adv_d(real, fake):
        return sum(F.mse_loss(r, torch.ones_like(r)) + F.mse_loss(f, torch.zeros_like(f))
                   for r, f in zip(real, fake)) / len(real)

    @staticmethod
    def _adv_g(fake):
        return sum(F.mse_loss(f, torch.ones_like(f)) for f in fake) / len(fake)

    @staticmethod
    def _feat_match(real_feats, fake_feats):
        loss, cnt = 0.0, 0
        for r_lv, f_lv in zip(real_feats, fake_feats):
            for r, f in zip(r_lv, f_lv):
                L = min(r.size(-1), f.size(-1))
                loss += F.l1_loss(f[..., :L], r[..., :L])
                cnt += 1
        return loss / max(cnt, 1)

    # --------------------------------------------------
    # forward helpers
    # --------------------------------------------------
    def forward(self, hubert, pitch, target_length):
        return self.gen(hubert, pitch, target_length=target_length)

    # --------------------------------------------------
    def training_step(self, batch, batch_idx):
        hub, pit, mel_real = batch  # mel_real: (B,T,80)
        opt_g, opt_d = self.optimizers()

        # zero‑grad both
        opt_g.zero_grad(); opt_d.zero_grad()

        # --------------------------------------------------
        # 1) warm‑up : only L1
        # --------------------------------------------------
        if self.current_epoch < self.warmup_epochs:
            mel_fake = self.gen(hub, pit, target_length=mel_real.size(1))
            loss_mel = F.l1_loss(mel_fake, mel_real) / self.grad_accum
            self.manual_backward(loss_mel)
            if (batch_idx + 1) % self.grad_accum == 0:
                opt_g.step(); opt_g.zero_grad()
            self.log_dict({"loss_mel": loss_mel}, prog_bar=True, on_step=True)
            return loss_mel

        # --------------------------------------------------
        # 2) discriminator update
        # --------------------------------------------------
        with torch.no_grad():
            mel_fake_det = self.gen(hub, pit, target_length=mel_real.size(1))  # detach later
        fake_in = mel_fake_det.transpose(1, 2).detach()  # (B,80,T)
        real_in = mel_real.transpose(1, 2)

        fk_mpd, _ = self.disc_mpd(fake_in)
        rl_mpd, _ = self.disc_mpd(real_in)
        fk_msd, _ = self.disc_msd(fake_in)
        rl_msd, _ = self.disc_msd(real_in)

        loss_d = (self._adv_d(rl_mpd, fk_mpd) + self._adv_d(rl_msd, fk_msd)) / self.grad_accum
        self.manual_backward(loss_d)
        if (batch_idx + 1) % self.grad_accum == 0:
            opt_d.step(); opt_d.zero_grad()

        # --------------------------------------------------
        # 3) generator update (fresh forward)
        # --------------------------------------------------
        mel_fake = self.gen(hub, pit, target_length=mel_real.size(1))  # new graph
        fake_in = mel_fake.transpose(1, 2)
        real_in = mel_real.transpose(1, 2)

        fk_mpd, fk_feat_mpd = self.disc_mpd(fake_in)
        fk_msd, fk_feat_msd = self.disc_msd(fake_in)
        _,  rl_feat_mpd     = self.disc_mpd(real_in)
        _,  rl_feat_msd     = self.disc_msd(real_in)

        loss_adv = (self._adv_g(fk_mpd) + self._adv_g(fk_msd)) / self.grad_accum
        loss_fm  = self._feat_match(rl_feat_mpd, fk_feat_mpd) + self._feat_match(rl_feat_msd, fk_feat_msd)
        loss_mel = F.l1_loss(mel_fake, mel_real) / self.grad_accum

        loss_g = (self.hparams.lambda_mel * loss_mel +
                  self.hparams.lambda_fm  * loss_fm  +
                  self.hparams.lambda_adv * loss_adv)

        self.manual_backward(loss_g)
        if (batch_idx + 1) % self.grad_accum == 0:
            opt_g.step(); opt_g.zero_grad()

        self.log_dict({
            "loss_d": loss_d,
            "loss_adv": loss_adv,
            "loss_fm": loss_fm,
            "loss_mel": loss_mel,
            "loss_g": loss_g,
        }, prog_bar=True, on_step=True)
        return loss_g

    # --------------------------------------------------
    @torch.no_grad()
    def validation_step(self, batch, _):
        hub, pit, mel_real = batch
        mel_fake = self.gen(hub, pit, target_length=mel_real.size(1))
        val_loss_mel = F.l1_loss(mel_fake, mel_real)
        #self.log("val_loss_mell", val_loss_mell, prog_bar=True)
        self.val_losses.append(val_loss_mel.detach())

    # --------------------------------------------------
    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.gen.parameters(), lr=self.hparams.lr_g, betas=(0.8, 0.99))
        opt_d = torch.optim.AdamW([
            *self.disc_mpd.parameters(), *self.disc_msd.parameters()
        ], lr=self.hparams.lr_d, betas=(0.8, 0.99))
        sch_g = torch.optim.lr_scheduler.StepLR(opt_g, self.hparams.sched_step, gamma=self.hparams.sched_gamma)
        sch_d = torch.optim.lr_scheduler.StepLR(opt_d, self.hparams.sched_step, gamma=self.hparams.sched_gamma)
        return ([opt_g, opt_d], [
            {"scheduler": sch_g, "interval": "step"},
            {"scheduler": sch_d, "interval": "step"},
        ])

    # --------------------------------------------------
    @torch.no_grad()
    def predict_step(self, batch, *_):
        hubert, pitch, _ = batch
        mel = self.gen(hubert, pitch, target_length=None)  # length auto by upsample_factor
        return mel

    def on_validation_epoch_end(self):
        if not self.val_losses:
            return
                
        avg_loss = torch.stack(self.val_losses).mean()

        self.log(
            "val_loss_mel",
            avg_loss,
            prog_bar=True,    # プログレスバーにも出す
            on_step=False,    # バッチごとでは出さない
            on_epoch=True     # エポック平均を出す
        )
        self.val_losses.clear()
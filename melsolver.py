import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model import RVCStyleVC, MultiPeriodDiscriminator, MultiScaleDiscriminator

class MelVCSystem(pl.LightningModule):
    """Conformer‑based Mel GAN: 2段階学習（ウォームアップL1→GAN+FM）."""

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
        warmup_epochs: int = 10,    # ウォームアップ期間（エポック数）
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.grad_accum = max(1, grad_accum)
        self.warmup_epochs = warmup_epochs

        self.gen = RVCStyleVC()
        self.disc_mpd = MultiPeriodDiscriminator()
        self.disc_msd = MultiScaleDiscriminator()

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

    def forward(self, hubert, pitch, target_length):
        return self.gen(hubert, pitch, target_length=target_length)

    def training_step(self, batch, batch_idx):
        hub, pit, mel_real = batch
        opt_g, opt_d = self.optimizers()
        if batch_idx % self.grad_accum == 0:
            opt_g.zero_grad(); opt_d.zero_grad()

        mel_fake = self.gen(hub, pit, target_length=mel_real.size(1))
        cut_len = min(mel_real.size(1), mel_fake.size(1))
        mel_real_c = mel_real[:, :cut_len]
        mel_fake_c = mel_fake[:, :cut_len]

        # ---------- Phase 1: Warmup (L1 のみ) ----------
        if self.current_epoch < self.warmup_epochs:
            loss_mel = F.l1_loss(mel_fake_c, mel_real_c) / self.grad_accum
            self.manual_backward(loss_mel)
            if (batch_idx + 1) % self.grad_accum == 0:
                opt_g.step()
            self.log_dict({
                "phase": 0,
                "loss_g": loss_mel,
                "loss_mel": loss_mel,
            }, prog_bar=True, on_step=True)
            return

        # ---------- Phase 2: GAN+FM+L1 ----------
        # --------- Discriminator update ---------
        fk_mpd, fk_feat_mpd = self.disc_mpd(mel_fake_c.transpose(1,2).detach())
        fk_msd, fk_feat_msd = self.disc_msd(mel_fake_c.transpose(1,2).detach())
        rl_mpd, rl_feat_mpd = self.disc_mpd(mel_real_c.transpose(1,2))
        rl_msd, rl_feat_msd = self.disc_msd(mel_real_c.transpose(1,2))

        loss_d = (self._adv_d(rl_mpd, fk_mpd) + self._adv_d(rl_msd, fk_msd)) / self.grad_accum
        self.manual_backward(loss_d)
        if (batch_idx + 1) % self.grad_accum == 0:
            opt_d.step()

        # --------- Generator adversarial + FM ---------
        fk_mpd, fk_feat_mpd = self.disc_mpd(mel_fake_c.transpose(1,2))
        fk_msd, fk_feat_msd = self.disc_msd(mel_fake_c.transpose(1,2))
        loss_adv = self._adv_g(fk_mpd) + self._adv_g(fk_msd)
        loss_fm = self._feat_match(rl_feat_mpd, fk_feat_mpd) + self._feat_match(rl_feat_msd, fk_feat_msd)
        loss_mel = F.l1_loss(mel_fake_c, mel_real_c) / self.grad_accum

        loss_g = (
            self.hparams.lambda_mel * loss_mel +
            self.hparams.lambda_fm * loss_fm +
            self.hparams.lambda_adv * loss_adv
        )
        self.manual_backward(loss_g)
        if (batch_idx + 1) % self.grad_accum == 0:
            opt_g.step()

        # ---- ログ ----
        if (batch_idx + 1) % self.grad_accum == 0:
            self.log_dict({
                "phase": 1,
                "loss_g": loss_g,
                "loss_d": loss_d,
                "loss_mel": loss_mel,
                "loss_adv": loss_adv,
                "loss_fm": loss_fm,
            }, prog_bar=True, on_step=True)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        hub, pit, mel_real = batch
        mel_fake = self.gen(hub, pit, target_length=mel_real.size(1))
        cut_len = min(mel_real.size(1), mel_fake.size(1))
        mel_real_c = mel_real[:, :cut_len]
        mel_fake_c = mel_fake[:, :cut_len]
        val_mel = F.l1_loss(mel_fake_c, mel_real_c)
        self.log("val_mel", val_mel, prog_bar=True)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.gen.parameters(), lr=self.hparams.lr_g, betas=(0.8, 0.99))
        opt_d = torch.optim.AdamW(list(self.disc_mpd.parameters()) + list(self.disc_msd.parameters()), lr=self.hparams.lr_d, betas=(0.8, 0.99))
        sched_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=self.hparams.sched_step, gamma=self.hparams.sched_gamma)
        sched_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=self.hparams.sched_step, gamma=self.hparams.sched_gamma)
        return ([opt_g, opt_d], [
            {"scheduler": sched_g, "interval": "step"},
            {"scheduler": sched_d, "interval": "step"},
        ])
    
    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
      hubert, pitch, _ = batch  # mel は不要
      z = self.gen.encoder(hubert, pitch)  # (B, C, T)


      mel = self.gen.generator(z)  # 明示補間推論
      return mel

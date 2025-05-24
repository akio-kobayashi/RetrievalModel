from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model import RVCStyleVC, MultiPeriodDiscriminator, MultiScaleDiscriminator
from stft_loss import MultiResolutionSTFTLoss
from transformers import get_cosine_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
from pitch_loss import PitchLoss

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
        lambda_cmp: float = 1.0,
        lambda_wav: float = 0.1,
        lambda_pitch: float = 1.0,
        sched_gamma: float = 0.5,
        sched_step: int = 200,
        grad_accum: int = 1,
        warmup_steps: int = 5_000,
        mse_steps: int = 2_000,
        adv_scale: float = 1.0,
        max_norm: float = 5.0,
        upsample_rates = [5, 2, 4, 2, 2],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.grad_accum = max(1, grad_accum)
        self.warmup_steps = warmup_steps
        self.mse_steps = mse_steps
        self.adv_scale = adv_scale
        self.max_norm = max_norm

        self.gen = RVCStyleVC(upsample_rates=upsample_rates)
        self.disc_mpd = MultiPeriodDiscriminator()
        self.disc_msd = MultiScaleDiscriminator()
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes  = [2048, 1024, 512, 256], #[2048, 1024, 512, 256],
            hop_sizes  = [512, 256, 128, 64],   #[512, 256, 128,  64],
            win_lengths = [1200, 600, 240, 120],
            #win_lengths= [2048, 1024, 512, 256], #[1200, 600, 240, 120],
            window     = "hann_window",
            factor_sc  = self.hparams.lambda_sc,   # λ_sc
            factor_mag = self.hparams.lambda_mag,  # λ_mag
            factor_cmp = self.hparams.lambda_cmp            
        )
        self.pitch_loss = PitchLoss()
        # 重み更新差分キャッシュは on_train_start で初期化
        self._prev_w: dict[str, torch.Tensor] = {}

    def on_train_start(self) -> None:
        # モデルがすでに適切な device に載った後に呼ばれます
        self._prev_w = {
            name: param.clone().detach()
            for name, param in self.gen.named_parameters()
        }

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
      step = int(self.global_step)
      hub, pit, wav_real = batch
      opt_g, opt_d = self.optimizers()
      if batch_idx % self.grad_accum == 0:
          opt_g.zero_grad(); opt_d.zero_grad()

      # ------------- Generator forward -------------
      wav_fake = self.gen(hub, pit)
      hop = self.hparams.hop
      num_frames = hub.size(1)
      cut_len = num_frames * hop
      cut_len = min(cut_len, wav_real.size(-1), wav_fake.size(-1))

      wav_real_c = wav_real[..., :cut_len]
      wav_fake_c = wav_fake[..., :cut_len]

      # STAGE-0: MSEのみ step < mse_steps
      if step < self.mse_steps:
          # 1) Compute Smooth-L1 loss
          loss_mse = F.l1_loss(wav_fake_c, wav_real_c)
          # 2) Backward
          self.manual_backward(loss_mse)

          # 3) Log raw gradient norm before clipping
          total_norm_g = torch.nn.utils.clip_grad_norm_(
              self.gen.parameters(), float('inf')
          )
          self.log("grad_norm/g", total_norm_g, on_step=True, prog_bar=False)

          # 4) Gradient accumulation step
          if (batch_idx + 1) % self.grad_accum == 0:
              # a) Optimizer step
              torch.nn.utils.clip_grad_norm_(self.gen.parameters(), self.max_norm)
              opt_g.step()

              # c) Compute and log parameter update magnitude & ratios
              name, p = next(self.gen.named_parameters())
              delta = (p - self._prev_w[name].to(p.device)).abs().mean()
              # d) Compute weight norm and update ratio
              weight_norm = p.abs().mean().clamp_min(1e-8)
              update_ratio = delta / weight_norm

              # e) Cache current weight for next step
              self._prev_w[name] = p.clone().detach()

              # f) Log update stats
              self.log("param_update_mean", delta, on_step=True, prog_bar=False)
              self.log("weight_norm/g",     weight_norm,   on_step=True, prog_bar=False)
              self.log("update_ratio/g",    update_ratio,  on_step=True, prog_bar=False)

          # 5) Log phase and scaled loss
          self.log("phase",    0.0,        on_step=True)
          self.log("loss_mse", loss_mse * self.grad_accum, on_step=True)
          self.log("loss_mse_epoch", loss_mse, on_step=False, on_epoch=True)
          return

      loss_wav = F.l1_loss(wav_fake_c, wav_real_c)
      loss_wav /= self.grad_accum
      loss_sc, loss_mag, loss_cmp = self.stft_loss(wav_fake_c, wav_real_c)
      loss_mag /= self.grad_accum       # 
      loss_sc  /= self.grad_accum       # 
      loss_cmp /= self.grad_accum
      loss_pitch = self.pitch_loss(wav_fake_c, wav_real_c)
      loss_pitch /= self.grad_accum
      self.log("loss_mag_epoch", loss_mag, on_step=False, on_epoch=True)

      # ======================================================
      #  STAGE-1 : STFT のみ  (step < warmup_steps)
      # ======================================================
      if step < self.warmup_steps:
          # ---- G update (STFTのみ) ----
          loss_g_total = (loss_mag + loss_sc)          
          loss_g = loss_g_total / self.grad_accum
        
          self.manual_backward(loss_g)
          total_norm_g = torch.nn.utils.clip_grad_norm_(self.gen.parameters(), float('inf'))
          self.log("grad_norm/g", total_norm_g, on_step=True, prog_bar=False)          
          if (batch_idx + 1) % self.grad_accum == 0:
              torch.nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=self.max_norm)            
              opt_g.step();

          # D は更新しない
          if (batch_idx + 1) % self.grad_accum == 0:
              self.log_dict({
                  "stage": 0.,
                  "loss_g": loss_g_total,
                  "loss_mag": loss_mag,
                  "loss_sc": loss_sc,
                  "loss_cmp": loss_cmp,
                  "loss_wav": loss_wav,
                  "loss_pitch": loss_pitch,
              }, prog_bar=True, on_step=True)
              self.log("loss_mag_epoch", loss_mag, on_step=False, on_epoch=True)
          return  {"loss_mag": loss_mag} # ← ここで終了
      # ======================================================
      #  STAGE-2 : GAN + FM + STFT
      # ======================================================

      # ---- Discriminator update (同じ処理) ----
      wav_fake_det = wav_fake.detach()
      wav_fake_c_det = wav_fake_det[..., :cut_len]
      rl_mpd, _ = self.disc_mpd(wav_real_c.unsqueeze(1))
      rl_msd, _ = self.disc_msd(wav_real_c.unsqueeze(1))
      fk_mpd, _ = self.disc_mpd(wav_fake_c_det.unsqueeze(1))
      fk_msd, _ = self.disc_msd(wav_fake_c_det.unsqueeze(1))
      loss_d = (self._adv_d(rl_mpd, fk_mpd) + self._adv_d(rl_msd, fk_msd)) / self.grad_accum
      self.manual_backward(loss_d)      
      total_norm_d = torch.nn.utils.clip_grad_norm_(
          list(self.disc_mpd.parameters()) + list(self.disc_msd.parameters()),
          float('inf'),
      )
      self.log("grad_norm/d", total_norm_d, on_step=True, prog_bar=False)
      if (batch_idx + 1) % self.grad_accum == 0:
          torch.nn.utils.clip_grad_norm_(self.disc_mpd.parameters(), self.max_norm)
          torch.nn.utils.clip_grad_norm_(self.disc_msd.parameters(), self.max_norm)
          opt_d.step();
          #opt_d.zero_grad()

      # ---- Generator adversarial + FM ----
      fk_mpd, fk_feat_mpd = self.disc_mpd(wav_fake_c.unsqueeze(1))
      fk_msd, fk_feat_msd = self.disc_msd(wav_fake_c.unsqueeze(1))
      loss_adv = self._adv_g(fk_mpd) + self._adv_g(fk_msd)
      _, rl_feat_mpd = self.disc_mpd(wav_real_c.unsqueeze(1).detach())
      _, rl_feat_msd = self.disc_msd(wav_real_c.unsqueeze(1).detach())
      loss_fm = self._feat_match(rl_feat_mpd, fk_feat_mpd) + self._feat_match(rl_feat_msd, fk_feat_msd)

      # ---- 最終 G 損失 ----
      loss_g_total = (self.adv_scale * loss_adv +
                      self.hparams.lambda_fm * loss_fm +
                      loss_mag + loss_sc +
                      self.hparams.lambda_wav * loss_wav)
      loss_g = loss_g_total / self.grad_accum  
      self.manual_backward(loss_g)
      total_norm_g = torch.nn.utils.clip_grad_norm_(self.gen.parameters(), float('inf'))
      self.log("grad_norm/g", total_norm_g, on_step=True, prog_bar=False)  
      if (batch_idx + 1) % self.grad_accum == 0:
          torch.nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=self.max_norm)            
          opt_g.step();
          #opt_g.zero_grad()

      if (batch_idx + 1) % self.grad_accum == 0:
          self.log_dict({
             "stage": 1.,
             "loss_d": loss_d * self.grad_accum,
             "loss_g": loss_g_total,
             "loss_adv": loss_adv,
             "loss_fm": loss_fm,
             "loss_mag": loss_mag,
             "loss_sc": loss_sc,
             "loss_cmp": loss_cmp,
             "loss_wav": loss_wav,
             "loss_pitch": loss_pitch,
          }, prog_bar=True, on_step=True)

    def on_training_epoch_end(self, outputs):
        mag_losses = [x["loss_mag"] for x in outputs]
        avg_loss_mag = torch.stack(mag_losses).mean()
        self.log("loss_mag_epoch", avg_loss_mag, prog_bar=True, on_epoch=True)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        #hub, pit, wav_real = batch
        #wav_fake = self.gen(hub, pit)
        #cut_len = min(wav_real.size(-1), wav_fake.size(-1))
        #wav_real = wav_real[..., :cut_len]
        #wav_fake = wav_fake[..., :cut_len]
        #sc, mag, cmp_loss = self.stft_loss(wav_fake, wav_real)
        #loss_pitch = self.pitch_loss(wav_fake, wav_real)
        #self.log_dict({"val_mag": mag, "val_sc": sc, "val_comp": cmp_loss, "val_pitch": loss_pitch}, prog_bar=True)
        pass
    
    # ---------------- optimizers & schedulers ----------------
    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.gen.parameters(), lr=self.hparams.lr_g, betas=(0.8, 0.99))
        opt_d = torch.optim.AdamW(list(self.disc_mpd.parameters()) + list(self.disc_msd.parameters()), lr=self.hparams.lr_d, betas=(0.8, 0.99))

        # w/o GAN training steps
        total_steps  = self.trainer.estimated_stepping_batches
        warmup_steps = self.warmup_steps        
        #sched_g = get_cosine_schedule_with_warmup(opt_g, num_warmup_steps=warmup_steps, num_training_steps = total_steps,num_cycles= 0.5)
        sched_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=self.hparams.sched_step, gamma=self.hparams.sched_gamma)
        sched_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=self.hparams.sched_step, gamma=self.hparams.sched_gamma)
        return ([opt_g, opt_d], [
            {"scheduler": sched_g, "interval": "step"},
            {"scheduler": sched_d, "interval": "step"},
        ])

# ============================================================
#  Fine‑tune variant with Soft‑DTW alignment                  
# ============================================================
class FineTuneVC(VCSystem):
    """VCSystem with Soft‑DTW loss to align sequences of unequal length."""

    def __init__(self, ckpt_path: str, dtw_gamma: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        # load base weights
        state = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state:
            self.load_state_dict(state["state_dict"], strict=False)
        else:
            self.load_state_dict(state, strict=False)
        self.soft_dtw = _SoftDTW(gamma=dtw_gamma, normalize=True)

    # override loss combination
    def training_step(self, batch, batch_idx):
        hub, pit, wav_real = batch
        opt_g, opt_d = self.optimizers()

        wav_fake_full = self.gen(hub, pit).detach()
        cut_len = min(wav_real.size(-1), wav_fake_full.size(-1))
        wav_real_c = wav_real[..., :cut_len]
        wav_fake_c = wav_fake_full[..., :cut_len]
        loss_wav = F.l1_loss(wav_fake_c, wav_real_c)

        # ---- D update ----
        opt_d.zero_grad()
        rl_mpd, _ = self.disc_mpd(wav_real_c.unsqueeze(1))
        rl_msd, _ = self.disc_msd(wav_real_c.unsqueeze(1))
        fk_mpd, _ = self.disc_mpd(wav_fake_c.unsqueeze(1))
        fk_msd, _ = self.disc_msd(wav_fake_c.unsqueeze(1))
        loss_d = self._adv_d(rl_mpd, fk_mpd) + self._adv_d(rl_msd, fk_msd)
        self.manual_backward(loss_d)
        torch.nn.utils.clip_grad_norm_(self.disc_mpd.parameters(), self.max_norm)
        torch.nn.utils.clip_grad_norm_(self.disc_msd.parameters(), self.max_norm)
        opt_d.step()

        # ---- G update ----
        wav_fake_full = self.gen(hub, pit)
        wav_fake_c = wav_fake_full[..., :cut_len]
        fk_mpd, fk_feat_mpd = self.disc_mpd(wav_fake_c.unsqueeze(1))
        fk_msd, fk_feat_msd = self.disc_msd(wav_fake_c.unsqueeze(1))
        loss_adv = self._adv_g(fk_mpd) + self._adv_g(fk_msd)

        _, rl_feat_mpd = self.disc_mpd(wav_real_c.unsqueeze(1).detach())
        _, rl_feat_msd = self.disc_msd(wav_real_c.unsqueeze(1).detach())
        loss_fm = self._feat_match(rl_feat_mpd, fk_feat_mpd) + self._feat_match(rl_feat_msd, fk_feat_msd)
        loss_sc, loss_mag, loss_cmp = self.stft_loss(wav_fake_c, wav_real_c)
        loss_pitch = self.pitch_loss(wav_fake_c, wav_real_c)
        loss_dtw = self.soft_dtw(wav_real, wav_fake_full)  # align full‑length sequences

        loss_g = (loss_adv + self.hparams.lambda_fm * loss_fm +
                   loss_mag + loss_sc + loss_cmp + loss_dtw + 
                   self.hparams.lambda_wav * loss_wav + self.hparams.lambda_pitch * loss_pitch)

        opt_g.zero_grad()
        self.manual_backward(loss_g)
        torch.nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=self.max_norm)            
        opt_g.step()

        self.log_dict({
            "loss_d": loss_d,
            "loss_g": loss_g,
            "loss_dtw": loss_dtw,
            "loss_adv": loss_adv,
            "loss_fm": loss_fm,
            "loss_mag": loss_mag,
            "loss_sc": loss_sc,
            "loss_wav": loss_wav,
        }, prog_bar=True, on_step=True)

# ------------------------------------------------------------
#  Simple Soft‑DTW implementation (batched)                    
# ------------------------------------------------------------
class _SoftDTW(nn.Module):
    def __init__(self, gamma: float = 0.1, normalize: bool = True):
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """x,y: (B,N) vs (B,M) sequences."""
        B, N = x.shape
        M = y.size(1)
        D = torch.cdist(x.unsqueeze(-1), y.unsqueeze(-1), p=1).squeeze(-1)  # (B,N,M)
        R = torch.zeros(B, N + 2, M + 2, device=x.device) + 1e6
        R[:, 0, 0] = 0
        for i in range(1, N + 1):
            d = D[:, i - 1]
            for j in range(1, M + 1):
                r0 = -R[:, i - 1, j - 1] / self.gamma
                r1 = -R[:, i - 1, j] / self.gamma
                r2 = -R[:, i, j - 1] / self.gamma
                rmax = torch.max(torch.stack([r0, r1, r2], dim=-1), dim=-1).values
                rsum = torch.exp(r0 - rmax) + torch.exp(r1 - rmax) + torch.exp(r2 - rmax)
                softmin = -self.gamma * (torch.log(rsum + 1.0e-9) + rmax)
                R[:, i, j] = d[:, j - 1] + softmin
        loss = R[:, N, M]
        if self.normalize:
            loss = loss / (N + M)
        return loss.mean()

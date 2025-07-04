import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW

from align_model import TransformerAligner
from melmodel import RVCStyleVC
from melsolver import MelVCSystem

def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True
        
class AlignmentRVCSystem(pl.LightningModule):
    def __init__(
        self,
        # ckpt
        align_ckpt: str,
        mel_ckpt: str,
        # optimizer
        lr: float = 2e-4,
        # Alignment model
        input_dim_hubert: int = 768,
        input_dim_pitch: int = 1,
        align_d_model: int = 256,
        align_nhead: int = 4,
        align_num_layers: int = 3,
        align_dim_ff: int = 512,
        align_dropout: float = 0.1,
        diag_w: float = 1.0,
        ce_w: float = 1.0,
        # RVC model
        latent_ch: int = 256,
        rvc_d_model: int = 256,
        rvc_n_conformer: int = 8,
        rvc_nhead: int = 8,
        # loss weights
        align_weight: float = 1.0,
        mel_weight: float = 1.0,
        update_aligner: bool = True,
        update_rvc: bool = True,
        load_from_pretrained = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 1) Alignment model（LoRAなし，学習済みをそのまま読み込み）
        self.aligner = TransformerAligner(
            input_dim_hubert=input_dim_hubert,
            input_dim_pitch=input_dim_pitch,
            d_model=align_d_model,
            nhead=align_nhead,
            num_layers=align_num_layers,
            dim_feedforward=align_dim_ff,
            dropout=align_dropout,
            diag_weight=diag_w,
            ce_weight=ce_w
        )
        if load_from_pretrained:
            ckpt = torch.load(align_ckpt, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            state_dict = {
                k.removeprefix("model.") if k.startswith("model.") else k: v
                for k, v in state_dict.items()
            }
            self.aligner.load_state_dict(state_dict, strict=False)

        # 2) 通常のRVCモデル（LoRAなし）
        mel_model = MelVCSystem(
            lr_g=2e-4,
            lr_d=2e-4,
            lambda_fm=2.0,
            lambda_mel=1.0,
            lambda_adv=1.0,
            sched_gamma=0.5,
            sched_step=200,
            grad_accum=1,
            warmup_epochs=10
        )

        # ---------- 2. checkpoint を読み込み ----------
        if load_from_pretrained:
            ckpt = torch.load(mel_ckpt, map_location="cpu")
      
            self.rvc = RVCStyleVC(
                latent_ch=latent_ch,
                d_model=rvc_d_model,
                n_conformer=rvc_n_conformer,
                nhead=rvc_nhead
            )
            self.rvc.load_state_dict(torch.load(mel_ckpt, map_location="cpu"))
        
        self.train_losses = []
        self.val_losses = []

        if load_from_pretrained:
            if not update_aligner:
                freeze_module(self.aligner)
            else:
                unfreeze_module(self.aligner)

            if not update_rvc:
                freeze_module(self.rvc)
            else:
                unfreeze_module(self.rvc)

    def forward(self, src_hubert, src_pitch, max_len: int, mel_target_len: int):
        pred_hubert, pred_pitch = self.aligner.greedy_decode(src_hubert, src_pitch, max_len)
        mel_pred = self.rvc(pred_hubert, pred_pitch, target_length=mel_target_len)
        return mel_pred

    def training_step(self, batch, batch_idx):
        src_h, src_p, tgt_h, tgt_p, mel_tgt = batch

        # (1) teacher-forcing 整列損失
        loss_tf, metrics = self.aligner(src_h, src_p, tgt_h, tgt_p)
        loss_align = loss_tf

        # (2) free-run 整列 L1 を 10 バッチに1回だけ追加
        if batch_idx % self.free_run_interval == 0 and self.free_run_weight > 0:
            with torch.no_grad():
                pred_h, pred_p = self.aligner.greedy_decode(
                    src_h[:1], src_p[:1], max_len=tgt_h.size(1)
                )
            T = tgt_h.size(1)
            # pad or trunc
            if pred_h.size(1) < T:
                pad = T - pred_h.size(1)
                pred_h = torch.cat([pred_h, torch.zeros(1, pad, pred_h.size(2), device=self.device)], dim=1)
                pred_p = torch.cat([pred_p, torch.zeros(1, pad, device=self.device)], dim=1)
            else:
                pred_h = pred_h[:, :T]
                pred_p = pred_p[:, :T]

            # free-run L1 損失（先頭サンプルのみ）
            loss_fr_h = F.l1_loss(pred_h, tgt_h[:1])
            loss_fr_p = F.l1_loss(pred_p, tgt_p[:1])
            loss_fr   = loss_fr_h + loss_fr_p

            # teacher-forcing + free-run
            loss_align = loss_tf + self.free_run_weight * loss_fr
            metrics['fr_l1_h'] = loss_fr_h
            metrics['fr_l1_p'] = loss_fr_p

        # (3) RVC 側の L1 損失
        pred_hubert = self.aligner.last_preds['hubert_pred']
        pred_pitch  = self.aligner.last_preds['pitch_pred']
        mel_pred = self.rvc(pred_hubert, pred_pitch, target_length=mel_tgt.size(1))
        loss_mel = F.l1_loss(mel_pred, mel_tgt)

        # (4) 合計損失
        total_loss = self.hparams.align_weight * loss_align \
                   + self.hparams.mel_weight   * loss_mel

        # ログ
        self.log_dict(metrics,        on_step=True, on_epoch=True)
        self.log("loss_align", loss_align, prog_bar=False)
        self.log("loss_mel",   loss_mel,   prog_bar=False)
        self.log("loss_total", total_loss, on_step=True, prog_bar=True)

        self.train_losses.append(total_loss.detach())
        return total_loss

    def validation_step(self, batch, batch_idx):
        src_h, src_p, tgt_h, tgt_p, mel_tgt = batch

        # teacher-forcing 整列損失
        loss_align, _ = self.aligner(src_h, src_p, tgt_h, tgt_p)

        # RVC 側の L1 損失
        pred_hubert = self.aligner.last_preds['hubert_pred']
        pred_pitch  = self.aligner.last_preds['pitch_pred']
        mel_pred = self.rvc(pred_hubert, pred_pitch, target_length=mel_tgt.size(1))
        loss_mel = F.l1_loss(mel_pred, mel_tgt)

        # バリデーション用ログ
        self.log("val_loss_mel", loss_mel, on_epoch=True, prog_bar=False)

        # epoch 終了時に平均を出すための蓄積
        self.val_losses.append(
            self.hparams.align_weight * loss_align
          + self.hparams.mel_weight   * loss_mel
        )
    
'''
    def training_step(self, batch, batch_idx):
        src_h, src_p, tgt_h, tgt_p, mel_tgt = batch
        loss_align, metrics = self.aligner(src_h, src_p, tgt_h, tgt_p)
        pred_hubert, pred_pitch = self.aligner.last_preds['hubert_pred'], self.aligner.last_preds['pitch_pred']
        #self.aligner.greedy_decode(src_h, src_p)
        mel_pred = self.rvc(pred_hubert, pred_pitch, target_length=mel_tgt.size(1))
        loss_mel = F.l1_loss(mel_pred, mel_tgt)
        total_loss = self.hparams.align_weight * loss_align + self.hparams.mel_weight * loss_mel
        self.log_dict(metrics, on_step=True, on_epoch=True)
        self.log("loss_align", loss_align, prog_bar=False)
        self.log("loss_mel", loss_mel, prog_bar=False)
        self.log("loss_total", total_loss, on_step=True, prog_bar=True)
        self.train_losses.append(total_loss.detach())
        return total_loss

    def validation_step(self, batch, batch_idx):
        src_h, src_p, tgt_h, tgt_p, mel_tgt = batch
        loss_align, _ = self.aligner(src_h, src_p, tgt_h, tgt_p)
        pred_hubert, pred_pitch = self.aligner.last_preds['hubert_pred'], self.aligner.last_preds['pitch_pred']       
        #self.aligner.greedy_decode(src_h, src_p)
        mel_pred = self.rvc(pred_hubert, pred_pitch, target_length=mel_tgt.size(1))
        loss_mel = F.l1_loss(mel_pred, mel_tgt)
        self.log("val_loss_mel", loss_mel)
        self.val_losses.append(
            self.hparams.align_weight * loss_align + self.hparams.mel_weight * loss_mel
        )
'''
    def on_train_epoch_end(self):
        avg = torch.stack(self.train_losses).mean()
        self.log("train_loss_epoch", avg, prog_bar=True)
        self.train_losses.clear()

    def on_validation_epoch_end(self):
        if not self.val_losses:
            return
        avg = torch.stack(self.val_losses).mean()
        self.log("val_loss", avg, prog_bar=True)
        self.val_losses.clear()

    def configure_optimizers(self):
        optimizer = AdamW(
          filter(lambda p: p.requires_grad, self.parameters()),
          lr = self.hparams.lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer,
          T_max = self.trainer.max_epochs,
          eta_min = self.hparams.lr/10.0
        )
        return {
          "optimizer": optimizer,
          "scheduler": scheduler,
          "gradient_clip_val": 1.0,
        }

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW

from align_model import TransformerAligner
from rvc_model import RVCStyleVC

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
        mel_weight: float = 1.0
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
        ckpt = torch.load(align_ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = {
            k.removeprefix("model.") if k.startswith("model.") else k: v
            for k, v in state_dict.items()
        }
        self.aligner.load_state_dict(state_dict, strict=True)


        # 2) 通常のRVCモデル（LoRAなし）
        self.rvc = RVCStyleVC(ckpt_path=mel_ckpt,
                              latent_ch = latent_ch,
                              d_model = rvc_d_model,
                              n_conformer = rvc_n_conformer,
                              rvc_nhead = rvc_head
                             )

        self.train_losses = []
        self.val_losses = []

    def forward(self, src_hubert, src_pitch, max_len: int, mel_target_len: int):
        pred_hubert, pred_pitch = self.aligner.greedy_decode(src_hubert, src_pitch, max_len)
        mel_pred = self.rvc(pred_hubert, pred_pitch, target_length=mel_target_len)
        return mel_pred

    def training_step(self, batch, batch_idx):
        src_h, src_p, tgt_h, tgt_p, mel_tgt = batch
        loss_align, metrics = self.aligner(src_h, src_p, tgt_h, tgt_p)
        pred_hubert, pred_pitch = self.aligner.greedy_decode(src_h, src_p)
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
        pred_hubert, pred_pitch = self.aligner.greedy_decode(src_h, src_p)
        mel_pred = self.rvc(pred_hubert, pred_pitch, target_length=mel_tgt.size(1))
        loss_mel = F.l1_loss(mel_pred, mel_tgt)
        self.log("val_loss_mel", loss_mel)
        self.val_losses.append(
            self.hparams.align_weight * loss_align + self.hparams.mel_weight * loss_mel
        )

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
          filter(lambda p: p.requrires_grad, self.prameters()),
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

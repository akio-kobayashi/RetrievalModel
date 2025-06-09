import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW

from align_model import TransformerAligner
from align_lora import apply_lora_to_transformeraligner
from mellora import RVCStyleVC_LoRA

class AlignmentRVCSystem(pl.LightningModule):
    def __init__(
        self,
        # ckpt
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
        # LoRA
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        # loss weights
        align_weight: float = 1.0,
        mel_weight: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()

        # 1) Base Alignment model
        base_align = TransformerAligner(
            input_dim_hubert=self.hparams.input_dim_hubert,
            input_dim_pitch=self.hparams.input_dim_pitch,
            d_model=self.hparams.align_d_model,
            nhead=self.hparams.align_nhead,
            num_layers=self.hparams.align_num_layers,
            dim_feedforward=self.hparams.align_dim_ff,
            dropout=self.hparams.align_dropout,
            diag_weight=self.hparams.diag_w,
            ce_weight=self.hparams.ce_w
        )
        # --- load pretrained weights for aligner (LoRA-less) ---
        ckpt_align = torch.load(self.hparams.align_ckpt, map_location="cpu")
        base_align.load_state_dict(ckpt_align, strict=True)

        # wrap with LoRA
        self.aligner = apply_lora_to_transformeraligner(
            base_align,
            rank=self.hparams.lora_rank,
            alpha=self.hparams.lora_alpha
        )
        
        # freeze parameters
        for name, p in self.aligner.named_parameters():
          if not ("A" in name or "B" in name):
            p.requires_grad = False

        # 2) LoRA-wrapped RVC model
        self.rvc = RVCStyleVC_LoRA(
            latent_ch=self.hparams.latent_ch,
            d_model=self.hparams.rvc_d_model,
            n_conformer=self.hparams.rvc_n_conformer,
            nhead=self.hparams.rvc_nhead,
            rank=self.hparams.lora_rank,
            alpha=self.hparams.lora_alpha
        )
        # --- load pretrained weights for RVC (LoRA-less) ---
        ckpt_rvc = torch.load(self.hparams.mel_ckpt, map_location="cpu")
        self.rvc.load_state_dict(ckpt_rvc, strict=False)

        # prepare loss storage
        self.train_losses = []
        self.val_losses = []

    def forward(self, src_hubert, src_pitch, max_len: int, mel_target_len: int):
        # 1) alignment → HuBERT & pitch
        pred_hubert, pred_pitch = self.aligner.greedy_decode(
            src_hubert, src_pitch, max_len
        )
        # 2) RVC generator → mel
        mel_pred = self.rvc(pred_hubert, pred_pitch, target_length=mel_target_len)
        return mel_pred

    def training_step(self, batch, batch_idx):
        src_h, src_p, tgt_h, tgt_p, mel_tgt = batch

        # alignment loss
        loss_align, metrics = self.aligner(src_h, src_p, tgt_h, tgt_p)

        # decode and mel-predict
        pred_hubert, pred_pitch = self.aligner.greedy_decode(src_h, src_p)
        mel_pred = self.rvc(pred_hubert, pred_pitch, target_length=mel_tgt.size(1))

        # mel reconstruction loss
        loss_mel = F.l1_loss(mel_pred, mel_tgt)

        # combined
        total_loss = (
            self.hparams.align_weight * loss_align
            + self.hparams.mel_weight * loss_mel
        )

        # logging
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
        # only LoRA params
        trainable = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = AdamW(trainable, lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr / 10.0
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

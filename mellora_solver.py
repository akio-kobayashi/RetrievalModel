import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from mellora import RVCStyleVC_LoRA

class RVCStyleSystem(pl.LightningModule):
    def __init__(
        self,
        lr: float = 2e-4,
        latent_ch: int = 256,
        d_model: int = 256,
        n_conformer: int = 8,
        nhead: int = 8,
        lora_rank: int = 4,
        lora_alpha: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()

        # ① LoRA付き RVC モデルを構築
        self.model = RVCStyleVC_LoRA(
            latent_ch=self.hparams.latent_ch,
            d_model=self.hparams.d_model,
            n_conformer=self.hparams.n_conformer,
            nhead=self.hparams.nhead,
            rank=self.hparams.lora_rank,
            alpha=self.hparams.lora_alpha
        )

        # --- load pretrained weights for RVC (LoRA-less) ---
        ckpt_rvc = torch.load("rvc_pretrained.ckpt", map_location="cpu")
        self.rvc.load_state_dict(ckpt_rvc, strict=False)

        # ② ロス記録用
        self.train_losses = []
        self.val_losses = []

    def forward(self, hubert, pitch, target_length=None, mask=None):
        # hubert: (B, T, 768), pitch: (B, T), returns mel: (B, T, 80)
        return self.model(hubert, pitch, target_length, mask)

    def training_step(self, batch, batch_idx):
        hubert, pitch, mel_target, lengths = batch
        # 予測
        mel_pred = self(hubert, pitch, target_length=mel_target.size(1), mask=None)
        # L1 ロス
        loss = F.l1_loss(mel_pred, mel_target)
        # ログ
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        hubert, pitch, mel_target, lengths = batch
        mel_pred = self(hubert, pitch, target_length=mel_target.size(1), mask=None)
        loss = F.l1_loss(mel_pred, mel_target)
        self.val_losses.append(loss.detach())

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
        # LoRA のみを更新対象とする
        lora_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = AdamW(lora_params, lr=self.hparams.lr)
        # 必要に応じてスケジューラを追加
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr / 10.0
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

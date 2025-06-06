import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from align_model import TransformerAligner
from align_lora import apply_lora_to_transformeraligner

# ----------------------------------------------------------------
# LightningModule for Alignment with LoRA
# ----------------------------------------------------------------
class AlignTransformerSystem(pl.LightningModule):
    def __init__(
        self,
        lr: float = 2e-4,
        input_dim_hubert: int = 768,
        input_dim_pitch: int = 1,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 512,
        dropout: float = 0.1,
        diag_w: float = 1.0,
        ce_w: float = 1.0,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        ckpt_path = None
    ):
        super().__init__()
        self.save_hyperparameters()

        # ① ベースの TransformerAligner を生成し、事前学習済みチェックポイントをロード
        base_model = TransformerAligner(
            input_dim_hubert=self.hparams.input_dim_hubert,
            input_dim_pitch=self.hparams.input_dim_pitch,
            d_model=self.hparams.d_model,
            nhead=self.hparams.nhead,
            num_layers=self.hparams.num_layers,
            dim_feedforward=self.hparams.dim_ff,
            dropout=self.hparams.dropout,
            diag_weight=self.hparams.diag_w,
            ce_weight=self.hparams.ce_w,
        )
        # （必要に応じてベースモデルの重みをロード）
        ckpt = torch.load(ckpt_path, map_location="cpu")
        base_model.load_state_dict(ckpt, strict=True)

        # ② LoRA を適用してモデルをラップ
        self.model = apply_lora_to_transformeraligner(
            base_model,
            rank=self.hparams.lora_rank,
            alpha=self.hparams.lora_alpha
        )

        # LoRA以外の重みをすべて凍結
        for name, param in self.model.named_parameters():
          # "A" または "B" が名前に含まれるもの（LoRA の行列）は学習対象のままにする
          if not ("A" in name or "B" in name):
            param.requires_grad = False

        # 以降、LoRA 部分のみを学習するために base_model 側の重みはすでに requires_grad=False になっているはず
        self.train_losses = []
        self.val_losses = []

    def forward(self, src_hubert, src_pitch, tgt_hubert, tgt_pitch):
        return self.model(src_hubert, src_pitch, tgt_hubert, tgt_pitch)

    def training_step(self, batch, batch_idx):
        src_h, src_p, tgt_h, tgt_p = batch
        loss, metrics = self(src_h, src_p, tgt_h, tgt_p)
        self.log_dict(metrics, on_step=True, on_epoch=True)
        self.train_losses.append(loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        src_h, src_p, tgt_h, tgt_p = batch
        loss, _ = self(src_h, src_p, tgt_h, tgt_p)
        self.val_losses.append(loss.detach())

    def configure_optimizers(self):
        # LoRA の A/B パラメータのみを最適化
        lora_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(lora_params, lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr / 10.0
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def greedy_decode(self, batch, max_len=200):
        src_h, src_p, _, _ = batch
        return self.model.greedy_decode(src_h, src_p, max_len)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.greedy_decode(batch)

    def on_train_epoch_end(self):
        avg_train = torch.stack(self.train_losses).mean()
        self.log('train_total', avg_train, on_step=False, on_epoch=True, prog_bar=True)
        self.train_losses.clear()

    def on_validation_epoch_end(self):
        if not self.val_losses:
            return
        avg_loss = torch.stack(self.val_losses).mean()
        self.log(
            "val_loss",
            avg_loss,
            prog_bar=True,    # プログレスバーにも出す
            on_step=False,    # バッチごとでは出さない
            on_epoch=True     # エポック平均を出す
        )
        self.val_losses.clear()

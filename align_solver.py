import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
#from kornia.losses import SoftDTW
from align_model import TransformerAligner

# ----------------------------------------------------------------
# LightningModule for Alignment
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
        ce_w: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerAligner(
            input_dim_hubert=self.hparams.input_dim_hubert,
            input_dim_pitch=self.hparams.input_dim_pitch,
            d_model=self.hparams.d_model,
            nhead=self.hparams.nhead,
            num_layers=self.hparams.num_layers,
            dim_feedforward=self.hparams.dim_ff,
            dropout=self.hparams.dropout,
            diag_weight=self.hparams.diag_w,
            ce_weight=self.hparams.ce_w
        )
        self.train_losses = []

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
        '''
        self.log(
            "val_loss",
            loss,
            on_step=False,    # バッチ単位ではログしない
            on_epoch=True,    # epoch 終了時に集計してログ
            prog_bar=True     # プログレスバーにも表示
        )
        '''
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

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
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack(outputs).mean()

        self.log(
            "val_loss",
            avg_val_loss,
            prog_bar=True,    # プログレスバーにも出す
            on_step=False,    # バッチごとでは出さない
            on_epoch=True     # エポック平均を出す
        )
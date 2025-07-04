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
        ce_w: float = 1.0,
        free_run_interval: int = 100,
        free_run_weight: float = 1.0,
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
        self.val_losses = []
        self.free_run_interval = free_run_interval
        self.free_run_weight = free_run_weight
        
    def forward(self, src_hubert, src_pitch, tgt_hubert, tgt_pitch):
        return self.model(src_hubert, src_pitch, tgt_hubert, tgt_pitch)

    '''
    def training_step(self, batch, batch_idx):
        src_h, src_p, tgt_h, tgt_p = batch
        loss, metrics = self(src_h, src_p, tgt_h, tgt_p)
        self.log_dict(metrics, on_step=True, on_epoch=True)
        self.train_losses.append(loss.detach())
        return loss
    '''
    def training_step(self, batch, batch_idx):
        src_h, src_p, tgt_h, tgt_p = batch

        # (1) Teacher‐forcing の損失とメトリクスを計算
        loss_tf, metrics = self(src_h, src_p, tgt_h, tgt_p)
        loss = loss_tf

        # (2) free‐run（greedy）L1 を 10 バッチに 1 回だけ追加
        if batch_idx % self.free_run_interval == 0 and self.free_run_weight > 0:
            # ── メモリ節約: 先頭1サンプルのみ＆勾配不要 ──
            with torch.no_grad():
                pred_h, pred_p = self.model.greedy_decode(
                    src_h[:1], src_p[:1], max_len=tgt_h.size(1)
                )
            T = tgt_h.size(1)
            # pad / trunc（先頭サンプルだけ処理）
            if pred_h.size(1) < T:
                pad = T - pred_h.size(1)
                pred_h = torch.cat([
                    pred_h,
                    torch.zeros(1, pad, pred_h.size(2), device=self.device)
                ], dim=1)
                pred_p = torch.cat([
                    pred_p,
                    torch.zeros(1, pad, device=self.device)
                ], dim=1)
            else:
                pred_h = pred_h[:, :T]
                pred_p = pred_p[:, :T]

            # free-run L1 損失（先頭サンプルのみ比較）
            loss_fr_h = F.l1_loss(pred_h, tgt_h[:1])
            loss_fr_p = F.l1_loss(pred_p, tgt_p[:1])
            loss_fr   = loss_fr_h + loss_fr_p

            # 合成＆メトリクス追加
            loss = loss_tf + self.free_run_weight * loss_fr
            metrics['fr_l1_h'] = loss_fr_h
            metrics['fr_l1_p'] = loss_fr_p        

        # (3) ログと return は従来どおり
        self.log_dict(metrics, on_step=True, on_epoch=True)
        self.train_losses.append(loss.detach())
        return loss
    
    def validation_step(self, batch, batch_idx):
        src_h, src_p, tgt_h, tgt_p = batch
        loss, _ = self(src_h, src_p, tgt_h, tgt_p)
        self.val_losses.append(loss.detach())

        #return loss.detach()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=self.hparams.lr/10.0
        )
        return {"optimizer": opt, "lr_scheduler": sched, "gradient_clip_val": 1.0}

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

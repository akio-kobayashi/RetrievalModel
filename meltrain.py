import argparse
from pathlib import Path
import yaml
import warnings

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# local modules (already refactored for *no crop*)
from melvcdataset import VCMelDataset, data_processing
from melsolver import MelVCSystem

warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue.*")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message="The number of training batches.*")

# --------------------------------------------------
# train() util
# --------------------------------------------------

def train(cfg: dict):
    # ─── dataset ───────────────────────────────────────────────
    stats = torch.load(cfg["stats_tensor"], map_location="cpu", weights_only=True)
    train_ds = VCMelDataset(cfg["train_csv"], stats)  # ← no cropping args
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=data_processing,
        pin_memory=True,
    )

    valid_ds = VCMelDataset(cfg["valid_csv"], stats)  # ← no cropping args
    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=data_processing,
        pin_memory=True,
    )


    # ─── model ────────────────────────────────────────────────
    model = MelVCSystem(
        lr_g=cfg.get("lr_g", 2e-4),
        lr_d=cfg.get("lr_d", 2e-4),
        lambda_fm=cfg.get("lambda_fm", 2.0),
        lambda_mel=cfg.get("lambda_mel", 1.0),
        lambda_adv=cfg.get("lambda_adv", 1.0),
        sched_gamma=cfg.get("sched_gamma", 0.5),
        sched_step=cfg.get("sched_step", 200),
        grad_accum=cfg.get("grad_accum", 1),
        warmup_epochs=cfg.get("warmup_epochs", 10),
    )

    # ─── callbacks / logger ───────────────────────────────────
    steps_per_epoch = len(train_dl)
    ckpt_cb = ModelCheckpoint(
        dirpath=cfg["ckpt_dir"],
        filename="{epoch:02d}-{loss_mel:.4f}",
        monitor="val_loss_mel",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    tb_logger = TensorBoardLogger(save_dir=cfg["log_dir"], name="vc")

    # ─── trainer ──────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=cfg.get("max_epochs", 500),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.get("gpus", 1),
        precision="16-mixed",
        default_root_dir=cfg["work_dir"],
        logger=tb_logger,
        callbacks=[ckpt_cb, lr_monitor],
        profiler="simple",
        check_val_every_n_epoch=0,  # disable validation
    )

    trainer.fit(model, train_dl, valid_dl)

# --------------------------------------------------
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Train MelRVC model (no crop)")
    parser.add_argument("--config", type=str, default="mel.yaml", help="YAML config")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train(cfg)

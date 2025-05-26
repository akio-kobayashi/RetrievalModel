import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# local modules
from melvcdataset import VCMelDataset, data_processing
from melsolver import MelVCSystem

import warnings
warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue.*")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message="The number of training batches.*")

def train(cfg):
    stats = torch.load(cfg["stats_tensor"], map_location="cpu", weights_only=True)
    train_ds = VCMelDataset(cfg["train_csv"], stats, max_sec=cfg.get("max_sec", 2.0))

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=data_processing,
        pin_memory=True,
    )

    # ---------------- Model ----------------
    model = MelVCSystem(
        lr_g        = cfg.get("lr_g",        2e-4),
        lr_d        = cfg.get("lr_d",        2e-4),
        lambda_fm   = cfg.get("lambda_fm",   2.0),
        lambda_mel  = cfg.get("lambda_mel",  1.0),
        lambda_adv  = cfg.get("lambda_adv",  1.0),
        sched_gamma = cfg.get("sched_gamma", 0.5),
        sched_step  = cfg.get("sched_step",  200),
        grad_accum  = cfg.get("grad_accum", 1),
        warmup_epochs = cfg.get("warmup_epochs", 10),
    )

    # ---------------- Checkpoint callback ----------------
    steps_per_epoch = len(train_dl)
    ckpt_cb = ModelCheckpoint(
        dirpath=cfg["ckpt_dir"],
        filename="{epoch:02d}-{loss_mel:.4f}",
        monitor="loss_mel",       # ← 学習メル損失で保存
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=steps_per_epoch,  # 1エポックごと
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    tb_logger = TensorBoardLogger(save_dir=cfg["log_dir"], name="vc")

    trainer = pl.Trainer(
        max_epochs=cfg.get("max_epochs", 500),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.get("gpus", 1),
        precision="16-mixed",
        default_root_dir=cfg["work_dir"],
        logger=tb_logger,
        callbacks=[ckpt_cb, lr_monitor],
        profiler="simple",
        # バリデーション不要
        check_val_every_n_epoch = None,
    )

    trainer.fit(model, train_dl)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description="Train MelRVC model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train(cfg)

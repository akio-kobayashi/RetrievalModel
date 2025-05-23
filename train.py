import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# local modules
from vcdataset import VCWaveDataset, data_processing
from solver import VCSystem

import warnings
# ---- Suppress noisy framework warnings ----
warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue.*")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message="The number of training batches.*")

################################################################################
#  Training entry point                                                        #
################################################################################

def train(cfg):
    # ---------------- Dataset & Dataloader ----------------
    stats = torch.load(cfg["stats_tensor"], map_location="cpu", weights_only=True)  # (mean,std)

    train_ds = VCWaveDataset(cfg["train_csv"], stats, target_sr=cfg.get("sr", 16_000), max_sec=cfg.get("max_sec", 2.0))
    #val_ds   = VCWaveDataset(cfg["val_csv"],   stats, target_sr=cfg.get("sr", 16_000), max_sec=cfg.get("max_sec", 2.0))

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=data_processing,
        pin_memory=True,
    )

    '''    
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=data_processing,
        pin_memory=True,
    )
    '''
    
    # ---------------- Model ----------------
    model = VCSystem(
        sr          = cfg.get("sr",          16_000),
        hop         = cfg.get("hop",         320),
        lr_g        = cfg.get("lr_g",        2e-4),
        lr_d        = cfg.get("lr_d",        2e-4),
        lambda_fm   = cfg.get("lambda_fm",   2.0),
        lambda_mag  = cfg.get("lambda_mag",  1.0),
        lambda_sc   = cfg.get("lambda_sc",   1.0),
        lambda_cmp =  cfg.get("lambda_cmp",  1.0),
        lambda_wav  = cfg.get("lambda_wav",  0.1),
        lambda_pitch= cfg.get("lambda_pitch", 1.0), 
        sched_gamma = cfg.get("sched_gamma", 0.5),
        sched_step  = cfg.get("sched_step",  200),
        grad_accum  = cfg.get("grad_accum", 1),
        warmup_steps= cfg.get("warmup_steps", 5000),
        mse_steps   = cfg.get("mse_steps", 2000),
        adv_scale   = cfg.get("adv_scale", 1.0), 
        max_norm    = cfg.get("max_norm", 5.0),
        upsample_rates = cfg.get("upsample_rates", [5, 2, 4, 2, 2]),
    )

    # ---------------- Checkpoint callback ----------------
    steps_per_epoch = len(train_dl)
    ckpt_cb = ModelCheckpoint(
        dirpath=cfg["ckpt_dir"],
        filename="{epoch:02d}-{loss_mag_epoch:.4f}",
        monitor="loss_mag",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=steps_per_epoch,   # ← 1 epoch ごとに保存
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")   # "epoch" でも可

    # ---------------- Logger ----------------
    tb_logger = TensorBoardLogger(save_dir=cfg["log_dir"], name="vc")

    # ---------------- Trainer ----------------
    trainer = pl.Trainer(
        max_epochs=cfg.get("max_epochs", 500),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.get("gpus", 1),
        precision="16-mixed",
        default_root_dir=cfg["work_dir"],
        logger=tb_logger,
        callbacks=[ckpt_cb, lr_monitor],
        profiler="simple",
        check_val_every_n_epoch = cfg.get("check_val_every_n_epoch", 10),
    )

    trainer.fit(model, train_dl)

################################################################################
#  CLI                                                                         #
################################################################################

if __name__ == "__main__":
  torch.set_float32_matmul_precision('high')
  parser = argparse.ArgumentParser(description="Train RVC-style VC model")
  parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
  args = parser.parse_args()

  with open(args.config, "r", encoding="utf-8") as f:
      cfg = yaml.safe_load(f)
      
  train(cfg)

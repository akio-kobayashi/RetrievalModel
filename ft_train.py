import argparse
from pathlib import Path
import yaml
import warnings

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from ft_dataset import KeySynchronizedDataset, collate_alignment_rvc
from ft_solver import AlignmentRVCSystem, freeze_module, unfreeze_module

warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue.*")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message="The number of training batches.*")
warnings.filterwarnings("ignore", message=r".*does not have many workers which may be a bottleneck.*")

def train(cfg: dict):
    # ─── dataset ──────────────────────
    stats = torch.load(cfg["stats_tensor"], map_location="cpu", weights_only=True)
    train_ds = KeySynchronizedDataset(cfg["train_csv"],
                                      cfg["train_mel_csv"],
                                      stats,
                                      shuffle=True,
                                      max_length=cfg.get('max_length', 512))
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_alignment_rvc,
        pin_memory=True,
    )
    valid_ds = KeySynchronizedDataset(cfg["valid_csv"],
                                      cfg["valid_mel_csv"],
                                      stats,
                                      shuffle=False)
    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_alignment_rvc,
        pin_memory=True,
    )

    # ─── model ────────────────────────
    model = AlignmentRVCSystem(
        align_ckpt    = cfg["align_ckpt"],
        mel_ckpt      = cfg["mel_ckpt"],
        lr            = cfg.get("lr", 2e-4),
        input_dim_hubert = cfg.get("input_dim_hubert", 768),
        input_dim_pitch  = cfg.get("input_dim_pitch", 1),
        align_d_model    = cfg.get("align_d_model", 256),
        align_nhead      = cfg.get("align_nhead", 4),
        align_num_layers = cfg.get("align_num_layers", 3),
        align_dim_ff     = cfg.get("align_dim_ff", 512),
        align_dropout    = cfg.get("align_dropout", 0.1),
        diag_w           = cfg.get("diag_w", 1.0),
        ce_w             = cfg.get("ce_w", 1.0),
        latent_ch        = cfg.get("latent_ch", 256),
        rvc_d_model      = cfg.get("rvc_d_model", 256),
        rvc_n_conformer  = cfg.get("rvc_n_conformer", 8),
        rvc_nhead        = cfg.get("rvc_nhead", 8),
        align_weight     = cfg.get("align_weight", 1.0),
        mel_weight       = cfg.get("mel_weight", 1.0),
        update_aligner   = cfg.get("update_aligner", True),
        update_rvc       = cfg.get("update_rvc", True),
        load_from_pretrained = cfg.get("load_from_pretrained", True),
    )

    load_from_pretrained = cfg.get("load_from_pretrained", True)
    if not load_from_pretrained:
        ckpt = cfg.get("model_ckpt", None)
        model = model = AlignmentRVCSystem.load_from_checkpoint(ckpt)
        update_aligner = cfg.get("update_aligner", True)
        update_rvc = cfg.get("update_rvc", True)
        if not update_aligner:
            freeze_module(model.aligner)
        else:
            unfreeze_module(model.aligner)
        if not update_rvc:
            freeze_module(model.rvc)
        else:
            unfreeze_module(model.rvc)
                
    # ─── callbacks / logger ─────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=cfg["ckpt_dir"],
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss_mel",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    tb_logger = TensorBoardLogger(save_dir=cfg["log_dir"], name="align_rvc")

    # ─── trainer ───────────────────────
    trainer = pl.Trainer(
        precision=32,
        max_epochs=cfg.get("max_epochs", 100),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.get("gpus", 1),
        gradient_clip_val=cfg.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm="norm",
        default_root_dir=cfg["work_dir"],
        logger=tb_logger,
        callbacks=[ckpt_cb, lr_monitor],
        profiler="simple",
        check_val_every_n_epoch=1,
    )

    # ─── training ──────────────────────
    trainer.fit(model, train_dl, valid_dl)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Train Alignment+RVC model")
    parser.add_argument(
        "--config", type=str, default="alignment_rvc_config.yaml",
        help="YAML config for training"
    )
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train(cfg)

#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ft_solver import AlignmentRVCSystem
from ft_dataset import KeySynchronizedDataset, collate_alignment_rvc

import warnings
# pkg_resourcesの警告を抑制
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="lightning_utilities.core.imports"
)

# TypedStorageの警告を抑制
warnings.filterwarnings(
    "ignore",
    message="TypedStorage is deprecated.*",
    category=UserWarning,
    module="torch._utils"
)

# CuDNN workaroundの警告を抑制
warnings.filterwarnings(
    "ignore",
    message="Applied workaround for CuDNN issue.*",
    category=UserWarning,
    module="torch.nn.modules.conv"
)

def attach_nan_hooks(model):
    def hook(mod, inp, out):
        out_tensor = out[0] if isinstance(out, (tuple, list)) else out
        if torch.isnan(out).any() or torch.isinf(out).any():
            name = mod.__class__.__name__
            print(f"[NaN] in {name}")
            for t in inp:
                if torch.isnan(t).any():
                    print("  └─ input has NaN")
            raise RuntimeError("NaN detected – abort")
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear,
                          torch.nn.MultiheadAttention,
                          torch.nn.LayerNorm,
                          torch.nn.Conv1d)):
            m.register_forward_hook(hook)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--stats', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--mel_csv', required=True)
    parser.add_argument('--out_dir', default='melgen_tensors')
    #parser.add_argument('--out_csv', default='melgen.csv')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_len', type=int, default=1000)
    #parser.add_argument('--config', type=str)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ─── 統計読み
    stats = torch.load(args.stats, map_location="cpu", weights_only=True)
    
    # ─── モデル読み込み
    '''
    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = AlignmentRVCSystem(
        align_ckpt    = cfg.get("align_ckpt", None),
        mel_ckpt      = cfg.get("mel_ckpt",None),
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
        load_from_pretrained = cfg.get("load_from_pretrained", False),
    )
    '''
    model = AlignmentRVCSystem.load_from_checkpoint(args.ckpt)
    model.eval()

    attach_nan_hooks(model.aligner)
    
    # ─── CSV読み込み ───
    valid_ds = KeySynchronizedDataset(args.csv,
                                      args.mel_csv,
                                      stats,
                                      shuffle=False)
    valid_dl = DataLoader(
        valid_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_alignment_rvc,
        pin_memory=True,
    )

    iterator = iter(valid_dl)

    while True:
        try:
            batch = next(iterator)
            # バッチ処理
            with torch.no_grad():
                src_h, src_p, tgt_h, tgt_p, mel_tgt = batch
                # mel_tgt_len = mel_tgt.size(1)
                mel_pred = model.forward(src_h.to(device), src_p.to(device), max_len=args.max_len,
                                         mel_target_len=args.max_len).cpu().squeeze(0)
                print(mel_pred.min(), mel_pred.max(),
                      mel_pred.mean(), mel_pred.std())                
                mel_pred = valid_ds.unnormalize(mel_pred)
                print(mel_pred.min(), mel_pred.max(),
                      mel_pred.mean(), mel_pred.std())
                out_path = os.path.join(args.out_dir, valid_ds.current_key + "_mel.pt")
                torch.save(mel_pred, out_path)
            
        except StopIteration:
            break

if __name__ == "__main__":
    main()

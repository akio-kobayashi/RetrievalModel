#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
import torch
from align_model import TransformerAligner
import os, sys

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--stats', required=True)
ap.add_argument('--csv', nargs='+', required=True)
ap.add_argument('--mel_csv', required=True)
ap.add_argument('--out_dir', default='aligned_tensors')
ap.add_argument('--out_csv', default='aligned.csv')
ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--max_len', type=int, default=400)
args = ap.parse_args()

Path(args.out_dir).mkdir(exist_ok=True, parents=True)

device = torch.device(args.device)

# ──────────────────── グローバル統計量 ────────────────────
pitch_stats = torch.load(args.stats, map_location="cpu")
PITCH_MEAN = pitch_stats["pitch_mean"].to(device)  # shape: ()
PITCH_STD = pitch_stats["pitch_std"].to(device)

# ──────────────────── モデルのロード ────────────────────
ckpt = torch.load(args.ckpt, map_location="cpu")
cfg = ckpt.get("config", {})  # state_dict しか無い場合は空 dict
net = TransformerAligner(**cfg).to(device)
net.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
net.eval()

# ──── mel CSV 読み込み ────
mel_df = pd.read_csv(args.mel_csv).set_index('key')

# ──── 出力用 DataFrame ────
out_rows = []

# ---- 評価用 CSV読み込み ---
eval_df = pd.read_csv(args.csv)

# ──────────────────── メインループ ────────────────────
for idx, row in eval_df.iterrows():

    # 1．特徴量ロード
    pt = torch.load(row['source'])
    hubert = pt['hubert'].float()
    log_f0 = pt['log_f0'].float()

    # 2．log-F0 を正規化
    log_f0_norm = (log_f0 - PITCH_MEAN) / (PITCH_STD + 1.0e-9)
    hubert = hubert.unsqueeze(0) # add batch dim
    log_f0_norm = log_f0_norm.unsqueeze(0)

    # 3．greedy デコード
    with torch.no_grad():
        hubert_hat, log_f0_hat_norm = net.greedy_decode(
            hubert, log_f0_norm, max_len=args.max_len
        )  # (1,T̂,768), (1,T̂)

    # 4．逆正規化
    log_f0_hat = log_f0_hat_norm * PITCH_STD + PITCH_MEAN

    # 5．保存
    mel_path = Path(mel_df.loc[row['key'], 'hubert'])
    pt_mel = torch.load(mel_path, map_location='cpu').float()
    mel = pt_mel['mel'].float()

    out_path = os.path.jon(args.out_dir, row['key']+"_aligned.pt")
    torch.save(
        {
            "hubert": hubert_hat.squeeze(0).cpu(),
            "log_f0": log_f0_hat.squeeze(0).cpu(),
            "mel": mel.cpu()
        },
        out_path,
    )

    out_rows.append({'key': row['key'], 'hubert': str(out_path)})
    
pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)


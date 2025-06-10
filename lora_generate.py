#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import pandas as pd
import torch

from align_model import TransformerAligner
from align_lora import apply_lora_to_transformeraligner
from mellora import RVCStyleVC_LoRA
from your_module import AlignmentRVCSystem  # モデル全体をwrapするLightningModule

def load_pitch_stats(stats_path, device):
    stats = torch.load(stats_path, map_location='cpu')
    return stats["pitch_mean"].to(device), stats["pitch_std"].to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--stats', required=True)
    parser.add_argument('--csv', required=True, nargs='+')
    parser.add_argument('--mel_csv', required=True)
    parser.add_argument('--out_dir', default='melgen_tensors')
    parser.add_argument('--out_csv', default='melgen.csv')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_len', type=int, default=400)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ─── 統計読み込み ───
    pitch_mean, pitch_std = load_pitch_stats(args.stats, device)

    # ─── モデル読み込み ───
    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt.get("hyper_parameters", {})
    model = AlignmentRVCSystem(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ─── CSV読み込み ───
    mel_df = pd.read_csv(args.mel_csv).set_index('key')
    in_df = pd.concat([pd.read_csv(f) for f in args.csv])
    out_rows = []

    for _, row in in_df.iterrows():
        # 特徴量ロード
        pt = torch.load(row['source'])
        hubert = pt['hubert'].float().unsqueeze(0).to(device)
        log_f0 = pt['log_f0'].float()
        log_f0_norm = ((log_f0 - pitch_mean) / (pitch_std + 1e-9)).unsqueeze(0).to(device)

        # 推論
        with torch.no_grad():
            mel_pred = model(hubert, log_f0_norm, max_len=args.max_len, mel_target_len=args.max_len)

        # 参照mel読み込み（構造保持のため）
        mel_path = Path(mel_df.loc[row['key'], 'hubert'])
        mel_ref = torch.load(mel_path, map_location='cpu')['mel'].float()

        # 保存
        out_path = os.path.join(args.out_dir, row['key'] + "_mel.pt")
        torch.save({
            "mel": mel_pred.squeeze(0).cpu(),
            "hubert": hubert.squeeze(0).cpu(),
            "log_f0": (log_f0_norm.squeeze(0) * pitch_std + pitch_mean).cpu()
        }, out_path)

        out_rows.append({'key': row['key'], 'hubert': out_path})

    pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()

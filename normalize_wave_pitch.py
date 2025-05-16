import argparse
import csv
from pathlib import Path
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

def compute_stats(csv):
    df = pd.read_csv(csv)
    f0_list = []
    for idx, row in df.iterrows():
        pt = torch.load(row["hubert"], map_location="cpu", weights_only=True)
        f0_list.append(pt["log_f0"].float())
    cat = torch.cat(f0_list)
    pitch_mean = cat.mean().item()
    pitch_std  = cat.std(unbiased=False).item() + 1e-9
    return pitch_mean, pitch_std

def main():
    ap = argparse.ArgumentParser(description="Compute global mean/std of pitch from analyzed data")
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--out", type=Path, help="output stats file")
    ap.add_argument("--stats", type=str)
    args = ap.parse_args()

    stats = torch.load(args.stats, map_location="cpu", weights_only=True)  # (mean,std)

    pitch_mean, pitch_std = compute_stats(args.csv)
    output_stats = {"mean": stats['mean'], "std": stats['std'], "pitch_mean": pitch_mean, "pitch_std": pitch_std}
    torch.save(output_stats, args.out)


if __name__ == "__main__":
    main()

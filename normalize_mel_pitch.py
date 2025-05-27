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
    mel_list = []

    for _, row in df.iterrows():
        pt = torch.load(row["hubert"], map_location="cpu", weights_only=True)
        f0_list.append(pt["log_f0"].float())
        mel = pt["mel"].float().squeeze()
        if mel.size(0) == 80:          # stored as (80, T)
            mel = mel.transpose(0, 1)
        if mel.size(1) != 80:
            raise ValueError(f"Unexpected mel shape: {mel.shape}")
        mel_list.append(mel)
    f0_cat = torch.cat(f0_list)
    mel_cat = torch.cat(mel_list, dim=0)
    pitch_mean = f0_cat.mean().item()
    pitch_std  = f0_cat.std(unbiased=False).item() + 1e-9
    mel_mean   = mel_cat.mean(dim=0)
    mel_std    = mel_cat.std(dim=0) + 1e-9    
    return mel_mean, mel_std, pitch_mean, pitch_std

def main():
    ap = argparse.ArgumentParser(description="Compute global mean/std of pitch from analyzed data")
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--out", type=Path, help="output stats file")
    #ap.add_argument("--stats", type=str)
    args = ap.parse_args()

    #stats = torch.load(args.stats, map_location="cpu", weights_only=True)  # (mean,std)

    mel_mean, mel_std, pitch_mean, pitch_std = compute_stats(args.csv)
    output_stats = {"mel_mean": mel_mean, "mel_std": mel_std, "pitch_mean": pitch_mean, "pitch_std": pitch_std}
    torch.save(output_stats, args.out)


if __name__ == "__main__":
    main()

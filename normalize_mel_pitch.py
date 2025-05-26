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
    for idx, row in df.iterrows():
        pt = torch.load(row["hubert"], map_location="cpu", weights_only=True)
        f0_list.append(pt["log_f0"].float())
        mel = pt["mel"].float()
        if mel.size(0) == 80 and mel.size(1) != 80:
            mel = mel.transpose(0, 1)
            
        mel_list.append(mel)
    cat = torch.cat(f0_list)
    pitch_mean = cat.mean().item()
    pitch_std  = cat.std(unbiased=False).item() + 1e-9
    cat = torch.cat(mel_list)
    mel_mean = cat.mean().item()
    mel_std = cat.std(unbiased=False.item() + 1e-9)
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

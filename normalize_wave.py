import argparse
import csv
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


def running_mean_std(mean, var, count, batch):
    """Update running mean/var with new batch tensor (1‑D)."""
    b_count = batch.numel()
    if b_count == 0:
        return mean, var, count

    b_mean = batch.mean()
    b_var = batch.var(unbiased=False)

    delta = b_mean - mean
    total = count + b_count

    new_mean = mean + delta * b_count / total
    new_var = (
        (var * count + b_var * b_count + delta.pow(2) * count * b_count / total) / total
    )
    return new_mean, new_var, total


def compute_stats(csv_path: Path, target_sr: int = 16_000):
    mean = torch.tensor(0.0)
    var = torch.tensor(0.0)
    count = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        wave_paths = [Path(r["wave"]) for r in reader]

    resampler_cache = {}

    for w in tqdm(wave_paths, desc="Accumulating stats"):
        wav, sr = torchaudio.load(w)
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != target_sr:
            if sr not in resampler_cache:
                resampler_cache[sr] = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler_cache[sr](wav)
        wav = wav.squeeze(0)
        mean, var, count = running_mean_std(mean, var, count, wav)

    std = var.sqrt()
    return mean.item(), std.item()


def main():
    ap = argparse.ArgumentParser(description="Compute global mean/std of wave column in CSV")
    ap.add_argument("csv", type=Path)
    ap.add_argument("out", type=Path, help="output .pt file to save tensor(mean,std)")
    ap.add_argument("sr", type=int, default=16000, help="target sampling rate")
    args = ap.parse_args()

    mean, std = compute_stats(args.csv, args.sr)
    stats = {"mean": mean, "std": std}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, args.out)
    print(f"Saved mean={mean:.6f}, std={std:.6f} → {args.out}")
    print(f"Saved mean={stats[0]:.6f}, std={stats[1]:.6f} → {args.out}")


if __name__ == "__main__":
    main()

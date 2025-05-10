import argparse
import csv
import random
from pathlib import Path
from typing import List, Dict

HEADER = ["key", "wave", "hubert"]

def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not set(HEADER).issubset(rows[0].keys()):
        raise ValueError(f"CSV must contain columns: {HEADER}")
    return rows

def save_csv(rows: List[Dict[str, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {path}")

def split_csv(src: Path, out_dir: Path, sample_n: int | None, train_ratio: float, seed: int):
    """Randomly sample *sample_n* rows then split train/val."""
    random.seed(seed)
    rows = load_csv(src)

    # --- optional subsampling ---
    if sample_n is not None and sample_n < len(rows):
        rows = random.sample(rows, sample_n)
        print(f"Sub‑sampled {sample_n} / {len(rows)} rows from CSV")

    random.shuffle(rows)
    n_train = int(len(rows) * train_ratio)
    train_rows, val_rows = rows[:n_train], rows[n_train:]

    save_csv(train_rows, out_dir / "train.csv")
    save_csv(val_rows,   out_dir / "val.csv")

# ------------------------------------------------------------
#  CLI                                                        
# ------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Random subsample and split CSV into train/val")
    ap.add_argument("--src", type=Path, required=True, help="input CSV file")
    ap.add_argument("--out_dir", type=Path, default="./", help="output directory")
    ap.add_argument("--num", type=int, default=None, help="number of rows to sample before split")
    ap.add_argument("--ratio", type=float, default=0.9, help="train split ratio")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    args = ap.parse_args()

    split_csv(args.src, args.out_dir, args.num, args.ratio, args.seed)

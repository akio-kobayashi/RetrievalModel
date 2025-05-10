import argparse
import csv
import random
from pathlib import Path

HEADER = ["key", "wave", "hubert"]


def split_csv(
    src: Path,
    out_dir: Path,
    train_ratio: float = 0.9,
    seed: int = 42,
):
    """Split CSV with columns key,wave,hubert into train/val sets."""

    random.seed(seed)
    with src.open(newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if not set(HEADER).issubset(reader[0].keys()):
        raise ValueError(f"CSV must contain columns: {HEADER}")

    random.shuffle(reader)
    n_train = int(len(reader) * train_ratio)
    train_rows = reader[:n_train]
    val_rows = reader[n_train:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"

    for path, rows in [(train_path, train_rows), (val_path, val_rows)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows → {path}")


def main():
    ap = argparse.ArgumentParser(description="Split dataset CSV into train/val")
    ap.add_argument("--src", type=Path, help="input CSV file")
    ap.add_argument("--out_dir", type=Path, default='./', help="output directory")
    ap.add_argument("--ratio", type=float, default=0.9, help="train split ratio")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed")
    args = ap.parse_args()

    split_csv(args.src, args.out_dir, args.ratio, args.seed)


if __name__ == "__main__":
    main()

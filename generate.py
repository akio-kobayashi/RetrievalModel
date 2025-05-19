from __future__ import annotations

"""Generate speech waveforms from HuBERT + pitch features listed in a CSV.

CSV must have at least these columns:
    key   : unique id used for output filename
    hubert: path to .pt tensor (keys: 'hubert', 'log_f0')

Example
-------
python generate.py \
    --csv val.csv \
    --ckpt checkpoints/epoch=199-step=10000.ckpt \
    --out_dir gen_wavs \
    --stats stats.pt         # optional: mean/std for de-normalisation
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio

from model import RVCStyleVC  # assumes PYTHONPATH includes project root

# -----------------------------------------------------------------------------
#  Utility
# -----------------------------------------------------------------------------

def load_generator(ckpt_path: Path, device: torch.device) -> RVCStyleVC:
    """Load only the Generator weights from a Lightning checkpoint."""
    state = torch.load(str(ckpt_path), map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    # pick keys starting with 'gen.'
    gen_sd: Dict[str, torch.Tensor] = {
        k.split("gen.")[1]: v for k, v in state.items() if k.startswith("gen.")
    }
    gen = RVCStyleVC()
    _ = gen.load_state_dict(gen_sd, strict=False)
    gen.eval().to(device)
    return gen


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"key", "hubert"}
    if not required.issubset(rows[0].keys()):
        raise ValueError(f"CSV must contain columns: {required}")
    return rows


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    gen = load_generator(args.ckpt, device)

    # stats for de-normalisation (optional)
    try:
        st = torch.load(args.stats, map_location="cpu")
        mean, std = st["mean"], st["std"]
        pitch_mean, pitch_std = st["pitch_mean"], st["pitch_std"]
        print(f"[info] Using mean/std = {mean:.4f}/{std:.4f} for inverse normalisation")
    except RuntimeError:
        print('specify stats file')
        exit(1)

    rows = load_rows(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for r in rows:
            key = r["key"]
            tens = torch.load(r["hubert"], map_location="cpu")
            hubert: torch.Tensor = tens["hubert"].unsqueeze(0).to(device)  # (1,T,768)
            pitch: torch.Tensor  = tens["log_f0"].unsqueeze(0).to(device)   # (1,T)
            pitch = (pitch - pitch_mean)/(pitch_std + 1.e-4)
            wav = gen(hubert, pitch).cpu().squeeze(0)  # (N,)

            # inverse normalisation if stats provided
            wav = wav * std + mean
            wav = wav.clamp(-1.0, 1.0)

            out_path = args.out_dir / f"{key}_gen.wav"
            torchaudio.save(str(out_path), wav.unsqueeze(0), args.sr)
            print("✓", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate waveforms from HuBERT features using RVC Generator")
    ap.add_argument("--csv", type=Path, required=True, help="CSV file with key, hubert columns")
    ap.add_argument("--ckpt", type=Path, required=True, help="Lightning checkpoint (*.ckpt)")
    ap.add_argument("--out_dir", type=Path, default=Path("gen"), help="output directory for wavs")
    ap.add_argument("--stats", type=Path, default=None, help="stats.pt with mean/std (optional)")
    ap.add_argument("--sr", type=int, default=16000, help="output sampling rate")
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    args = ap.parse_args()

    main(args)

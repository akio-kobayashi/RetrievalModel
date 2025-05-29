"""
Generate mel-spectrograms from HuBERT + pitch features listed in a CSV.

CSV must have at least:
    key   : unique id for output filename
    hubert: path to .pt tensor (keys: 'hubert', 'log_f0')

Example
-------
python generate_mel.py \
    --csv val.csv \
    --ckpt checkpoints/epoch=199-step=10000.ckpt \
    --out_dir gen_mels \
    --stats stats.pt          # must contain 'mel_mean', 'mel_std', 'pitch_mean', 'pitch_std'
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List
import torchaudio
import torch
import numpy as np
import matplotlib.pyplot as plt
from melmodel import RVCStyleVC      # Generator that outputs mel (B,T,80)
from gan_feature_pipeline import spectral_de_normalize_torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def load_generator(ckpt_path: Path, device: torch.device) -> RVCStyleVC:
    """Load only the Generator weights from a Lightning checkpoint."""
    state = torch.load(str(ckpt_path), map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    gen_sd: Dict[str, torch.Tensor] = {k.split("gen.")[1]: v
                                       for k, v in state.items() if k.startswith("gen.")}
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

def save_png(mel: np.ndarray, path: Path):
    """Save mel (T, 80) as PNG for quick preview."""
    plt.figure(figsize=(6, 3))
    plt.imshow(mel.T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    gen = load_generator(args.ckpt, device)

    # ─── stats for de-normalisation ──────────────────────────────────────────
    st = torch.load(args.stats, map_location="cpu") if args.stats else None
    if st is None or not {"mel_mean", "mel_std", "pitch_mean", "pitch_std"}.issubset(st.keys()):
        raise RuntimeError("stats.pt must contain mel_mean, mel_std, pitch_mean, pitch_std")
    mel_mean = torch.as_tensor(st["mel_mean"], dtype=torch.float).view(1, -1)   # (1,80)
    mel_std  = torch.as_tensor(st["mel_std"],  dtype=torch.float).view(1, -1)
    pitch_mean = float(st["pitch_mean"])
    pitch_std  = float(st["pitch_std"])

    rows = load_rows(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for r in rows:
            key = r["key"]
            wav,_ = torchaudio.load(r["wave"])
            tens = torch.load(r["hubert"], map_location=device)
            hubert = tens["hubert"].unsqueeze(0).to(device)  # (1,T,768)
            pitch  = tens["log_f0"].unsqueeze(0).to(device)   # (1,T)
            ref = tens["mel"].squeeze()
            pitch  = (pitch - pitch_mean) / (pitch_std + 1e-9)
            print(wav.shape)
            print(hubert.shape)
            print(ref.transpose(0, 1).shape)
            # ---------- Generator forward ----------
            mel = gen(hubert, pitch, target_length=ref.shape[-1]).cpu().squeeze(0)         # (T, 80)
            print(mel.shape)
            # inverse-norm
            mel = mel * mel_std + mel_mean                    # (T,80)
            mel = mel.clamp(min=-4.0, max=4.0)                # sanity clip
            loss = F.l1_loss(mel, ref).detach().numpy().item()
            print(loss)
            # ---------- save ----------
            out_pt = args.out_dir / f"{key}_gen.pt"
            torch.save(mel, out_pt)

            if args.save_png:
                out_png = args.out_dir / f"{key}_gen.png"
                save_png(mel.numpy(), out_png)

            print("✓", out_pt)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate mel-spectrograms with RVC Generator")
    ap.add_argument("--csv",     type=Path, required=True, help="CSV with key, hubert columns")
    ap.add_argument("--ckpt",    type=Path, required=True, help="Lightning checkpoint (*.ckpt)")
    ap.add_argument("--out_dir", type=Path, default=Path("gen_mels"), help="output directory (.pt)")
    ap.add_argument("--stats",   type=Path, required=True, help="stats.pt with mel_mean/std etc.")
    ap.add_argument("--cpu",     action="store_true", help="force CPU inference")
    ap.add_argument("--save_png",action="store_true", help="also save PNG preview")
    args = ap.parse_args()

    main(args)

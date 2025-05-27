import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
from einops import rearrange

class VCMelDataset(Dataset):
    """
    Return **full‑length** (hubert, pitch, mel) tuples.

    * HuBERT / pitch / mel are pre‑computed .pt tensors.
    * No random cropping – every sample is returned as‑is.
    * Pitch series length must equal HuBERT length.
    * Mel is HiFi‑GAN spec (hop = 256).  Saved as (80, T) or (T, 80).
    """

    def __init__(
        self,
        csv_path: str | Path,
        stats_tensor: Dict[str, float],
        sr: int = 16000,
    ) -> None:
        # ─── load CSV rows ────────────────────────────────────────────
        self.rows: List[Dict[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if "hubert" not in r:
                    raise ValueError("CSV must have 'hubert' column")
                self.rows.append(r)

        # ─── statistics ───────────────────────────────────────────────
        if {"pitch_mean", "pitch_std", "mel_mean", "mel_std"}.issubset(stats_tensor.keys()):
            self.pitch_mean = float(stats_tensor["pitch_mean"])
            self.pitch_std  = float(stats_tensor["pitch_std"]) + 1e-9
            self.mel_mean   = torch.as_tensor(stats_tensor["mel_mean"], dtype=torch.float).view(1, -1)
            self.mel_std    = torch.as_tensor(stats_tensor["mel_std" ], dtype=torch.float).view(1, -1) + 1e-9
        else:
            self._compute_stats(stats_tensor)

    # ------------------------------------------------------------------
    def _compute_stats(self, stats_tensor: Dict[str, float]):
        """Derive mean / std from all rows (slow ‑ one‑off)."""
        f0_list, mel_list = [], []
        for row in self.rows:
            pt = torch.load(row["hubert"], map_location="cpu", weights_only=True)
            f0_list.append(pt["log_f0"].float())
            mel = pt["mel"].float()
            if mel.size(0) == 80:          # stored as (80, T)
                mel = mel.transpose(0, 1)
            if mel.size(1) != 80:
                raise ValueError(f"Unexpected mel shape: {mel.shape}")
            mel_list.append(mel)
        f0_cat = torch.cat(f0_list)
        mel_cat = torch.cat(mel_list, dim=0)
        self.pitch_mean = f0_cat.mean().item()
        self.pitch_std  = f0_cat.std(unbiased=False).item() + 1e-9
        self.mel_mean   = mel_cat.mean(dim=0, keepdim=True)
        self.mel_std    = mel_cat.std (dim=0, unbiased=False, keepdim=True) + 1e-9

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        pt = torch.load(row["hubert"], map_location="cpu", weights_only=True)

        hubert: torch.Tensor = pt["hubert"].float()   # (T, 768)
        pitch:  torch.Tensor = pt["log_f0"].float()   # (T,)
        mel:    torch.Tensor = pt["mel"].float().squeeze()      # (80, T) or (T, 80)

        # ensure mel is (T, 80)
        if mel.ndim != 2:
            raise ValueError(f"mel tensor must be 2‑D, got {mel.shape}")
        if mel.size(0) == 80 and mel.size(1) != 80:
            mel = mel.transpose(0, 1)
        if mel.size(1) != 80:
            raise ValueError(f"Unexpected mel shape after transpose: {mel.shape}")

        # normalise
        pitch_norm = (pitch - self.pitch_mean) / self.pitch_std
        mel_norm   = (mel   - self.mel_mean)  / self.mel_std

        return hubert, pitch_norm, mel_norm

# ---------------------------------------------------------------------------
# Collate: pad to longest sequence in batch (unchanged)
# ---------------------------------------------------------------------------

def data_processing(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    huberts, pitches, mels = zip(*batch)
    B = len(batch)
    T_max = max(h.size(0) for h in huberts)
    D = huberts[0].size(1)
    M = 80

    h_pad = torch.zeros(B, T_max, D)
    p_pad = torch.zeros(B, T_max)
    m_pad = torch.zeros(B, max(m.size(0) for m in mels), M)

    for i, (h, p, m) in enumerate(batch):
        h_pad[i, :h.size(0)] = h
        p_pad[i, :p.size(0)] = p
        m_pad[i, :m.size(0)] = m

    return h_pad, p_pad, m_pad

# ------------------------------------------------------------
#  Stand-alone test
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--hubert_hop", type=int, default=320)
    parser.add_argument("--mel_hop", type=int, default=256)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--max_sec", type=float, default=2.0)
    args = parser.parse_args()

    stats = torch.load(args.stats, map_location="cpu", weights_only=True)
    ds = VCMelDataset(
        args.csv, stats,
        hubert_hop=args.hubert_hop,
        mel_hop=args.mel_hop,
        max_sec=args.max_sec,
        sr=args.sr,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, collate_fn=data_processing_mel)
    h, p, m = next(iter(dl))
    print("Shapes: HuBERT", h.shape, "| pitch", p.shape, "| mel", m.shape)

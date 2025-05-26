import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
import math

class VCMelDataset(Dataset):
    """
    Return (hubert, pitch, mel) with global normalization and HuBERT基準ランダムクロップ.

    - waveファイル不要
    - hubert, pitch, mel は.ptに保存
    - pitch系列長はhubertと常に一致
    - melはHiFiGANスペック
    - max_sec=None なら全長返す（評価時用）
    """

    def __init__(
        self,
        csv_path: str | Path,
        stats_tensor: Dict[str, float],
        hubert_hop: int = 320,
        mel_hop: int = 256,
        max_sec: Optional[float] = 2.0,
        sr: int = 16000,
    ) -> None:
        self.rows: List[Dict[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if "hubert" not in r:
                    raise ValueError("CSV must have 'hubert' column")
                self.rows.append(r)

        self.hubert_hop = hubert_hop
        self.mel_hop = mel_hop
        self.sr = sr
        self.max_frames = int(max_sec * sr // hubert_hop) if max_sec else None

        # pitch正規化
        if "pitch_mean" in stats_tensor and "pitch_std" in stats_tensor:
            self.pitch_mean = float(stats_tensor["pitch_mean"])
            self.pitch_std  = float(stats_tensor["pitch_std"]) + 1e-9
        else:
            f0_list = []
            for row in self.rows:
                pt = torch.load(row["hubert"], map_location="cpu", weights_only=True)
                f0_list.append(pt["log_f0"].float())
            cat = torch.cat(f0_list)
            self.pitch_mean = cat.mean().item()
            self.pitch_std  = cat.std(unbiased=False).item() + 1e-9

        # mel正規化（ベクトル, 各次元単位）
        if "mel_mean" in stats_tensor and "mel_std" in stats_tensor:
            self.mel_mean = torch.as_tensor(stats_tensor["mel_mean"]).float()
            self.mel_std  = torch.as_tensor(stats_tensor["mel_std"]).float() + 1e-9
        else:
            mel_list = []
            for row in self.rows:
                pt = torch.load(row["hubert"], map_location="cpu", weights_only=True)
                mel = pt["mel"].float()
                mel_list.append(mel)
            cat = torch.cat(mel_list, dim=0)
            self.mel_mean = cat.mean(dim=0)
            self.mel_std  = cat.std(dim=0, unbiased=False) + 1e-9

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        pt = torch.load(row["hubert"], map_location="cpu", weights_only=True)
        hubert: torch.Tensor = pt["hubert"].float()   # (T, D)
        pitch:  torch.Tensor = pt["log_f0"].float()   # (T,)
        mel:    torch.Tensor = pt["mel"].float()      # (T_mel, mel_dim)

        T = hubert.size(0)
        max_frames = self.max_frames or T
        if T > max_frames:
            t0 = torch.randint(0, T - max_frames + 1, (1,)).item()
        else:
            t0 = 0
        t1 = min(t0 + max_frames, T)

        # HuBERT・ピッチ（完全同期）
        hubert_crop = hubert[t0:t1]
        pitch_crop = pitch[t0:t1]
        pitch_norm = (pitch_crop - self.pitch_mean) / self.pitch_std

        # サンプル単位の時刻
        start_sample = t0 * self.hubert_hop
        end_sample   = t1 * self.hubert_hop

        # メルフレーム区間（時刻同期, floorで整数化）
        mel_start = int(math.floor(start_sample / self.mel_hop))
        mel_end   = int(math.floor(end_sample   / self.mel_hop))
        mel_crop  = mel[mel_start:mel_end]
        mel_norm  = (mel_crop - self.mel_mean) / self.mel_std

        return hubert_crop, pitch_norm, mel_norm

def data_processing_mel(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    huberts, pitches, mels = zip(*batch)
    B = len(batch)
    T_max = max(h.size(0) for h in huberts)
    D = huberts[0].size(1)
    M = mels[0].size(1) if M := mels[0].size(1) else 80

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

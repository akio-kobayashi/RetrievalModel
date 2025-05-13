import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

# ------------------------------------------------------------
#  Dataset                                                    
# ------------------------------------------------------------
class VCWaveDataset(Dataset):
    """Return (hubert, pitch, wav) with optional random cropping.

    Parameters
    ----------
    csv_path : str | Path
        CSV containing columns `hubert` (tensor pt) and `wave` (audio).
    stats_tensor : {"mean": float, "std": float}
        Global waveform mean/std for normalisation.
    target_sr : int
        Resample audio to this rate.
    hop : int
        Samples per HuBERT frame (16 kHz → 320).
    max_sec : float | None
        Crop length in seconds. None disables cropping.
    """

    def __init__(
        self,
        csv_path: str | Path,
        stats_tensor: Dict[str, float],
        target_sr: int = 16_000,
        hop: int = 320,
        max_sec: Optional[float] = 2.0,
    ) -> None:
        self.rows: List[Dict[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if "hubert" not in r or "wave" not in r:
                    raise ValueError("CSV must have 'hubert' and 'wave' columns")
                self.rows.append(r)

        self.mean = float(stats_tensor["mean"])
        self.std = float(stats_tensor["std"]) + 1e-9

        # Global stats for pitch normalization (compute if missing)
        if "pitch_mean" in stats_tensor and "pitch_std" in stats_tensor:
            self.pitch_mean = float(stats_tensor["pitch_mean"])
            self.pitch_std  = float(stats_tensor["pitch_std"]) + 1e-9
        else:
            # load all log_f0 to compute global mean/std
            f0_list = []
            for row in self.rows:
                pt = torch.load(row["hubert"], map_location="cpu", weights_only=True)
                f0_list.append(pt["log_f0"].float())
            cat = torch.cat(f0_list)
            self.pitch_mean = cat.mean().item()
            self.pitch_std  = cat.std(unbiased=False).item() + 1e-9

        self.target_sr = target_sr
        self.hop = hop
        self.max_samples = int(max_sec * target_sr) if max_sec else None
        self._resampler: Optional[T.Resample] = None

    def __len__(self):
        return len(self.rows)

    # --------------------------------------------------------
    def _get_resampler(self, orig_sr: int):
        if orig_sr == self.target_sr:
            return None
        if self._resampler is None or self._resampler.orig_freq != orig_sr:
            self._resampler = T.Resample(orig_sr, self.target_sr)
        return self._resampler

    # --------------------------------------------------------
    def __getitem__(self, idx: int):
        row = self.rows[idx]

        # HuBERT & pitch
        pt = torch.load(row["hubert"], map_location="cpu", weights_only=True)
        hubert: torch.Tensor = pt["hubert"].float()  # (T,D)
        pitch:  torch.Tensor = pt["log_f0"].float()  # (T,)

        # Normalize pitch globally
        pitch_norm = (pitch - self.pitch_mean) / self.pitch_std

        # Waveform
        wav, sr = torchaudio.load(row["wave"])
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)  # mono
        resampler = self._get_resampler(sr)
        if resampler is not None:
            wav = resampler(wav)
        wav = wav.squeeze(0)
        
        # グローバル正規化
        wav = (wav - self.mean) / self.std

        # Random crop
        if self.max_samples and wav.size(0) > self.max_samples:
            # フレームでの最大長
            max_frames = self.max_samples // self.hop
            num_frames = hubert.size(0)
            if num_frames > max_frames:
                # HuBERT フレームをランダム選択
                t0 = torch.randint(0, num_frames - max_frames + 1, (1,)).item()
                t1 = t0 + max_frames
                hubert = hubert[t0:t1]
                pitch_norm  = pitch_norm[t0:t1]
                # 対応する波形サンプルの start/end を計算
                start = t0 * self.hop
                end   = start + self.max_samples
                wav   = wav[start:end]
        # ローカル正規化
        local_mean = wav.mean()
        local_std = wav.std(unbiased=False) + 1.0e-9
        wav_norm = (wav - local_mean)/local_std

        return hubert, pitch_norm, wav_norm

# ------------------------------------------------------------
#  Collate                                                    
# ------------------------------------------------------------

def data_processing(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    huberts, pitches, waves = zip(*batch)
    B = len(batch)
    T_max = max(h.size(0) for h in huberts)
    N_max = max(w.size(0) for w in waves)
    D = huberts[0].size(1)

    h_pad = torch.zeros(B, T_max, D)
    p_pad = torch.zeros(B, T_max)
    w_pad = torch.zeros(B, N_max)

    for i, (h, p, w) in enumerate(batch):
        h_pad[i, :h.size(0)] = h
        p_pad[i, :p.size(0)] = p
        w_pad[i, :w.size(0)] = w

    return h_pad, p_pad, w_pad

# ------------------------------------------------------------
#  Stand‑alone test                                           
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    stats = torch.load(args.stats, map_location="cpu", weights_only=True)
    ds = VCWaveDataset(args.csv, stats, target_sr=args.sr)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, collate_fn=data_processing)
    h, p, w = next(iter(dl))
    print("Shapes:", h.shape, p.shape, w.shape)

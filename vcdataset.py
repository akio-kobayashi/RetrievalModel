import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F


class VCWaveDataset(Dataset):
    """Dataset that yields (hubert, pitch, wav_real).

    CSV file must contain columns:
        hubert : path to .pt tensor  (keys: 'hubert', 'log_f0')
        wave   : path to wav/flac/etc audio file

    Args
    ----
    csv_path:
        Path to CSV listing sample pairs.
    stats_tensor:
        Tensor with shape (2,) holding global mean, std of *output wave*.
        Used to normalise waveform on load:  (wav - mean) / std.
    target_sr:
        Waveform will be resampled to this rate if necessary.
    """

    def __init__(
        self,
        csv_path: str | Path,
        stats_tensor: torch.Tensor,
        target_sr: int = 16_000,
    ):
        self.rows: List[Dict[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if "hubert" not in r or "wave" not in r:
                    raise ValueError("CSV must have 'hubert' and 'wave' columns")
                self.rows.append(r)

        self.mean = stats_tensor[0].item()
        self.std = stats_tensor[1].item()
        self.target_sr = target_sr
        self._resampler: Optional[T.Resample] = None

    # ------------------------------------------------------------
    def __len__(self):
        return len(self.rows)

    # ------------------------------------------------------------
    def _get_resampler(self, orig_sr: int):
        if orig_sr == self.target_sr:
            return None
        if self._resampler is None or self._resampler.orig_freq != orig_sr:
            self._resampler = T.Resample(orig_sr, self.target_sr)
        return self._resampler

    # ------------------------------------------------------------
    def __getitem__(self, idx: int):
        row = self.rows[idx]

        # -------- HuBERT + pitch tensor --------
        pt = torch.load(row["hubert"], map_location="cpu")
        if "hubert" not in pt or "log_f0" not in pt:
            raise KeyError("Tensor file must contain 'hubert' and 'log_f0' keys")
        hubert: torch.Tensor = pt["hubert"].float()         # (T,D=768)
        pitch: torch.Tensor = pt["log_f0"].float()           # (T,)

        # -------- waveform --------
        wav, sr = torchaudio.load(row["wave"])
        resampler = self._get_resampler(sr)
        if resampler is not None:
            wav = resampler(wav)
        wav = (wav - self.mean) / (self.std + 1e-9)          # normalise

        return hubert, pitch, wav.squeeze(0)  # wav: (N,)


# ------------------------------------------------------------
#  Collate function                                            
# ------------------------------------------------------------

def data_processing(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Pad variable-length sequences and stack into a batch.

    Returns
    -------
    hubert_batch : (B, T_h_max, D)
    pitch_batch  : (B, T_h_max)
    wav_batch    : (B, N_wav_max)
    """

    B = len(batch)
    huberts, pitches, waves = zip(*batch)

    # time lengths
    th = [h.size(0) for h in huberts]
    twav = [w.size(0) for w in waves]
    T_h_max = max(th)
    N_wav_max = max(twav)

    D = huberts[0].size(1)

    h_pad = torch.zeros(B, T_h_max, D, dtype=huberts[0].dtype)
    p_pad = torch.zeros(B, T_h_max, dtype=pitches[0].dtype)
    w_pad = torch.zeros(B, N_wav_max, dtype=waves[0].dtype)

    for i in range(B):
        h_pad[i, : th[i]] = huberts[i]
        p_pad[i, : th[i]] = pitches[i]
        w_pad[i, : twav[i]] = waves[i]

    return h_pad, p_pad, w_pad

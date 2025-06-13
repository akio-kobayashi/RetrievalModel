from torch.utils.data import Dataset
from pathlib import Path
import torch
import pandas as pd
import random

def collate_alignment_rvc(batch):
    """
    入力: List of (src_hubert, src_pitch, tgt_hubert, tgt_pitch, mel)
    出力:
      - src_hubert_pad: (B, T_src_max, D)
      - src_pitch_pad : (B, T_src_max)
      - tgt_hubert_pad: (B, T_tgt_max, D)
      - tgt_pitch_pad : (B, T_tgt_max)
      - mel_pad       : (B, T_mel_max, 80)
    """
    src_h_list, src_p_list, tgt_h_list, tgt_p_list, mel_list = zip(*batch)
    B = len(batch)
    D = src_h_list[0].size(1)  # embedding dim = 768 or 256

    T_src_max = max(x.size(0) for x in src_h_list)
    T_tgt_max = max(x.size(0) for x in tgt_h_list)
    T_mel_max = max(x.size(0) for x in mel_list)

    src_hubert_pad = torch.zeros(B, T_src_max, D)
    src_pitch_pad  = torch.zeros(B, T_src_max)
    tgt_hubert_pad = torch.zeros(B, T_tgt_max, D)
    tgt_pitch_pad  = torch.zeros(B, T_tgt_max)
    mel_pad        = torch.zeros(B, T_mel_max, 80)

    for i, (sh, sp, th, tp, mel) in enumerate(batch):
        src_hubert_pad[i, :sh.size(0)] = sh
        src_pitch_pad[i, :sp.size(0)]  = sp
        tgt_hubert_pad[i, :th.size(0)] = th
        tgt_pitch_pad[i, :tp.size(0)]  = tp
        mel_pad[i, :mel.size(0)]       = mel

    return src_hubert_pad, src_pitch_pad, tgt_hubert_pad, tgt_pitch_pad, mel_pad

class KeySynchronizedDataset(Dataset):
    def __init__(
        self,
        align_csv: str | Path,
        mel_csv: str | Path,
        stats_tensor: dict,
        shuffle: bool = True,
        map_location: str = "cpu",
    ):
        # --- 読み込み ---
        align_df = pd.read_csv(align_csv)
        mel_df = pd.read_csv(mel_csv).set_index("key")

        self.align_map = {}
        self.mel_map = {}

        for _, row in align_df.iterrows():
            key = row.get("key")
            if key in mel_df.index:
                self.align_map[key] = {"source": row["source"], "target": row["target"]}
                self.mel_map[key] = mel_df.loc[key, "hubert"]

        # 共通keyのみに限定
        self.keys = list(set(self.align_map.keys()) & set(self.mel_map.keys()))
        if shuffle:
            random.shuffle(self.keys)

        self.pitch_mean = float(stats_tensor["pitch_mean"])
        self.pitch_std  = float(stats_tensor["pitch_std"]) + 1e-9
        self.mel_mean   = torch.tensor(stats_tensor["mel_mean"], dtype=torch.float).view(1, -1)
        self.mel_std    = torch.tensor(stats_tensor["mel_std"],  dtype=torch.float).view(1, -1) + 1e-9

        self.map_location = map_location

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]

        # ─ alignment data ─
        paths = self.align_map[key]
        src_pt = torch.load(paths["source"], map_location=self.map_location)
        tgt_pt = torch.load(paths["target"], map_location=self.map_location)
        src_hubert = src_pt["hubert"].float()
        src_pitch  = (src_pt["log_f0"].float() - self.pitch_mean) / self.pitch_std
        tgt_hubert = tgt_pt["hubert"].float()
        tgt_pitch  = (tgt_pt["log_f0"].float() - self.pitch_mean) / self.pitch_std

        # ─ mel data ─
        mel_pt = torch.load(self.mel_map[key], map_location=self.map_location)
        mel = mel_pt["mel"].float()
        if mel.size(0) == 80:
            mel = mel.transpose(0, 1)
        mel = (mel - self.mel_mean) / self.mel_std

        return src_hubert, src_pitch, tgt_hubert, tgt_pitch, mel

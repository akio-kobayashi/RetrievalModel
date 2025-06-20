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
        align_df = pd.read_csv(align_csv)
        mel_df = pd.read_csv(mel_csv)
        # 型をそろしておく
        align_df["key"] = align_df["key"].astype(str).str.strip()
        mel_df["key"]   = mel_df["key"].astype(str).str.strip()
        mel_df = mel_df.set_index("key")

        align_keys = set(align_df["key"])
        mel_keys   = set(mel_df.index)

        common_keys = align_keys & mel_keys
        missing_in_mel   = align_keys - mel_keys
        missing_in_align = mel_keys - align_keys

        print(f"[DEBUG] align keys: {len(align_keys)}")
        print(f"[DEBUG] mel   keys: {len(mel_keys)}")
        print(f"[DEBUG] common : {len(common_keys)}")
        if missing_in_mel:
            print(f"[WARN] {len(missing_in_mel)} keys from align.csv not in mel.csv: {list(missing_in_mel)[:5]} …")
        if missing_in_align:
            print(f"[WARN] {len(missing_in_align)} keys from mel.csv   not in align.csv: {list(missing_in_align)[:5]} …")

        # 共通キーのみで map を作り直し
        self.align_map = {}
        self.mel_map   = {}
        for key in common_keys:
            row = align_df[align_df["key"] == key].iloc[0]
            self.align_map[key] = {"source": row["source"], "target": row["target"]}
            self.mel_map[key]   = mel_df.loc[key, "hubert"]

        self.keys = list(common_keys)
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
        if mel.ndim == 3 and mel.shape[0] == 1:
            mel = mel.squeeze(0)
            
        if mel.shape[0] == 80 and mel.shape[1] != 80:
            mel = mel.transpose(0, 1)
        elif mel.shape[1] != 80:
            raise ValueError(f"unexpected shape {mel.shape}")
        
        mel = (mel - self.mel_mean) / self.mel_std

        return src_hubert, src_pitch, tgt_hubert, tgt_pitch, mel

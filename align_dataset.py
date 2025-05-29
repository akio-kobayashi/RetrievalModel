import csv
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


def collate_h2h(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Collate function to pad a batch of (src_hubert, src_pitch, tgt_hubert, tgt_pitch).
    Returns padded tensors:
      src_hubert_pad: (B, T_src_max, D)
      src_pitch_pad : (B, T_src_max)
      tgt_hubert_pad: (B, T_tgt_max, D)
      tgt_pitch_pad : (B, T_tgt_max)
    """
    src_h_list, src_p_list, tgt_h_list, tgt_p_list = zip(*batch)
    B = len(batch)
    # determine max lengths
    T_src_max = max(t.size(0) for t in src_h_list)
    T_tgt_max = max(t.size(0) for t in tgt_h_list)
    D = src_h_list[0].size(1)

    # initialize padded tensors
    src_hubert_pad = torch.zeros(B, T_src_max, D)
    src_pitch_pad  = torch.zeros(B, T_src_max)
    tgt_hubert_pad = torch.zeros(B, T_tgt_max, D)
    tgt_pitch_pad  = torch.zeros(B, T_tgt_max)

    for i, (sh, sp, th, tp) in enumerate(batch):
        src_hubert_pad[i, :sh.size(0)] = sh
        src_pitch_pad[i, :sp.size(0)]  = sp
        tgt_hubert_pad[i, :th.size(0)] = th
        tgt_pitch_pad[i, :tp.size(0)]  = tp

    return src_hubert_pad, src_pitch_pad, tgt_hubert_pad, tgt_pitch_pad


class Hubert2HubertDataset(Dataset):
    """
    Loads paired HuBERT features and pitch from .pt files.
    CSV must have columns 'source' and 'target', each pointing to a .pt file containing:
      - 'hubert': Tensor of shape (T, 768)
      - 'log_f0': Tensor of shape (T,)
    Returns tuples (src_hubert, src_pitch, tgt_hubert, tgt_pitch).
    """
    def __init__(self, csv_path: str or Path, map_location: str = 'cpu') -> None:
        self.map_location = map_location
        self.rows: List[dict] = []
        csv_path = Path(csv_path)
        with csv_path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'source' not in row or 'target' not in row:
                    raise ValueError("CSV must contain 'source' and 'target' columns")
                self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        # load source .pt dict
        src_dict = torch.load(row['source'], map_location=self.map_location)
        src_hubert = src_dict['hubert'].float()   # (T_src, 768)
        src_pitch  = src_dict['log_f0'].float()   # (T_src,)
        # load target .pt dict
        tgt_dict = torch.load(row['target'], map_location=self.map_location)
        tgt_hubert = tgt_dict['hubert'].float()   # (T_tgt, 768)
        tgt_pitch  = tgt_dict['log_f0'].float()   # (T_tgt,)
        return src_hubert, src_pitch, tgt_hubert, tgt_pitch


if __name__ == '__main__':
    # Example usage
    from torch.utils.data import DataLoader
    ds = Hubert2HubertDataset('path/to/pairs.csv')
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_h2h)
    src_h_pad, src_p_pad, tgt_h_pad, tgt_p_pad = next(iter(dl))
    print('Shapes:', src_h_pad.shape, src_p_pad.shape, tgt_h_pad.shape, tgt_p_pad.shape)

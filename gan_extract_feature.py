import os
import torch
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

from gan_feature_pipeline import get_mel_spectrogram  # 古い仕様のHiFiGAN互換

def replace_mel_in_tensor(wave_path, tensor_path, save_path):
    """
    HuBERT特徴量テンソル内の'mel'だけを差し替えて保存する
    """
    mel = get_mel_spectrogram(
        wave_path,
        resample_rate=22050,
        num_mels=80,
        n_fft=1024,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=8000
    )
    data = torch.load(tensor_path, map_location="cpu")
    data["mel"] = mel  # 差し替え
    torch.save(data, save_path)
    print(f"✓ Updated: {save_path.name}")

def main(args):
    df = pd.read_csv(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []

    for i, row in df.iterrows():
        wave_path = Path(row["wave"])
        orig_tensor_path = Path(row["hubert"])
        key = row.get("key", orig_tensor_path.stem)

        if not wave_path.exists():
            print(f"× Missing wave: {wave_path}")
            continue
        if not orig_tensor_path.exists():
            print(f"× Missing .pt: {orig_tensor_path}")
            continue

        new_tensor_path = output_dir / orig_tensor_path.name
        replace_mel_in_tensor(wave_path, orig_tensor_path, new_tensor_path)

        new_rows.append({
            "key": key,
            "wave": str(wave_path.resolve()),
            "hubert": str(new_tensor_path.resolve())
        })

    # 新CSVに保存
    new_csv_path = Path(args.new_csv)
    pd.DataFrame(new_rows).to_csv(new_csv_path, index=False)
    print(f"\n✔ Saved new CSV to: {new_csv_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="元のCSV（列: key, wave, hubert）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help=".ptを保存する先ディレクトリ")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="新しいCSVの保存パス")
    args = parser.parse_args()
    main(args)

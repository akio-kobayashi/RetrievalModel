import os
import torch
import pandas as pd
import torchaudio
from pathlib import Path
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

# --- HiFi-GAN互換のメルスペクトル抽出関数 -------------------
from gan_feature_pipeline import get_mel_spectrogram  # 古い仕様に準拠した関数

def replace_mel_in_tensor(wave_path, tensor_path, save_path=None):
    """
    既存の.ptファイルから'mel'のみ差し替えて保存する．
    """
    # 1. 22.05kHzのHiFi-GAN用logメルスペクトル計算
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

    # 2. 元テンソルをロード
    data = torch.load(tensor_path, map_location="cpu")
    
    # 3. melのみ差し替え
    data["mel"] = mel  # 形状: [n_mels, T]

    # 4. 保存
    save_path = save_path or tensor_path
    torch.save(data, save_path)
    print(f"✓ Updated: {Path(save_path).name}")

# -------------------------------------------------------------

def main(args):
    df = pd.read_csv(args.csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in df.iterrows():
        wave_path = Path(row["wave"])
        tensor_path = Path(row["hubert"])
        
        if not wave_path.exists():
            print(f"× Missing wave: {wave_path}")
            continue
        if not tensor_path.exists():
            print(f"× Missing tensor: {tensor_path}")
            continue

        save_path = out_dir / tensor_path.name
        replace_mel_in_tensor(wave_path, tensor_path, save_path)

# -------------------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="CSV with 'wave' and 'hubert' columns")
    parser.add_argument("--output_dir", type=str, default="./revised_features",
                        help="Directory to save updated tensors")
    args = parser.parse_args()
    main(args)

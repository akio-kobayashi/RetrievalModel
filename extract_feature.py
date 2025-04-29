from argparse import ArgumentParser
import yaml
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import glob
import os, sys
import torch
from transformers import HubertModel
import feature_pipeline as F

def process_and_save_features(
    filepath,
    save_path,
    hubert_model,
    hubert_layer_index=9,
    f0_method='harvest',
    eps=1e-5,
    hubert_sr=16000,
    mel_sr=22050,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80
):
    # 1. モノラル16kHz読み込み
    waveform, _ = F.load_and_resample(filepath, target_sr=hubert_sr)

    # 2. HuBERT特徴量抽出
    hubert_features = F.extract_hubert_features(waveform, hubert_model, hubert_layer_index)

    # 3. F0抽出＋NaN補間＋log変換
    log_f0 = F.extract_log_f0_world(waveform, sr=hubert_sr, method=f0_method, eps=eps)

    # 4. HuBERT特徴とF0の長さを揃える
    log_f0 = F.resample_f0_to_match_hubert(log_f0, len(hubert_features))

    # 5. HiFiGAN用の22.05kHzリサンプリング＆logメルスペクトル抽出
    mel = F.compute_log_mel_spectrogram(filepath, target_sr=mel_sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels)

    # 6. 辞書にまとめて保存
    data_to_save = {
        "hubert": hubert_features,  # [T, hidden_size]
        "log_f0": log_f0,            # [T]
        "mel": mel                   # [n_mels, T_mel]
    }
    torch.save(data_to_save, save_path)

    return hubert_features, log_f0

def main(args):
  model = HubertModel.from_pretrained('rinna/japanese-hubert-base')
  model.eval()

  df = pd.read_csv(args.wave_csv)
  for idx, row in df.iterrows():
    path = row['source']
    save_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(path))[0] + '.pt')
    if not os.path.exists(save_path):
        process_and_save_features(path, save_path, model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--wave_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./')

    args=parser.parse_args()
       
    main(args)

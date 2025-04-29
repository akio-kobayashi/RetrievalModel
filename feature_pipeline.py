import torch
import torchaudio
import pyworld as pw
import numpy as np

import subprocess
import os
import uuid

# 0. ラウドネス正規化
def normalize_audio(input_wav_path):
    # /tmpに一時ファイル名を作成（UUIDで衝突防止）
    output_wav_path = f"/tmp/{uuid.uuid4().hex}.wav"

    # ffmpegコマンド構築
    command = [
        "ffmpeg",
        "-y",  # 出力ファイルが既に存在していても上書き
        "-i", input_wav_path,
        "-af", "loudnorm",  # ラウドネス正規化
        output_wav_path
    ]

    # subprocessで実行
    subprocess.run(command, check=True)

    return output_wav_path

# 1. 音声読み込みとモノラル16kHzリサンプリング
def load_and_resample(filepath, target_sr=16000):
    normpath = normalize_audio(filepath)
    waveform, sr = torchaudio.load(normpath)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # モノラル化
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0), target_sr  # [T], sr

# 2. 日本語HuBERTを使って指定層出力を抽出
def extract_hubert_features(waveform, model, layer_index):
    with torch.no_grad():
        outputs = model(waveform.unsqueeze(0), output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (layer0, layer1, ..., layerN)
        features = hidden_states[layer_index].squeeze(0)  # remove batch dim
    return features

# 3. WorldでF0抽出＋NaN補間＋log変換
def extract_log_f0_world(waveform, sr=16000, method='harvest', eps=1e-5):
    x = waveform.cpu().numpy().astype(np.float64)
    
    if method == 'harvest':
        f0, _ = pw.harvest(x, sr)
    elif method == 'dio':
        f0, _ = pw.dio(x, sr)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    f0 = torch.from_numpy(f0).float()  # [T']
    
    # NaN補間
    isnan = torch.isnan(f0)
    if isnan.any():
        idx = torch.arange(f0.size(0))
        valid_idx = idx[~isnan]
        valid = f0[~isnan]
        if valid.numel() > 1:
            f0 = torch.interp(idx.float(), valid_idx.float(), valid)
        elif valid.numel() == 1:
            f0.fill_(valid.item())
        else:
            f0.fill_(0.0)
    
    # 無声（0Hz）対策 → 小さい値に置き換え
    f0[f0 <= 0] = eps
    
    # log変換
    log_f0 = torch.log(f0)

    return log_f0  # [T']

# 4. HuBERT特徴とF0の長さを合わせる
def resample_f0_to_match_hubert(f0, hubert_length):
    """
    F0系列をHuBERT特徴系列の長さに線形補間して合わせる関数．

    Args:
        f0 (Tensor): [T_f0] F0系列
        hubert_length (int): HuBERT特徴系列の長さ

    Returns:
        f0_resampled (Tensor): [T_hubert] 長さ揃えたF0系列
    """
    f0_np = f0.cpu().numpy()
    t_original = np.linspace(0, 1, len(f0_np))
    t_target = np.linspace(0, 1, hubert_length)
    f0_interp = np.interp(t_target, t_original, f0_np)
    return torch.from_numpy(f0_interp).to(f0.device).float()    

# 5. HiFiGAN条件でメルスペクトルを計算（22.05kHz, 80メル, logスケール）
def compute_log_mel_spectrogram(filepath, target_sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80):
    waveform, sr = torchaudio.load(filepath)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=0,
        f_max=target_sr // 2,
        power=1.0,
        normalized=False
    )
    mel = mel_spec_transform(waveform)  # [1, n_mels, T]
    mel = torch.log(torch.clamp(mel, min=1e-5))  # logメル
    return mel.squeeze(0)  # [n_mels, T]

# 6. F0系列のグローバルmean/std計算（NaN除去済み前提）
def compute_global_f0_stats(f0_list):
    if isinstance(f0_list[0], np.ndarray):
        all_f0 = torch.from_numpy(np.concatenate(f0_list, axis=0)).float()
    else:
        all_f0 = torch.cat(f0_list, dim=0).float()
    
    valid_f0 = all_f0[torch.logical_and(~torch.isnan(all_f0), all_f0 > 0)]
    mean = valid_f0.mean().item()
    std = valid_f0.std().item()
    return mean, std

# 7. メルスペクトルのグローバルmean/std計算
def compute_global_mel_stats(mel_list):
    all_mels = torch.cat([mel for mel in mel_list], dim=1)  # [n_mels, total_T]
    mean = all_mels.mean(dim=1)  # [n_mels]
    std = all_mels.std(dim=1)    # [n_mels]
    return mean, std

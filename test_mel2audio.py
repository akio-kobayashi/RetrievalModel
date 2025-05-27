#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert mel-spectrogram tensors ("mel" key) stored in HuBERT-feature .pt files 
referenced by a CSV (hubert column) to 22.05-kHz waveforms using NVIDIA/HiFi-GAN.
"""
from __future__ import annotations

import csv, os, warnings
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------

def load_hifigan(device: torch.device):
    hifigan, train_setup, denoiser = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", "nvidia_hifigan"
    )
    return hifigan.to(device), train_setup, denoiser.to(device)

# ---------------------------------------------------------------------------

@torch.no_grad()
def mel2wav(
    mel: torch.Tensor,
    hifigan,
    denoiser,
    max_wav_value: float,
    denoise: float,
    wav_path: Path,
):
    """
    • mel : tensor shape (T, 80)  or  (80, T)
    • Saves as wav_path
    """
    if mel.ndim != 2:
        raise ValueError(f"expected 2-D tensor, got {mel.shape}")

    if mel.size(1) == 80:             # (T,80)
        mel = mel.transpose(0, 1)     # → (80,T)

    if mel.size(0) != 80:
        raise ValueError(f"mel dim != 80 after transpose: {mel.shape}")

    mel = mel.unsqueeze(0)            # (1,80,T)
    audio = hifigan(mel).squeeze(1)   # (1,T')

    if denoise > 0:
        audio = denoiser(audio, denoise)

    audio = torch.clamp(audio, -1.0, 1.0) * max_wav_value
    #print(audio.shape)
    torchaudio.save(wav_path, audio.squeeze(0).short().cpu(), 22050)
    print("✓", wav_path)

# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hifigan, setup, denoiser = load_hifigan(device)
    max_wav_value = setup["max_wav_value"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        for row in rows:
            pt_path = Path(row["hubert"])
            tens = torch.load(pt_path, map_location=device)
            mel = tens["mel"].float()

            wav_path = out_dir / (pt_path.stem + ".wav")

            mel2wav(
                mel,
                hifigan,
                denoiser,
                max_wav_value,
                args.denoising_strength,
                wav_path,
            )

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = ArgumentParser(description="Convert mel stored in Hubert-pt referenced by CSV → wav using HiFi-GAN")
    parser.add_argument("--csv", type=str, required=True, help="CSV with 'hubert' column")
    parser.add_argument("--output_dir", type=str, default="./wav_out", help="output directory")
    parser.add_argument("--denoising_strength", type=float, default=0.005)
    args = parser.parse_args()

    main(args)
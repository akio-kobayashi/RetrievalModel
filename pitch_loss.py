import torch.nn.functional as F
import torch.nn as nn
import torch

class PitchLoss(nn.Module):
  def __init__(self, sr=16000, frame_size=512, hop_size=256, f0_min=80, f0_max=500):
    super().__init__()
    self.sr = sr
    self.frame_size = frame_size
    self.hop_size = hop_size
    self.f0_min = f0_min
    self.f0_max = f0_max

  def autocorr_pitch(self, wav):
    num_frames = (wav.shape[-1] - self.frame_size) // self.hop_size + 1
    frames = []
    for i in range(num_frames):
      start = i * self.hop_size
      end = start + self.frame_size
      frame = wav[..., start:end] * torch.hann_window(self.frame_size, device=wav.device)
      frames.append(frame)
    frames = torch.stack(frames, dim=1)  # (B, T, F)

    # naive autocorrelation by FFT
    fft = torch.fft.rfft(frames, dim=-1)
    power = fft * torch.conj(fft)
    acf = torch.fft.irfft(power, n=self.frame_size, dim=-1)

    # pitch range
    min_lag = self.sr // self.f0_max
    max_lag = self.sr // self.f0_min
    acf_range = acf[..., min_lag:max_lag]

    pitch_period = torch.argmax(acf_range, dim=-1) + min_lag
    f0 = self.sr / pitch_period.float()
    logf0 = torch.log(f0 + 1e-6)
    return logf0

  def forward(self, wav_fake, wav_real):
    logf0_fake = self.autocorr_pitch(wav_fake)
    logf0_real = self.autocorr_pitch(wav_real)
    voiced_mask = (torch.exp(logf0_real) > 1e-2).float()
    logf0_loss = F.l1_loss(logf0_fake * voiced_mask, logf0_real * voiced_mask)
    return logf0_loss

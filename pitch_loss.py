import torchcrepe   # 例: CREPE を使ってピッチ抽出
import torch.nn.functional as F
import torch.nn as nn
import torch

class PitchLoss(nn.Module):
  def __init__(self, sr=16_000, hop=256):
    super().__init__()
    self.sr = sr
    self.hop = hop

  def forward(self, wav_fake, wav_real):
    # 2) log f₀ を抽出（フレーム単位）
    f0_real = torchcrepe.predict(wav_real.float(), self.sr, hop_length=self.hop, model="tiny")  # (B, F)
    f0_fake = torchcrepe.predict(wav_fake.float(), self.sr, hop_length=self.hop, model="tiny")  # (B, F)
    logf0_real = torch.log(f0_real + 1.e-6)
    logf0_fake = torch.log(f0_fake + 1.e-6)

    # 3) Voiced/unvoiced のマスク（CREPE が confidence を返す場合など）
    mask_voiced = (logf0_real > 0).float()

    # 4) ピッチ損失（smooth L1 または MSE）
    loss_pitch = F.smooth_l1_loss(logf0_fake * mask_voiced,
                                  logf0_real * mask_voiced,
                                  reduction="sum") / (mask_voiced.sum() + 1e-8)
    return loss_pitch

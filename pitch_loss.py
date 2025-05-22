import torchcrepe   # 例: CREPE を使ってピッチ抽出
import torch.nn.functional as F

class PitchLoss(nn.Module):
  def __init__(self, sr=16_000, hop=256):
    self.sr = sr
    self.hop = hop

  def forward(self, wav_fake, wav_real):
    # 2) log f₀ を抽出（フレーム単位）
    logf0_real = torchcrepe.predict(wav_real, self.sr, hop_length=self.hop, model="full").log_f0  # (B, F)
    logf0_fake = torchcrepe.predict(wav_fake, self.sr, hop_length=self.hop, model="full").log_f0  # (B, F)

    # 3) Voiced/unvoiced のマスク（CREPE が confidence を返す場合など）
    mask_voiced = (logf0_real > 0).float()

    # 4) ピッチ損失（smooth L1 または MSE）
    loss_pitch = F.smooth_l1_loss(logf0_fake * mask_voiced,
                                  logf0_real * mask_voiced,
                                  reduction="sum") / (mask_voiced.sum() + 1e-8)
    return loss_pitch

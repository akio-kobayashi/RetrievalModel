import torch
from melsolver import MelVCSystem  # 例：定義済みのクラス
from pathlib import Path
import argparse

def convert(args):
    ckpt_path = args.ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # MelVCSystemの構成と一致するように初期化（メモリを食わない）
    model = MelVCSystem()
    model.load_state_dict(ckpt["state_dict"], strict=False)

    # gen（= RVCStyleVC）だけ保存
    torch.save(model.gen.state_dict(), args.output)

if __name__ == "__main__":
  torch.set_float32_matmul_precision('high')
  parser = argparse.ArgumentParser(description="Convert RVC-style VC model")
  parser.add_argument("--ckpt", type=str, help="Path to check point file")
  parser.add_argument("--output", type=str, help="Path to output state_dict file")
  args = parser.parse_args()

  convert(args)

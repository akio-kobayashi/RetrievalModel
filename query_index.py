from argparse import ArgumentParser
import warnings, os
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from hubert_retriever import IVFPQFaissRetriever, load_query_tensor, replace_hubert_in_tensor

# ------------------------------------------------------------
#  cosine similarity along T dimension                        
# ------------------------------------------------------------

def cosine_similarity(x: torch.Tensor, y: torch.Tensor):
    """x, y: (T, D) → return mean cosine sim (scalar)."""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x * y).sum(dim=-1).mean()

# ------------------------------------------------------------
#  main                                                       
# ------------------------------------------------------------

def main(args):
    # ----- load retriever -----
    retr = IVFPQFaissRetriever(
        index_path=args.index,
        memmap_path=args.memmap,
        metadata_path=args.metadata,
        gpu_device=args.gpu,
    )

    # ----- load query tensor (T,D) -----
    q = torch.load(args.file, map_location="cpu")["hubert"].float()  # (T,D)
    print("Query shape:", q.shape)

    # ----- frame‑wise search ---------------
    similarities = []
    topk_vecs   = []  # store top‑1 vec for optional replacement

    for t, frame in enumerate(q):
        vecs_np, _ = retr.search(frame.unsqueeze(0), topk=args.topk, batch_first=True)  # (1, topk, D)
        vecs_t = torch.from_numpy(vecs_np[0])  # (topk,D)
        topk_vecs.append(vecs_t[0])            # save top1
        sim_t = cosine_similarity(frame, vecs_t[0])
        similarities.append(sim_t.item())
        if (t + 1) % 100 == 0:
            print(f" processed {t+1}/{len(q)} frames…", end="
")

    top1_mean_sim = sum(similarities) / len(similarities)
    print(f"
Mean cosine (top‑1): {top1_mean_sim:.4f}")

    # ----- histogram -----
    fig_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.file))[0] + "_sim_hist.png")
    plt.figure(figsize=(6, 4))
    plt.hist(similarities, bins=40, color="steelblue", edgecolor="black")
    plt.xlabel("Cosine similarity (top-1)")
    plt.ylabel("Frame count")
    plt.title("Distribution of frame‑wise similarities")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print("Histogram saved →", fig_path)

    # ----- save replaced tensor (stack of top1 vecs) -----
    new_hubert = torch.stack(topk_vecs)  # (T,D)
    dst_pt = os.path.join(args.output_dir, os.path.basename(args.file))
    replace_hubert_in_tensor(args.file, new_hubert.cpu(), dst_pt)
    print("Replaced tensor saved →", dst_pt)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--file", required=True, help="query .pt file with HuBERT tensor")
    ap.add_argument("--memmap", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--output_dir", default="./out")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)

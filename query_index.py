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

    # ----- batch search -----
    vecs_np, infos_all = retr.search(q, topk=args.topk, batch_first=True)  # (T, topk, D)  # len=T, each (vecs(topk,D), infos)

    # stack to (topk, T, D)
    vecs = torch.from_numpy(vecs_np).permute(1, 0, 2)  # (topk, T, D)
    infos = list(zip(*infos_all))  # len=topk, each T dicts  # list length T
    print("Retrieved shape:", vecs.shape)

    # ----- similarity per k -----
    similarities = []
    for k in range(args.topk):
        sim = cosine_similarity(q, vecs[k])
        similarities.append(sim.item())
        print(f"[top{k+1}] mean cosine = {sim:.4f}")

    # ----- histogram -----
    fig_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.file))[0] + "_sim_hist.png")
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, args.topk + 1), similarities, color="steelblue")
    plt.xlabel("k-th neighbour")
    plt.ylabel("Mean Cosine Similarity")
    plt.title("Average similarity per rank")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print("Histogram saved →", fig_path)

    # ----- save top‑1 replaced tensor -----
    dst_pt = os.path.join(args.output_dir, os.path.basename(args.file))
    replace_hubert_in_tensor(args.file, vecs[0].cpu(), dst_pt)
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

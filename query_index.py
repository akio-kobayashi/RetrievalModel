from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from hubert_retriever import IVFPQFaissRetriever
import hubert_retriever as H

def cosine_similarity(x, y):
    """x: (T, D), y: (T, D)"""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x * y).sum(dim=-1).mean()  # 平均類似度

def main(args):
    retr = IVFPQFaissRetriever(
        index_path=args.index,
        memmap_path=args.memmap,
        metadata_path=args.metadata,
    )
    
    q = H.load_query_tensor(args.file)  # (T, D)
    print("Query shape:", q.shape)
    
    vecs, infos = retr.search(q, topk=args.topk)  # vecs: (topk, T, D)
    vecs = torch.from_numpy(vecs)
    print("Retrieved shape:", vecs.shape)

    similarities = []
    for i in range(args.topk):
        sim = cosine_similarity(q, vecs[i])
        similarities.append(sim.item())
        print(f"[{i}] Similarity: {sim.item():.4f} - Info: {infos[i]}")

    # 類似度のヒストグラム表示
    plt.figure(figsize=(6, 4))
    plt.hist(similarities, bins=10, range=(0.8, 1.0), edgecolor='black')
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("Similarity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 出力保存（任意）
    dst_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.file))[0] + '.pt')
    H.replace_hubert_in_tensor(args.file, vecs.cpu(), dst_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--file', type=str)
    parser.add_argument('--memmap', type=str, default='./memmap')
    parser.add_argument('--index', type=str, default='./index')
    parser.add_argument('--metadata', type=str, default='./metadata')
    parser.add_argument('--topk', type=int, default=1)
    args = parser.parse_args()
    main(args)

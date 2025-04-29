# 再実行（セッションリセットのため）
import numpy as np
import faiss
import torch
import pickle
from typing import Literal, List
import os
from pathlib import Path
from tqdm import tqdm

NormalizationStrategy = Literal["none", "l2", "cosine"]


def normalize_features_np(features: np.ndarray, strategy: NormalizationStrategy) -> np.ndarray:
    if strategy == "none":
        return features
    elif strategy in ("l2", "cosine"):
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        return features / norms
    else:
        raise ValueError(f"Unsupported normalization strategy: {strategy}")


def save_features_as_memmap(features: np.ndarray, memmap_path: str):
    fp = np.memmap(memmap_path, dtype='float32', mode='w+', shape=features.shape)
    fp[:] = features[:]
    del fp  # flush to disk


def save_metadata(metadata: List[dict], path: str):
    with open(path, 'wb') as f:
        pickle.dump(metadata, f)


class IVFPQFaissIndexer:
    def __init__(
        self,
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8,
        strategy: NormalizationStrategy = "cosine"
    ):
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.strategy = strategy

    def build_index(
        self,
        features: torch.Tensor,             # [N, D]
        metadata: List[dict],               # [{"file": str, "frame": int}, ...]
        memmap_path: str,
        index_path: str,
        metadata_path: str,
        gpu_device: int = 0
    ):
        features_np = features.cpu().float().numpy()
        save_features_as_memmap(features_np, memmap_path)
        save_metadata(metadata, metadata_path)

        norm_features = normalize_features_np(features_np, self.strategy)
        d = norm_features.shape[1]

        # Faiss: build and train IVFPQ index
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.m, self.nbits)

        assert index.is_trained is False, "Index should not be pre-trained."

        # Train on normalized features
        print("Training Faiss IndexIVFPQ...")
        index.train(norm_features)  # ← これが「トレーニング」

        # Move to GPU
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_device, index)

        # Add features
        gpu_index.add(norm_features)

        # Save CPU index (converted from GPU)
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), index_path)
        print(f"Index saved to {index_path}")

def load_hubert_features_from_tensor_files(
    directory: str,
    key: str = "hubert"
) -> tuple[torch.Tensor, List[dict]]:
    """
    ディレクトリ内のtensorファイルからhubert特徴量を読み込み、1フレームごとに分解して結合。
    Returns:
        features: torch.Tensor [N, D]
        metadata: List[dict] 対応ファイル名・フレーム番号
    """
    features = []
    metadata = []
    files = sorted(Path(directory).glob("*.pt"))

    for f in tqdm(files, desc="Loading tensor files"):
        data = torch.load(f, map_location="cpu")
        if key not in data:
            continue
        hubert = data[key]  # [T, D]
        for t in range(hubert.size(0)):
            features.append(hubert[t])
            metadata.append({"file": f.name, "frame": t})

    features_tensor = torch.stack(features)  # [N, D]
    return features_tensor, metadata


def index_from_tensor_directory(
    tensor_dir: str,
    memmap_path: str,
    index_path: str,
    metadata_path: str,
    key: str = "hubert",
    strategy: Literal["none", "l2", "cosine"] = "cosine",
    nlist: int = 100,
    m: int = 8,
    nbits: int = 8,
    gpu_device: int = 0
):
    """
    ディレクトリ内の.ptファイル群からhubert特徴を抽出し、インデックスとmemmapを保存する。
    """
    features, metadata = load_hubert_features_from_tensor_files(tensor_dir, key=key)
    indexer = IVFPQFaissIndexer(nlist=nlist, m=m, nbits=nbits, strategy=strategy)
    indexer.build_index(
        features=features,
        metadata=metadata,
        memmap_path=memmap_path,
        index_path=index_path,
        metadata_path=metadata_path,
        gpu_device=gpu_device
    )

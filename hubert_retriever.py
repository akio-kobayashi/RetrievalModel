import numpy as np
import faiss
import torch
import pickle
from typing import List, Literal, Tuple
from pathlib import Path

NormalizationStrategy = Literal["none", "l2", "cosine"]


def load_memmap(memmap_path: str, shape: Tuple[int, int]) -> np.memmap:
    return np.memmap(memmap_path, dtype='float32', mode='r', shape=shape)


def load_metadata(metadata_path: str) -> List[dict]:
    with open(metadata_path, 'rb') as f:
        return pickle.load(f)


class IVFPQFaissRetriever:
    def __init__(
        self,
        index_path: str,
        memmap_path: str,
        memmap_shape: Tuple[int, int],
        metadata_path: str,
        strategy: NormalizationStrategy = "cosine",
        gpu_device: int = 0
    ):
        self.strategy = strategy
        self.mmap = load_memmap(memmap_path, memmap_shape)
        self.metadata = load_metadata(metadata_path)

        index_cpu = faiss.read_index(index_path)
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, gpu_device, index_cpu)

    def normalize_features_np(self, features: np.ndarray) -> np.ndarray:
        if self.strategy == "none":
            return features
        elif self.strategy in ("l2", "cosine"):
            norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
            return features / norms
        else:
            raise ValueError(f"Unsupported normalization strategy: {self.strategy}")

    def search(
        self,
        queries: torch.Tensor,  # [B, D]
        topk: int = 5
    ) -> List[Tuple[np.ndarray, List[dict]]]:
        queries_np = queries.cpu().float().numpy()
        queries_np = self.normalize_features_np(queries_np)

        _, indices = self.index.search(queries_np.astype('float32'), topk)

        results = []
        for idx_list in indices:
            retrieved_vectors = np.stack([self.mmap[i] for i in idx_list])  # [topk, D]
            retrieved_info = [self.metadata[i] for i in idx_list]
            results.append((retrieved_vectors, retrieved_info))

        return results

def load_query_features_from_tensor(
    filepath: str,
    key: str = "hubert"
) -> torch.Tensor:
    """
    指定されたファイルからクエリ用のHuBERT特徴量を取得（平均ベクトルに変換）
    """
    data = torch.load(filepath, map_location="cpu")
    if key not in data:
        raise KeyError(f"'{key}' not found in {filepath}")
    hubert = data[key]  # [T, D]
    return hubert.mean(dim=0, keepdim=True)  # [1, D]


def query_similar_features(
    query_file: str,
    index_path: str,
    memmap_path: str,
    memmap_shape: Tuple[int, int],
    metadata_path: str,
    strategy: Literal["none", "l2", "cosine"] = "cosine",
    topk: int = 5,
    gpu_device: int = 0
):
    """
    クエリファイルから特徴を抽出し、Retrieverを使って近傍特徴とメタデータを返す。
    """
    query_vector = load_query_features_from_tensor(query_file, key="hubert")  # [1, D]

    retriever = IVFPQFaissRetriever(
        index_path=index_path,
        memmap_path=memmap_path,
        memmap_shape=memmap_shape,
        metadata_path=metadata_path,
        strategy=strategy,
        gpu_device=gpu_device
    )

    results = retriever.search(query_vector, topk=topk)

    # 1件のクエリしか処理していないので結果は1件だけ
    retrieved_vectors, retrieved_metadata = results[0]
    return retrieved_vectors, retrieved_metadata

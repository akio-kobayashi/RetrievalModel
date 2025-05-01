import numpy as np
import faiss
import torch
import pickle
from typing import Literal, List
from pathlib import Path
from tqdm import tqdm

NormalizationStrategy = Literal["none", "l2", "cosine"]

# ------------------------------------------------------------
#  Utility                                                     
# ------------------------------------------------------------

def normalize_features_np(features: np.ndarray, strategy: NormalizationStrategy) -> np.ndarray:
    """Return *copy* of features if normalized, otherwise original ref."""
    if strategy == "none":
        return features
    elif strategy in ("l2", "cosine"):
        n = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        return features / n
    raise ValueError(strategy)


def save_features_as_memmap(features: np.ndarray, memmap_path: str):
    fp = np.memmap(memmap_path, dtype="float32", mode="w+", shape=features.shape)
    fp[:] = features  # direct slice copy → flush when del
    del fp


def save_metadata(metadata: List[dict], path: str):
    with open(path, "wb") as f:
        pickle.dump(metadata, f)

# ------------------------------------------------------------
#  FAISS Indexer                                               
# ------------------------------------------------------------

class IVFPQFaissIndexer:
    """IVFPQ index builder with minimal RAM/VRAM footprint.

    Key changes vs. previous version:
    * **train()** uses `sample_rate = nlist*256` (or full size if smaller)
    * **add()** is processed in manageable batches (default 100k)
    * optional **float16 cloning** for lower VRAM
    """

    def __init__(
        self,
        nlist: int = 1024,
        m: int = 16,
        nbits: int = 8,
        strategy: NormalizationStrategy = "cosine",
        batch_size: int = 100_000,
        use_fp16: bool = True,
    ):
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.strategy = strategy
        self.batch_size = batch_size
        self.use_fp16 = use_fp16

    # ----------------------- main API -----------------------
    def build_index(
        self,
        features: torch.Tensor,  # [N,D] on CPU
        metadata: List[dict],
        memmap_path: str,
        index_path: str,
        metadata_path: str,
        gpu_device: int = 0,
    ):
        # ---------- persist raw features (non‑normalised) ----------
        features_np = features.numpy().astype("float32", copy=False)
        save_features_as_memmap(features_np, memmap_path)
        save_metadata(metadata, metadata_path)

        # ---------- normalise for distance metric ----------
        norm_feats = normalize_features_np(features_np, self.strategy)
        d = norm_feats.shape[1]

        # ---------- create IVFPQ index ----------
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.m, self.nbits)

        # ---------- (1) train with subsample ----------
        n_train = min(self.nlist * 256, norm_feats.shape[0])
        perm = np.random.permutation(norm_feats.shape[0])[:n_train]
        print(f"Training IVFPQ on {n_train:,} vectors…")
        index.train(norm_feats[perm])

        # ---------- (2) move index to GPU ----------
        res = faiss.StandardGpuResources()
        cloner_opts = faiss.GpuClonerOptions()
        cloner_opts.useFloat16 = self.use_fp16
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_device, index, cloner_opts)

        # ---------- (3) add in batches to save VRAM ----------
        N = norm_feats.shape[0]
        bs = self.batch_size
        for start in tqdm(range(0, N, bs), desc="Adding to index"):
            end = min(start + bs, N)
            gpu_index.add(norm_feats[start:end])

        # ---------- (4) save CPU copy ----------
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), index_path)
        print(f"Index saved → {index_path}")

# ------------------------------------------------------------
#  Tensor directory helper (unchanged except minor speed tweak)
# ------------------------------------------------------------

def load_hubert_features_from_tensor_files(directory: str, key: str = "hubert") -> tuple[torch.Tensor, List[dict]]:
    feats_list, meta_list = [], []
    for p in tqdm(sorted(Path(directory).glob("*.pt"))):
        data = torch.load(p, map_location="cpu")
        if key not in data:
            continue
        hubert = data[key].float()  # [T,D]
        feats_list.append(hubert)
        # metadata: vectorised追加
        meta_list.extend({"file": p.name, "frame": i} for i in range(hubert.size(0)))
    features_tensor = torch.cat(feats_list, dim=0)  # [N,D]
    return features_tensor, meta_list

# ------------------------------------------------------------
#  End-to-end wrapper
# ------------------------------------------------------------

def index_from_tensor_directory(
    tensor_dir: str,
    memmap_path: str,
    index_path: str,
    metadata_path: str,
    key: str = "hubert",
    strategy: NormalizationStrategy = "cosine",
    nlist: int = 512,
    m: int = 8,
    nbits: int = 8,
    gpu_device: int = 0,
    indexer_kwargs = dict(
        batch_size = 10_000,
        use_fp16 = True,
        train_sample_mut = 128,
    )
):
    feats, meta = load_hubert_features_from_tensor_files(tensor_dir, key)
    indexer = IVFPQFaissIndexer(nlist=nlist, m=m, nbits=nbits, strategy=strategy)
    indexer.build_index(
        features=feats,
        metadata=meta,
        memmap_path=memmap_path,
        index_path=index_path,
        metadata_path=metadata_path,
        gpu_device=gpu_device,
    )

import numpy as np
import faiss
import torch
import pickle
from typing import Literal, List, Tuple
from pathlib import Path
from tqdm import tqdm
import time
import psutil

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
        train_sample_mult: int = 256,
        use_fp16: bool = False
    ):
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.strategy = strategy
        self.batch_size = batch_size
        self.train_sample_mult = train_sample_mult
        self.use_fp16 = use_fp16

    # ----------------------- main API -----------------------
    def build_index(
        self,
        features: torch.Tensor,          # [N,D] on CPU
        metadata: List[dict],
        memmap_path: str,
        index_path: str,
        metadata_path: str,
    ):
        t0 = time.time()
        # 1) memmap 保存
        feats_np = features.numpy().astype("float32", copy=False)
        np.memmap(memmap_path, "float32", "w+", shape=feats_np.shape)[:] = feats_np
        save_metadata(metadata, metadata_path)
        del features  # RAM 釈放
        N, D = feats_np.shape
        print(f"[1/4] memmap+metadata done → {memmap_path} | vec {N:,} dim {D}")

        # 2) index & train
        train_n = min(self.nlist * self.train_sample_mult, N)
        samp = np.random.choice(N, train_n, replace=False)
        train_mat = normalize_features_np(feats_np[samp], self.strategy)
        index = faiss.IndexIVFPQ(faiss.IndexFlatL2(D), D, self.nlist, self.m, self.nbits)
        index.train(train_mat)
        print("[2/4] train finished on", train_n, "vecs")

        # 3) add in batches
        bs = self.batch_size
        bar = tqdm(range(0, N, bs), desc="Adding", unit="vec",
                    total=(N + bs - 1)//bs,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        for s in bar:
            e = min(s + bs, N)
            chunk = normalize_features_np(feats_np[s:e], self.strategy)
            index.add(chunk)
            bar.set_postfix(mem=f"{psutil.virtual_memory().percent}%")
        del feats_np
        print("[3/4] add complete, saving index…")

        # 4) save
        faiss.write_index(index, index_path)
        print(f"[4/4] index saved → {index_path} | elapsed {time.time()-t0:.1f}s")


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
    )

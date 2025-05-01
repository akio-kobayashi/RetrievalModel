import numpy as np
import faiss
import torch
import pickle
from typing import List, Literal, Tuple, Optional
from pathlib import Path

NormalizationStrategy = Literal["none", "l2", "cosine"]

# ------------------------------------------------------------
#  Utility                                                    
# ------------------------------------------------------------

def _normalize_np(x: np.ndarray, strategy: NormalizationStrategy) -> np.ndarray:
    if strategy == "none":
        return x
    if strategy in ("l2", "cosine"):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    raise ValueError(strategy)


def _load_metadata(path: str) -> List[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)
    
# ------------------------------------------------------------
#  Tensor key replacement helper                              
# ------------------------------------------------------------

def replace_hubert_in_tensor(
    src_path: str,
    new_tensor: torch.Tensor,
    dst_path: Optional[str] = None,
    key: str = "hubert",
):
    """Load a .pt file, replace its `key` tensor with `new_tensor`, and save.

    Args:
        src_path: input .pt file path.
        new_tensor: tensor to overwrite `key`.
        dst_path: if None, overwrite *in‑place*; otherwise save to new file.
        key: tensor entry name to replace (default "hubert").
    """
    data = torch.load(src_path, map_location="cpu")
    data[key] = new_tensor
    torch.save(data, dst_path or src_path)
    
# ------------------------------------------------------------
#  Retriever                                                  
# ------------------------------------------------------------

class IVFPQFaissRetriever:
    """Retrieve nearest neighbours from IVFPQ index built by `faiss_indexer.py`."""

    def __init__(
        self,
        index_path: str,
        memmap_path: str,
        metadata_path: str,
        strategy: NormalizationStrategy = "cosine",
        gpu_device: int = -1,
        fp16: bool = True,
        memmap_shape: Optional[Tuple[int, int]] = None,
    ):
        self.strategy = strategy

        # ---------- load CPU index ----------
        index_cpu = faiss.read_index(index_path)
        d = index_cpu.d
        n = index_cpu.ntotal

        # ---------- memmap of raw (non-normalised) features ----------
        if memmap_shape is None:
            memmap_shape = (n, d)
        self.mmap = np.memmap(memmap_path, dtype="float32", mode="r", shape=memmap_shape)

        # ---------- metadata ----------
        self.metadata = _load_metadata(metadata_path)
        assert len(self.metadata) == n, "metadata length mismatch with index size"

        # ---------- move index to GPU ----------
        res = faiss.StandardGpuResources()
        cloner_opts = faiss.GpuClonerOptions()
        cloner_opts.useFloat16 = fp16

        if gpu_device is not None and gpu_device >=0 and gpu_device < faiss.get_num_gpus():
            self.index = faiss.index_cpu_to_gpu(res, gpu_device, index_cpu, cloner_opts)
        else:
            self.index = index_cpu
            
    # --------------------------------------------------------
    def search(self, queries: torch.Tensor, topk: int = 5, batch_first: bool = True):
        """Search nearest neighbours.

        Args:
            queries (Tensor): (B, D) HuBERT (or same) features on *any* device.
            topk (int): number of neighbours.
        Returns:
            List[Tuple[np.ndarray, List[dict]]]: one list per query.
        """
        q = queries.detach().cpu().float().numpy()
        q = _normalize_np(q, self.strategy).astype("float32", copy=False)

        if batch_first:
          vecs  = self.mmap[idxs]                               # (B, topk, D)
          infos = [[self.metadata[i] for i in row] for row in idxs]
          return vecs, infos
        else:
          results = []
          for row in idxs:
            vecs_row  = self.mmap[row]                        # (topk, D)
            infos_row = [self.metadata[i] for i in row]
            results.append((vecs_row, infos_row))
          return results

# ------------------------------------------------------------
#  Helper: CLI-like single‑query convenience                   
# ------------------------------------------------------------

def load_query_tensor(file: str, key: str = "hubert") -> torch.Tensor:
    data = torch.load(file, map_location="cpu")
    if key not in data:
        raise KeyError(f"{key} not in {file}")
    return data[key].float()
    #return data[key].mean(0, keepdim=True)  # (1,D)


def query_similar_features(
    query_file: str,
    index_path: str,
    memmap_path: str,
    metadata_path: str,
    strategy: NormalizationStrategy = "cosine",
    topk: int = 5,
    gpu_device: int = -1,
):
    q = load_query_tensor(query_file)
    retr = IVFPQFaissRetriever(
        index_path=index_path,
        memmap_path=memmap_path,
        metadata_path=metadata_path,
        strategy=strategy,
        gpu_device=gpu_device,
    )
    (vecs, infos), = retr.search(q, topk=topk)  # unpack single query
    return vecs, infos

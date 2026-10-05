import warnings

import numpy as np
import torch

try:
    import faiss
except ImportError:  # Loaded only when UMAP KNN is used.
    faiss = None

@torch.no_grad()
def torch_neighbor_search(train_latents, query_latents, k=10, batch_size=2048):
    if not isinstance(train_latents, torch.Tensor) or not isinstance(query_latents, torch.Tensor):
        raise TypeError("VAE KNN expects torch.Tensors")
    if query_latents.ndim == 1:
        query_latents = query_latents.unsqueeze(0)
    if train_latents.ndim != 2 or query_latents.ndim != 2:
        raise ValueError("train_latents and query_latents must be 2D: [N,D] and [M,D]")
    if train_latents.shape[1] != query_latents.shape[1]:
        raise ValueError(f"Dim mismatch: train D={train_latents.shape[1]}, query D={query_latents.shape[1]}")
    if train_latents.device != query_latents.device:
        raise ValueError("train_latents and query_latents must be on the same device")

    out_d, out_i = [], []
    for s in range(0, query_latents.shape[0], batch_size):
        q = query_latents[s:s + batch_size]
        dists = torch.cdist(q, train_latents)
        d, i = torch.topk(dists, k, largest=False, dim=1)
        out_d.append(d)
        out_i.append(i)
    return (torch.cat(out_d, 0).detach().cpu().numpy(),
        torch.cat(out_i, 0).detach().cpu().numpy())


def faiss_flat_l2_search(umap_train, umap_query, k=50, return_distance=False, batch_size=None, gpu_id=None):
    if hasattr(umap_train, "detach"):
        umap_train = umap_train.detach().cpu().numpy()
    if hasattr(umap_query, "detach"):
        umap_query = umap_query.detach().cpu().numpy()

    umap_query = np.asarray(umap_query)
    if umap_query.ndim == 1:
        umap_query = umap_query[None, :]

    umap_train = np.ascontiguousarray(np.asarray(umap_train, dtype=np.float32))
    umap_query = np.ascontiguousarray(np.asarray(umap_query, dtype=np.float32))

    N, D = umap_train.shape
    if umap_query.shape[1] != D:
        raise ValueError(f"Dim mismatch: train D={D}, query D={umap_query.shape[1]}")

    if faiss is None:
        raise RuntimeError("UMAP KNN requires faiss-cpu or a GPU-enabled FAISS installation")
    index = faiss.IndexFlatL2(D)
    if gpu_id is not None:
        if hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, gpu_id, index)
        else:
            warnings.warn("GPU FAISS is unavailable; using CPU FAISS for UMAP KNN", RuntimeWarning)
    index.add(umap_train)

    if batch_size is None:
        distances, indices = index.search(umap_query, k)
        return (distances, indices) if return_distance else indices

    all_d, all_i = [], []
    for start in range(0, umap_query.shape[0], batch_size):
        q = umap_query[start:start + batch_size]
        d, i = index.search(q, k)
        all_d.append(d)
        all_i.append(i)
    distances = np.vstack(all_d)
    indices = np.vstack(all_i)
    return (distances, indices) if return_distance else indices

def search_neighbors(mode, train_repr, query_repr, k, return_distance=True, **kwargs):
    mode = (mode or "none").lower()
    if mode == "none":
        return (None, None) if return_distance else None
    if mode == "vae":
        batch_size = kwargs.get("batch_size", 2048)
        distances, indices = torch_neighbor_search(train_repr, query_repr, k=k, batch_size=batch_size)
        return (distances, indices) if return_distance else indices
    if mode == "umap":
        batch_size = kwargs.get("batch_size", None)
        gpu_id = kwargs.get("gpu_id")
        distances, indices = faiss_flat_l2_search(train_repr, query_repr, k=k, return_distance=True, batch_size=batch_size, gpu_id=gpu_id)
        return (distances, indices) if return_distance else indices
    raise ValueError("mode must be one of: 'vae', 'umap', 'none'")


def recon_sse_torch(X, X_recons):
    err = (X - X_recons) ** 2
    return torch.sum(err, dim=(1, 2))

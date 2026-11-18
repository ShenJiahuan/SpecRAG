"""FAISS index creation, persistence, and search operations."""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

import config


def create_index(dimension: int, nlist: int = 100, m: int = 8, use_pq: bool = True, use_opq: bool = False) -> faiss.Index:
    """
    Create a new FAISS IVF index with or without Product Quantization.
    
    Args:
        dimension: Dimension of the embedding vectors.
        nlist: Number of centroids (clusters).
        m: Number of sub-quantizers for Product Quantization (only used if use_pq=True).
        use_pq: If True, create IndexIVFPQ (with Product Quantization).
                If False, create IndexIVFFlat (without Product Quantization, more accurate but larger).
        use_opq: If True, use Optimized Product Quantization (OPQ) with IVFPQ.
                 Ignored if use_pq is False.
    
    Returns:
        A new FAISS IVF index (IndexIVFPQ, IndexIVFFlat, or IndexPreTransform).
    """

    # Construct the factory string
    factory_string = ""
    
    if use_pq:
        if use_opq:
            factory_string += f"OPQ{m},"
        
        factory_string += f"IVF{nlist},PQ{m}"
    else:
        factory_string += f"IVF{nlist},Flat"
    
    print(f"Creating index using factory string: '{factory_string}'")
    
    # Create the index using the factory
    # index_factory(d, description, metric=METRIC_L2)
    # Default metric is L2, which is what we want.
    index = faiss.index_factory(dimension, factory_string)
    
    return index


def add_vectors(index: faiss.Index, vectors: np.ndarray) -> None:
    """
    Add vectors to the FAISS index.
    
    Args:
        index: FAISS index to add vectors to.
        vectors: numpy array of shape (n, dimension).
    """
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    
    # IVF indices need training before adding vectors
    if not index.is_trained:
        print(f"Training index with {len(vectors)} vectors...")
        index.train(vectors)
    
    index.add(vectors)


def save_index(index: faiss.Index, path: Path | str = config.INDEX_FILE) -> None:
    """
    Save FAISS index to disk.
    
    Args:
        index: FAISS index to save.
        path: Path to save the index to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    print(f"Index saved to {path}")


def load_index(path: Path | str = config.INDEX_FILE) -> faiss.Index:
    """
    Load FAISS index from disk.
    
    Args:
        path: Path to load the index from.
    
    Returns:
        Loaded FAISS index.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    
    return faiss.read_index(str(path))


def search(
    index: faiss.Index,
    query_vectors: np.ndarray,
    top_k: int = 5,
    nprobe: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index for similar vectors.
    
    Args:
        index: FAISS index to search.
        query_vectors: Query vectors of shape (n_queries, dimension).
        top_k: Number of nearest neighbors to return.
        nprobe: Number of clusters to visit during search (for IVF indices).
    
    Returns:
        Tuple of (distances, indices) arrays, each of shape (n_queries, top_k).
    """
    if query_vectors.dtype != np.float32:
        query_vectors = query_vectors.astype(np.float32)
    
    if query_vectors.ndim == 1:
        query_vectors = query_vectors.reshape(1, -1)
    
    # Set nprobe if it's an IVF index
    # Handle wrapped indices (e.g., IndexPreTransform for OPQ)
    scan_index = index
    while hasattr(scan_index, "index"):
        scan_index = faiss.downcast_index(scan_index.index)
    
    if hasattr(scan_index, 'nprobe'):
        scan_index.nprobe = nprobe
    
    distances, indices = index.search(query_vectors, top_k)
    return distances, indices


def save_metadata(metadata: list[dict[str, Any]], path: Path | str = config.METADATA_FILE) -> None:
    """
    Save metadata (article info for each vector) to disk.
    
    Args:
        metadata: List of metadata dicts, one per vector in the index.
        path: Path to save the metadata to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Metadata saved to {path}")


def load_metadata(path: Path | str = config.METADATA_FILE) -> list[dict[str, Any]]:
    """
    Load metadata from disk.
    
    Args:
        path: Path to load the metadata from.
    
    Returns:
        List of metadata dicts.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        if str(path).endswith(".jsonl"):
            return [json.loads(line) for line in f]
        return json.load(f)

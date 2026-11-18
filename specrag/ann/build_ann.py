"""Build FAISS ANN index from pre-computed embeddings."""

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from ann import faiss_index


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build FAISS IVF index from embeddings")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wiki-18",
        help="Dataset name (subfolder in data directory)",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=32768,
        help="Number of centroids (clusters) for IVF",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=256,
        help="Number of sub-quantizers for Product Quantization (only used if --use-pq is True)",
    )
    parser.add_argument(
        "--use-pq",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=True,
        help="Use Product Quantization (IVFPQ). Set to False for IVFFlat (no compression, more accurate). Default: True",
    )
    parser.add_argument(
        "--use-opq",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=True,
        help="Use Optimized Product Quantization (OPQ) with IVFPQ. Default: False",
    )
    parser.add_argument(
        "--index-file",
        type=str,
        default=None,
        help="Path to save the index file (default: data/<dataset>/index.faiss or index_flat.faiss)",
    )
    parser.add_argument(
        "--exclude-id-prefix",
        type=str,
        default=None,
        help="Exclude passages whose ID starts with this prefix (e.g. 'infobox-' to build a text-only index)",
    )
    return parser.parse_args()


def get_paths_and_config(args: argparse.Namespace) -> Tuple[Path, Path, str]:
    """Determine paths and index type based on arguments."""
    data_dir = Path(config.DATA_DIR) / args.dataset
    embeddings_dir = data_dir / "embeddings"

    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

    # Determine output path
    if args.index_file:
        index_path = Path(args.index_file)
    else:
        if args.use_pq:
            index_path = data_dir / ("index_opq.faiss" if args.use_opq else "index.faiss")
        else:
            index_path = data_dir / "index_flat.faiss"

    if args.use_pq:
        index_type = "OPQ+IVFPQ" if args.use_opq else "IVFPQ"
    else:
        index_type = "IVFFlat"

    return embeddings_dir, index_path, index_type


def print_build_config(
    args: argparse.Namespace, index_type: str, embeddings_dir: Path, index_path: Path
) -> None:
    """Print the index building configuration."""
    print(f"Building {index_type} index:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Embeddings directory: {embeddings_dir}")
    print(f"  Index output: {index_path}")
    print(f"  IVF nlist: {args.nlist}")
    if args.use_pq:
        print(f"  PQ m: {args.m}")
        if args.use_opq:
            print(f"  OPQ: enabled (m={args.m})")
    else:
        print("  Product Quantization: disabled (full vectors stored)")
    print()


def _load_keep_mask(npy_file: Path, exclude_id_prefix: str) -> np.ndarray:
    """Return a boolean mask of passages to keep (those NOT matching the prefix)."""
    metadata_file = npy_file.parent / "metadata.json"
    if not metadata_file.exists():
        # No metadata → keep everything
        n = len(np.load(npy_file, mmap_mode="r"))
        return np.ones(n, dtype=bool)
    with open(metadata_file, "r") as f:
        meta = json.load(f)
    passage_ids = meta.get("passage_ids", [])
    embeddings_len = len(np.load(npy_file, mmap_mode="r"))
    mask = np.ones(embeddings_len, dtype=bool)
    for i, pid in enumerate(passage_ids):
        if i >= embeddings_len:
            break
        if pid.startswith(exclude_id_prefix):
            mask[i] = False
    return mask


def collect_training_samples(
    npy_files: List[Path], dimension: int, nlist: int, exclude_id_prefix: str = None
) -> np.ndarray:
    """Collect random samples from embedding files for training the index."""
    # FAISS recommends at least 39*nlist training points for good clustering
    # We use double the recommended amount to ensure better quality.
    # For nlist=32768, that's ~2.56M training points
    min_training_samples = max(78 * nlist, 3_000_000)
    print(f"\nCollecting training samples (target: {min_training_samples:,} vectors)...")

    # First, count total vectors across all files
    total_available = 0
    file_sizes = []
    for npy_file in tqdm(npy_files, desc="Counting vectors"):
        try:
            embeddings = np.load(npy_file, mmap_mode="r")  # Memory-map to avoid loading all
            n = len(embeddings)
            if exclude_id_prefix:
                mask = _load_keep_mask(npy_file, exclude_id_prefix)
                n = int(mask.sum())
            file_sizes.append((npy_file, n))
            total_available += n
        except Exception as e:
            print(f"\n⚠ Error reading {npy_file.name}: {e}")
            continue

    print(f"Total available vectors: {total_available:,}")

    # Determine how many samples to collect
    n_training_samples = min(min_training_samples, total_available)
    print(f"Collecting {n_training_samples:,} random samples for training...")

    # Randomly sample vectors from all files
    training_vectors = []
    for npy_file, file_size in tqdm(file_sizes, desc="Sampling vectors"):
        try:
            if file_size == 0:
                continue

            # Determine how many samples from this file proportionally
            n_samples_from_file = int(n_training_samples * file_size / total_available)

            if n_samples_from_file > 0:
                embeddings = np.load(npy_file)

                # Verify shape
                if embeddings.shape[1] != dimension:
                    print(f"\n⚠ Warning: Skipping {npy_file.name} - dimension mismatch")
                    continue

                # Apply exclusion filter
                if exclude_id_prefix:
                    mask = _load_keep_mask(npy_file, exclude_id_prefix)
                    embeddings = embeddings[mask]
                    if len(embeddings) == 0:
                        continue

                # Randomly sample from this file
                if n_samples_from_file >= len(embeddings):
                    sampled = embeddings
                else:
                    indices = np.random.choice(len(embeddings), n_samples_from_file, replace=False)
                    sampled = embeddings[indices]

                training_vectors.append(sampled)

                # Break early if we have enough samples
                if sum(len(v) for v in training_vectors) >= n_training_samples:
                    break
        except Exception as e:
            print(f"\n⚠ Error sampling from {npy_file.name}: {e}")
            continue

    return np.vstack(training_vectors)


def populate_index(
    index: Any, npy_files: List[Path], dimension: int, exclude_id_prefix: str = None
) -> Tuple[int, Dict[str, Any], Dict[int, str]]:
    """Add embeddings to the FAISS index and build LangChain docstore dictionary."""
    from langchain_core.documents import Document

    print("\nAdding all embeddings to index...")
    total_vectors = 0
    total_excluded = 0
    docstore_dict = {}
    index_to_docstore_id = {}

    for _, npy_file in enumerate(tqdm(npy_files, desc="Processing files"), start=1):
        try:
            embeddings = np.load(npy_file)

            # Verify shape
            if embeddings.shape[1] != dimension:
                print(
                    f"\n⚠ Warning: Skipping {npy_file.name} - dimension mismatch "
                    f"(expected {dimension}, got {embeddings.shape[1]})"
                )
                continue

            # Load metadata for LangChain Docstore
            metadata_file = npy_file.parent / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    meta_data = json.load(f)
                passage_ids = meta_data.get("passage_ids", [])
                passage_texts = meta_data.get("passage_texts", [])
            else:
                passage_ids = []
                passage_texts = []

            # Apply exclusion filter
            if exclude_id_prefix:
                mask = np.ones(len(embeddings), dtype=bool)
                for j, pid in enumerate(passage_ids):
                    if j >= len(embeddings):
                        break
                    if pid.startswith(exclude_id_prefix):
                        mask[j] = False
                n_excluded = int((~mask).sum())
                total_excluded += n_excluded
                if n_excluded == len(embeddings):
                    continue  # skip entirely
                keep_indices = np.where(mask)[0]
                embeddings = embeddings[keep_indices]
                passage_ids = [passage_ids[i] for i in keep_indices if i < len(passage_ids)]
                passage_texts = [passage_texts[i] for i in keep_indices if i < len(passage_texts)]

            # Add to index
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            index.add(embeddings)

            num_embeddings = len(embeddings)
            for j in range(num_embeddings):
                doc_id = str(uuid.uuid4())
                index_to_docstore_id[total_vectors + j] = doc_id

                text_content = passage_texts[j] if j < len(passage_texts) else ""
                passage_id = passage_ids[j] if j < len(passage_ids) else ""

                docstore_dict[doc_id] = Document(
                    page_content=text_content,
                    metadata={"passage_id": passage_id, "source": str(npy_file)},
                )

            total_vectors += num_embeddings

        except Exception as e:
            print(f"\n⚠ Error processing {npy_file.name}: {e}")
            continue

    print(f"\n✓ Successfully added {total_vectors:,} vectors to the index")
    if exclude_id_prefix:
        print(f"  Excluded {total_excluded:,} passages with id prefix '{exclude_id_prefix}'")
    print(f"  Index stats: ntotal={index.ntotal}")

    return total_vectors, docstore_dict, index_to_docstore_id


def save_langchain_vectorstore(
    index: Any,
    docstore_dict: Dict[str, Any],
    index_to_docstore_id: Dict[int, str],
    index_path: Path,
) -> None:
    """Save FAISS index, docstore (SQLite), and id mapping for LangChain loading."""
    import pickle
    import sqlite3

    import faiss as faiss_lib

    langchain_dir = Path(f"{index_path}_langchain")
    langchain_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save FAISS index
    faiss_path = langchain_dir / "index.faiss"
    print(f"\nSaving FAISS index to {faiss_path}...")
    faiss_lib.write_index(index, str(faiss_path))

    # 2. Save index_to_docstore_id mapping (small)
    mapping_path = langchain_dir / "index_mapping.pkl"
    print(f"Saving id mapping to {mapping_path}...")
    with open(mapping_path, "wb") as fh:
        pickle.dump(index_to_docstore_id, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # 3. Save docstore to SQLite
    db_path = langchain_dir / "index.db"
    print(f"Saving docstore to {db_path}...")
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, page_content TEXT, metadata TEXT)")
    conn.execute("PRAGMA synchronous=OFF")

    batch = []
    for doc_id, doc in docstore_dict.items():
        meta = json.dumps(doc.metadata) if hasattr(doc, "metadata") else "{}"
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        batch.append((doc_id, content, meta))
        if len(batch) >= 10000:
            conn.executemany("INSERT OR REPLACE INTO docs VALUES (?, ?, ?)", batch)
            batch.clear()
    if batch:
        conn.executemany("INSERT OR REPLACE INTO docs VALUES (?, ?, ?)", batch)

    conn.commit()
    conn.close()
    print(f"  Saved {len(docstore_dict)} documents ({db_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Saving LangChain vectorstore to {langchain_dir}...")


def main() -> None:
    """Main execution flow for building the index."""
    args = parse_args()
    embeddings_dir, index_path, index_type = get_paths_and_config(args)

    print_build_config(args, index_type, embeddings_dir, index_path)

    # Find all .npy files
    npy_files = sorted(embeddings_dir.rglob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {embeddings_dir}")
    print(f"Found {len(npy_files)} embedding files")

    # Load first file to get dimension
    first_embeddings = np.load(npy_files[0])
    dimension = first_embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")

    # Create index
    print(f"\nCreating {index_type} index...")
    index = faiss_index.create_index(
        dimension=dimension,
        nlist=args.nlist,
        m=args.m,
        use_pq=args.use_pq,
        use_opq=args.use_opq if args.use_pq else False,
    )

    if args.exclude_id_prefix:
        print(f"  Excluding passages with id prefix: '{args.exclude_id_prefix}'")

    # Collect training data and train index
    training_data = collect_training_samples(npy_files, dimension, args.nlist, args.exclude_id_prefix)
    print(f"\nCollected {len(training_data):,} training vectors")
    print("Training index...")
    index.train(training_data.astype(np.float32))
    print("Training complete!")

    # Add embeddings to index
    total_vectors, docstore_dict, index_to_docstore_id = populate_index(
        index, npy_files, dimension, args.exclude_id_prefix
    )

    # Save raw index
    print(f"\nSaving raw faiss index to {index_path}...")
    faiss_index.save_index(index, index_path)

    # Save LangChain vectorstore
    save_langchain_vectorstore(index, docstore_dict, index_to_docstore_id, index_path)

    print("\n✓ Index building complete!")


if __name__ == "__main__":
    main()

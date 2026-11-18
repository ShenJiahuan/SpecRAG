#!/usr/bin/env python3
"""Generate a small synthetic corpus, embed it, and build an IVF-PQ demo index.

This script lets you verify the end-to-end SpecRAG pipeline in ~1 minute
without downloading the full 21M-chunk Wikipedia dump first.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config
from ann import build_ann, faiss_index
from embeddings.embeddings import embed_texts, get_openai_client


TOPICS = [
    ("Speculative Prefetching", "SpecRAG overlaps disk-resident KV cache loading with cluster-based IVF-PQ search once candidate top-K results stabilize."),
    ("Position-Independent Caching", "CacheBlend and EPIC precompute document chunk KV caches offline and selectively recompute a small subset of tokens at inference time."),
    ("Inverted File Product Quantization", "IVF-PQ partitions corpus embeddings into coarse Voronoi cells and scans the closest nprobe lists in order of centroid distance."),
    ("PagedAttention Memory", "vLLM manages KV cache blocks using PagedAttention so physical GPU memory blocks can be allocated dynamically per request."),
    ("Rotary Position Embedding", "RoPE encodes token positions via rotation matrices, allowing cached chunk keys to be rotated to their runtime prompt offsets."),
]


def main():
    parser = argparse.ArgumentParser(description="Build a small synthetic IVF-PQ index for quick testing")
    parser.add_argument("--num-passages", type=int, default=512,
                        help="Number of passages to generate (must be >= 256 for 8-bit PQ)")
    parser.add_argument("--nlist", type=int, default=64, help="Number of IVF clusters")
    parser.add_argument("--m", type=int, default=64, help="Number of PQ subquantizers")
    parser.add_argument("--embedding-url", type=str, default=None,
                        help="Embedding API base URL (default: config.OPENAI_EMBEDDING_BASE_URL)")
    args = parser.parse_args()

    if args.num_passages < max(256, args.nlist):
        raise ValueError("--num-passages must be at least max(256, --nlist) for 8-bit IVF-PQ training.")

    if args.embedding_url:
        config.OPENAI_EMBEDDING_BASE_URL = args.embedding_url

    demo_dir = Path(config.DATA_DIR) / "demo"
    emb_chunk_dir = demo_dir / "embeddings" / "chunk_00000"
    emb_chunk_dir.mkdir(parents=True, exist_ok=True)

    passage_ids = []
    passage_texts = []
    for i in range(args.num_passages):
        title, base = TOPICS[i % len(TOPICS)]
        body = (
            f"{title} (Passage {i}). {base} "
            f"In distributed retrieval-augmented serving, chunk {i} provides detailed technical context "
            f"about storage bandwidth, latency hiding, transformer attention layers, and ANN recall tradeoffs. " * 2
        )
        passage_ids.append(f"demo-{i}")
        passage_texts.append(body.strip())

    queries = [
        {"question_text": "How does SpecRAG hide disk KV cache loading latency during IVF-PQ search?"},
        {"question_text": "How do CacheBlend and EPIC reuse precomputed KV caches across positions?"},
        {"question_text": "Why do IVF-PQ top-K candidates stabilize early during cluster scanning?"},
        {"question_text": "How does RoPE allow cached chunk keys to be adjusted to new prompt offsets?"},
        {"question_text": "How does PagedAttention manage GPU KV cache memory blocks in vLLM?"},
    ]
    queries_path = demo_dir / "queries.jsonl"
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    print(f"[demo] Embedding {len(passage_texts)} passages via {config.OPENAI_EMBEDDING_BASE_URL} ...")
    client = get_openai_client()
    vectors, _ = embed_texts(passage_texts, client=client, show_progress=True)
    np.save(emb_chunk_dir / "embeddings.npy", vectors)
    with open(emb_chunk_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "passage_ids": passage_ids,
                "passage_texts": passage_texts,
                "num_embeddings": len(vectors),
                "embedding_dim": int(vectors.shape[1]),
            },
            f,
            indent=2,
        )

    print(f"[demo] Training OPQ+IVF-PQ index (nlist={args.nlist}, m={args.m}) ...")
    index = faiss_index.create_index(
        dimension=int(vectors.shape[1]),
        nlist=args.nlist,
        m=args.m,
        use_pq=True,
        use_opq=True,
    )
    index.train(vectors.astype(np.float32))
    _, docstore_dict, index_to_docstore_id = build_ann.populate_index(
        index, [emb_chunk_dir / "embeddings.npy"], int(vectors.shape[1])
    )
    index_path = demo_dir / "index_opq.faiss"
    faiss_index.save_index(index, index_path)
    build_ann.save_langchain_vectorstore(index, docstore_dict, index_to_docstore_id, index_path)
    print(f"[demo] Ready! Queries: {queries_path}")
    print(f"[demo] Index  : {demo_dir / 'index_opq.faiss_langchain'}")


if __name__ == "__main__":
    main()

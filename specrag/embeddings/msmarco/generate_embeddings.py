#!/usr/bin/env python3
"""
Generate embeddings for the MS MARCO passage dataset.

The MS MARCO corpus format is JSONL with:
    {"id": "0", "title": "", "text": "passage text..."}

We use the text field as the embedding input (title is empty for MS MARCO).

Usage:
    python embeddings/msmarco/generate_embeddings.py [OPTIONS]

Options:
    --chunk-size: Number of passages per checkpoint chunk (default: 100000)
    --start-from: Resume from a specific chunk number (default: 0)
    --max-chunks: Maximum number of chunks to process (default: all)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from embeddings.embeddings import embed_texts, get_openai_client


class EmbeddingCheckpointer:
    """Manages checkpointing for embedding generation."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.metadata_file = self.output_dir / "metadata.json"

    def load_checkpoint(self) -> dict:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r") as f:
                return json.load(f)
        return {"last_chunk": -1, "total_processed": 0}

    def save_checkpoint(self, chunk_num: int, total_processed: int):
        with open(self.checkpoint_file, "w") as f:
            json.dump({"last_chunk": chunk_num, "total_processed": total_processed}, f, indent=2)

    def save_chunk_embeddings(
        self,
        chunk_num: int,
        embeddings: np.ndarray,
        passage_ids: list[str],
        passage_texts: list[str],
    ):
        chunk_dir = self.output_dir / f"chunk_{chunk_num:05d}"
        chunk_dir.mkdir(exist_ok=True)
        np.save(chunk_dir / "embeddings.npy", embeddings)
        metadata = {
            "passage_ids": passage_ids,
            "passage_texts": passage_texts,
            "num_embeddings": len(embeddings),
            "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
        }
        with open(chunk_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def get_all_chunks(self) -> list[int]:
        chunks = []
        for chunk_dir in sorted(self.output_dir.glob("chunk_*")):
            if chunk_dir.is_dir():
                chunk_num = int(chunk_dir.name.split("_")[1])
                chunks.append(chunk_num)
        return sorted(chunks)

    def save_metadata(self, metadata: dict):
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)


def read_passages(
    data_file: Path, chunk_size: int, start_chunk: int = 0
) -> Iterator[tuple[int, list[dict]]]:
    """Read passages from the MS MARCO JSONL file in chunks."""
    chunk = []
    chunk_num = 0

    print(f"Reading from {data_file.name}...")
    with open(data_file, "r") as f:
        for line in f:
            stripped = line.strip().strip("\x00")
            if not stripped:
                continue
            try:
                passage = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            chunk.append(passage)

            if len(chunk) >= chunk_size:
                if chunk_num >= start_chunk:
                    yield chunk_num, chunk
                chunk = []
                chunk_num += 1

    if chunk and chunk_num >= start_chunk:
        yield chunk_num, chunk


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for MS MARCO dataset"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="corpus.jsonl",
        help="Data file name in data/msmarco/ (default: corpus.jsonl)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Number of passages to process per chunk",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Chunk number to start from (for resuming)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.EMBEDDING_BATCH_SIZE,
        help="Batch size for embedding API calls",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=config.EMBEDDING_NUM_WORKERS,
        help="Number of concurrent workers for API calls",
    )
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "msmarco"
    data_file = data_dir / args.data_file

    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)

    output_dir = data_dir / "embeddings"

    # Initialize checkpointer
    checkpointer = EmbeddingCheckpointer(str(output_dir))
    checkpoint = checkpointer.load_checkpoint()
    last_chunk = checkpoint["last_chunk"]
    total_processed = checkpoint["total_processed"]
    start_chunk = max(args.start_from, last_chunk + 1)

    print(f"Data file: {data_file}")
    print(f"Output directory: {output_dir}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Starting from chunk: {start_chunk}")
    print(f"Total processed so far: {total_processed}")
    print()

    client = get_openai_client()
    chunks_processed = 0

    for chunk_num, passages in read_passages(data_file, args.chunk_size, start_chunk):
        if args.max_chunks and chunks_processed >= args.max_chunks:
            print(f"\nReached maximum chunks limit ({args.max_chunks}). Stopping.")
            break

        print(f"Processing chunk {chunk_num} ({len(passages)} passages)...")

        passage_ids = [p["id"] for p in passages]
        passage_texts = [p["text"] for p in passages]

        try:
            embeddings, skipped_texts = embed_texts(
                passage_texts,
                client=client,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                show_progress=True,
            )

            if skipped_texts:
                skipped_log_file = output_dir / "skipped_texts.jsonl"
                with open(skipped_log_file, "a") as f:
                    for skip_info in skipped_texts:
                        skip_info["chunk_num"] = chunk_num
                        skip_info["passage_id"] = passage_ids[
                            skip_info["index"] % len(passage_ids)
                        ]
                        f.write(json.dumps(skip_info) + "\n")
                print(
                    f"  Warning: {len(skipped_texts)} texts were too long and assigned random embeddings"
                )

            checkpointer.save_chunk_embeddings(
                chunk_num, embeddings, passage_ids, passage_texts
            )

            total_processed += len(passages)
            checkpointer.save_checkpoint(chunk_num, total_processed)

            print(f"Chunk {chunk_num} saved ({embeddings.shape[0]} embeddings)")
            print(f"  Total processed: {total_processed}\n")

            chunks_processed += 1

        except Exception as e:
            print(f"Error processing chunk {chunk_num}: {e}")
            print(
                f"  You can resume from chunk {chunk_num} by running with --start-from {chunk_num}"
            )
            sys.exit(1)

    # Save overall metadata
    all_chunks = checkpointer.get_all_chunks()
    metadata = {
        "data_file": args.data_file,
        "chunk_size": args.chunk_size,
        "total_chunks": len(all_chunks),
        "total_passages": total_processed,
        "chunks": all_chunks,
    }
    checkpointer.save_metadata(metadata)

    print("\n" + "=" * 60)
    print("Embedding generation complete!")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Total passages: {total_processed}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

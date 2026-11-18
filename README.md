# SpecRAG

Source code for **SpecRAG: Speculative KV Cache Prefetching for Position-Independent Caching in RAG Serving over Cluster-Based ANN Indexes**.

SpecRAG hides disk-resident KV cache loading latency behind ongoing IVF-based ANN search in position-independent caching (PIC) RAG systems:
1. **Online Convergence Detection (`faiss/`)**: Monitors intermediate top-$K$ set stability inside the FAISS IVF search loop and invokes an asynchronous callback once the candidate set stabilizes.
2. **Speculative KV Data Path (`lmcache/`)**: Streams candidate chunk KV cache layers from disk (`O_DIRECT`) into a GPU speculative buffer during the remaining ANN scan, reconciles hits/misses against the final top-$K$, and overlaps remaining reads with CacheBlend or EPIC selective recomputation.
3. **RAG Orchestrator & Minimal Runner (`specrag/`)**: Coordinates embedding, FAISS IVF-PQ retrieval, speculative prefetch dispatch, and vLLM streaming inference.

## Repository Structure

```text
SpecRAG/
├── faiss/        # Patched FAISS (v1.13.2) with IVF intermediate callback & convergence detector
├── lmcache/      # Patched LMCache (v0.3.9) with speculative KV prefetch, O_DIRECT reader, CacheBlend & EPIC
└── specrag/      # LangChain + FAISS RAG client, index builder, and minimal benchmark scripts
```

## Prerequisites

- Linux (x86_64) with NVIDIA GPU(s) and CUDA 12.x
- Python 3.10–3.12, `cmake >= 3.24`, `swig`, `g++`, OpenBLAS/LAPACK (`sudo apt-get install -y cmake swig libblas-dev liblapack-dev`)
- An SSD-backed directory supporting `O_DIRECT` (avoid `tmpfs` such as `/tmp` for `--kv-cache-dir`)

## Installation

You can install everything into a single virtual environment (or separate client/server environments if desired).

```bash
# Create and activate a virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# 1. Build and install patched FAISS (CPU IVF-PQ + Python bindings)
cmake -S faiss -B faiss/build \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_OPT_LEVEL=avx2 \
    -DPython_EXECUTABLE="$(which python3)"
cmake --build faiss/build -j"$(nproc)"
uv pip install ./faiss/build/faiss/python

# 2. Install vLLM and build patched LMCache from source
uv pip install vllm xformers
uv pip install -r lmcache/requirements/build.txt
uv pip install -e ./lmcache --no-build-isolation

# 3. Install RAG orchestrator dependencies
uv pip install -e ./specrag
```

## Quick Start: Run a Minimal Case

### 1. Start the Embedding Server

Start `Qwen/Qwen3-Embedding-0.6B` on port `8001` (shares GPU 0 using 15% memory, or set `CUDA_VISIBLE_DEVICES` to another GPU):

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Embedding-0.6B \
    --runner pooling \
    --port 8001 \
    --gpu-memory-utilization 0.15
```

### 2. Run a Self-Contained Demo Case

Build a small synthetic OPQ+IVF-PQ index (`specrag/data/demo/`) and run an end-to-end comparison across `baseline` (no caching), `cacheblend`, and `cacheblend+speculative` (SpecRAG):

```bash
cd specrag

# Generate synthetic passages, compute embeddings, and build IVF-PQ index
python3 scripts/prepare_demo_data.py --embedding-url http://localhost:8001/v1

# Run baseline vs. CacheBlend vs. CacheBlend+SpecRAG
./scripts/run_example.sh \
    --dataset demo \
    --query-file data/demo/queries.jsonl \
    --num-queries 5 \
    --nprobe 32 \
    --gpu 0 \
    --gpu-mem 0.70 \
    --kv-cache-dir ./kv_cache
```

To also test EPIC (`epic` and `epic+speculative`), pass `--modes baseline,cacheblend,cacheblend+speculative,epic,epic+speculative`.

### 3. Run on Wikipedia / Natural Questions (Paper Setup)

Place the DPR Wikipedia corpus at `specrag/data/wiki-18/wiki-18.jsonl` (`{"id": "...", "contents": "..."}`) and NQ queries at `specrag/data/nq/nq.jsonl` (`{"question_text": "..."}`):

```bash
cd specrag

# 1) Encode corpus chunks (omit --max-chunks for full corpus)
OPENAI_EMBEDDING_BASE_URL=http://localhost:8001/v1 \
    python3 embeddings/wiki-18/generate_embeddings.py --max-chunks 5

# 2) Build OPQ+IVF-PQ index + SQLite docstore
python3 ann/build_ann.py --dataset wiki-18 --nlist 4096 --m 256

# 3) Run benchmark case (nprobe=1024, K=10)
./scripts/run_example.sh \
    --model Qwen/Qwen3-4B \
    --modes baseline,cacheblend,cacheblend+speculative \
    --dataset wiki-18 \
    --query-file data/nq/nq.jsonl \
    --num-queries 20 \
    --nprobe 1024 \
    --top-k 10 \
    --gpu 0 \
    --kv-cache-dir ./kv_cache
```

You can also start the server manually with `./scripts/serve_vllm.sh --help` and invoke `python3 inference/rag_inference.py --help` directly.

## License

- `faiss/` is based on [FAISS](https://github.com/facebookresearch/faiss) (MIT License).
- `lmcache/` is based on [LMCache](https://github.com/LMCache/LMCache) (Apache-2.0 License).

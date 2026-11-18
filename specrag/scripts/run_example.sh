#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen3-4B"
MODES_CSV="baseline,cacheblend,cacheblend+speculative"
DATASET="wiki-18"
INDEX_PATH=""
QUERY_FILE="$REPO_ROOT/data/nq/nq.jsonl"
SINGLE_QUERY=""
NUM_QUERIES=20
NPROBE=1024
TOP_K=10
GPU_ID="0"
GPU_MEM="0.75"
LLM_PORT="8000"
PREFETCH_PORT="7800"
EMBEDDING_URL="${OPENAI_EMBEDDING_BASE_URL:-http://localhost:8001/v1}"
KV_CACHE_DIR="$REPO_ROOT/kv_cache"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
Usage: run_example.sh [OPTIONS]

Run a minimal end-to-end SpecRAG benchmark case across modes and compare TTFT.
Assumes an OpenAI-compatible embedding server (Qwen/Qwen3-Embedding-0.6B) is
already running at --embedding-url.

Options:
  --model MODEL           Generation model (default: Qwen/Qwen3-4B)
  --modes M1,M2,...       Comma-separated modes (default: baseline,cacheblend,cacheblend+speculative)
  --dataset NAME          Dataset folder under data/ (default: wiki-18)
  --index-path DIR        Explicit path to LangChain FAISS index folder
  --query-file PATH       Query dataset file (default: data/nq/nq.jsonl)
  --query "TEXT"          Run a single query instead of --query-file
  --num-queries N         Number of queries from --query-file (default: 20)
  --nprobe N              IVF nprobe search budget (default: 1024)
  --top-k K               Retrieved chunks per query (default: 10)
  --gpu ID                GPU ID for the LLM server (default: 0)
  --gpu-mem FRAC          vLLM GPU memory utilization (default: 0.75)
  --llm-port PORT         Port for local vLLM server (default: 8000)
  --prefetch-port PORT    Port for LMCache prefetch API (default: 7800)
  --embedding-url URL     Embedding server base URL (default: http://localhost:8001/v1)
  --kv-cache-dir DIR      Directory on SSD for disk KV caches (default: ./kv_cache)
  -h, --help              Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --model) MODEL="$2"; shift 2 ;;
        --modes) MODES_CSV="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --index-path) INDEX_PATH="$2"; shift 2 ;;
        --query-file) QUERY_FILE="$2"; shift 2 ;;
        --query) SINGLE_QUERY="$2"; shift 2 ;;
        --num-queries) NUM_QUERIES="$2"; shift 2 ;;
        --nprobe) NPROBE="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM="$2"; shift 2 ;;
        --llm-port) LLM_PORT="$2"; shift 2 ;;
        --prefetch-port) PREFETCH_PORT="$2"; shift 2 ;;
        --embedding-url) EMBEDDING_URL="$2"; shift 2 ;;
        --kv-cache-dir) KV_CACHE_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if ! curl -sf "${EMBEDDING_URL%/}/models" >/dev/null 2>&1; then
    echo "Error: Embedding endpoint is not reachable at ${EMBEDDING_URL}" >&2
    echo "Start it first, e.g.:" >&2
    echo "  CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Embedding-0.6B --runner pooling --port 8001 --gpu-memory-utilization 0.15" >&2
    exit 1
fi

RUN_DIR="$REPO_ROOT/logs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

SERVER_PID=""
stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[server] Stopping vLLM (pid=$SERVER_PID)..."
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 -- "-$SERVER_PID" 2>/dev/null || kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap stop_server EXIT

wait_for_server() {
    local url="http://localhost:${LLM_PORT}/v1/models"
    echo "[server] Waiting for $url ..."
    for _ in $(seq 1 120); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[server] vLLM exited unexpectedly. Check log in $RUN_DIR" >&2
            return 1
        fi
        if curl -sf "$url" >/dev/null 2>&1; then
            echo "[server] Ready."
            return 0
        fi
        sleep 5
    done
    echo "[server] Timed out waiting for vLLM." >&2
    return 1
}

IFS=',' read -ra MODES <<< "$MODES_CSV"

for mode in "${MODES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Running mode: $mode  (model=$MODEL, nprobe=$NPROBE, K=$TOP_K)"
    echo "============================================================"

    SERVER_LOG="$RUN_DIR/vllm_${mode}.log"
    setsid "$SCRIPT_DIR/serve_vllm.sh" \
        --mode "$mode" \
        --model "$MODEL" \
        --gpu "$GPU_ID" \
        --port "$LLM_PORT" \
        --prefetch-port "$PREFETCH_PORT" \
        --gpu-mem "$GPU_MEM" \
        --kv-cache-dir "$KV_CACHE_DIR" \
        >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    wait_for_server

    CLIENT_ARGS=(
        --dataset "$DATASET"
        --llm-model "$MODEL"
        --mode "$mode"
        --nprobe "$NPROBE"
        --top-k "$TOP_K"
        --embedding-base-url "$EMBEDDING_URL"
        --chat-base-url "http://localhost:${LLM_PORT}/v1"
        --prefetch-url "http://localhost:${PREFETCH_PORT}/v1/speculative_prefetch"
        --warmup-first
        --log-dir "$RUN_DIR"
    )
    if [[ -n "$INDEX_PATH" ]]; then
        CLIENT_ARGS+=(--index-path "$INDEX_PATH")
    fi
    if [[ "$mode" == *"+speculative" ]]; then
        CLIENT_ARGS+=(--speculative-prefetch --adaptive-convergence)
    fi
    if [[ -n "$SINGLE_QUERY" ]]; then
        CLIENT_ARGS+=(--query "$SINGLE_QUERY" --repeat 5)
    else
        CLIENT_ARGS+=(--query-file "$QUERY_FILE" --num-queries "$NUM_QUERIES")
    fi

    "$PYTHON_BIN" "$REPO_ROOT/inference/rag_inference.py" "${CLIENT_ARGS[@]}"

    stop_server
    sleep 3
done

echo ""
echo "============================================================"
echo "  Comparison Summary ($RUN_DIR)"
echo "============================================================"
"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import glob
import json
import os
import sys

run_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(run_dir, "*.json")))
print(f"{'Mode':<26} {'E2E TTFT (ms)':>14} {'LLM TTFT (ms)':>14} {'FAISS (ms)':>11} {'Hit Rate':>10}")
print("-" * 79)
for path in files:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    mode = data["config"]["mode"]
    s = data["summary"]
    e2e = f"{s['avg_e2e_ttft_ms']:.1f}" if s.get("avg_e2e_ttft_ms") is not None else "N/A"
    llm = f"{s['avg_llm_ttft_ms']:.1f}" if s.get("avg_llm_ttft_ms") is not None else "N/A"
    faiss_ms = f"{s['avg_faiss_ms']:.1f}" if s.get("avg_faiss_ms") is not None else "N/A"
    hit = f"{s['avg_prefetch_hit_rate']:.1f}%" if s.get("avg_prefetch_hit_rate") is not None else "-"
    print(f"{mode:<26} {e2e:>14} {llm:>14} {faiss_ms:>11} {hit:>10}")
PY

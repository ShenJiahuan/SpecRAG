#!/usr/bin/env bash
set -euo pipefail

MODE="cacheblend+speculative"
MODEL="Qwen/Qwen3-4B"
GPU_ID="0"
PORT="8000"
PREFETCH_PORT="7800"
GPU_MEM="0.8"
MAX_MODEL_LEN="32768"
KV_CACHE_DIR="./kv_cache"
CHUNK_SEP=""
USE_ODIRECT="${USE_ODIRECT:-true}"

usage() {
    cat <<'EOF'
Usage: serve_vllm.sh [OPTIONS]

Start a local vLLM (+ LMCache) server for SpecRAG evaluation.

Options:
  --mode MODE             baseline | cacheblend | cacheblend+speculative | epic | epic+speculative
                          (default: cacheblend+speculative)
  --model MODEL           HuggingFace model ID (default: Qwen/Qwen3-4B)
  --gpu ID                CUDA device ID (default: 0)
  --port PORT             vLLM OpenAI HTTP port (default: 8000)
  --prefetch-port PORT    LMCache speculative prefetch HTTP port (default: 7800)
  --gpu-mem FRAC          vLLM gpu_memory_utilization (default: 0.8)
  --max-model-len LEN     Maximum context length (default: 32768)
  --kv-cache-dir DIR      Disk directory for KV cache files (default: ./kv_cache;
                          use a filesystem supporting O_DIRECT, not tmpfs)
  --chunk-separator STR   Override atomic single-token chunk separator
  -h, --help              Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --mode) MODE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --prefetch-port) PREFETCH_PORT="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --kv-cache-dir) KV_CACHE_DIR="$2"; shift 2 ;;
        --chunk-separator) CHUNK_SEP="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

resolve_chunk_sep() {
    case "$1" in
        Qwen/Qwen3-*) echo '<|file_sep|>' ;;
        mistralai/Mistral-Small-3.1-24B-Instruct-2503) echo '<SPECIAL_999>' ;;
        zai-org/GLM-4-32B-0414) echo '<|end_of_image|>' ;;
        *)
            echo "Error: no default atomic chunk separator for model '$1'. Pass --chunk-separator." >&2
            return 1
            ;;
    esac
}

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONHASHSEED=0

COMMON_VLLM_ARGS=(
    "$MODEL"
    --port "$PORT"
    --gpu-memory-utilization "$GPU_MEM"
    --max-model-len "$MAX_MODEL_LEN"
    --no-enable-prefix-caching
    --enforce-eager
)

if [[ "$MODE" == "baseline" ]]; then
    echo "[serve_vllm] Starting baseline vLLM (no KV cache reuse): model=$MODEL gpu=$GPU_ID port=$PORT"
    exec vllm serve "${COMMON_VLLM_ARGS[@]}"
fi

if [[ -z "$CHUNK_SEP" ]]; then
    CHUNK_SEP="$(resolve_chunk_sep "$MODEL")"
fi

mkdir -p "$KV_CACHE_DIR"
KV_CACHE_ABS="$(cd "$KV_CACHE_DIR" && pwd)"
rm -rf "${KV_CACHE_ABS:?}"/*

export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_ENABLE_BLENDING=True
export LMCACHE_USE_LAYERWISE=True
export LMCACHE_BLEND_SPECIAL_STR="$CHUNK_SEP"
export LMCACHE_BLEND_CHECK_LAYERS="1"

export LMCACHE_LOCAL_CPU=False
export LMCACHE_MAX_LOCAL_CPU_SIZE="${CPU_BUFFER_GB:-10}"
export LMCACHE_LOCAL_DISK="file://${KV_CACHE_ABS}/"
export LMCACHE_MAX_LOCAL_DISK_SIZE="${MAX_DISK_GB:-100}"
export LMCACHE_DISK_IO_WORKERS="${DISK_IO_WORKERS:-32}"
export LMCACHE_EXTRA_CONFIG="{\"use_odirect\": ${USE_ODIRECT}, \"force_store_wait\": true}"

case "$MODE" in
    cacheblend|cacheblend+speculative)
        export LMCACHE_BLENDING_ALGORITHM="cacheblend"
        export LMCACHE_BLEND_RECOMPUTE_RATIOS="0.15"
        ;;
    epic|epic+speculative)
        export LMCACHE_BLENDING_ALGORITHM="epic"
        export LMCACHE_EPIC_RECOMP_RATIO="0.16"
        ;;
    *)
        echo "Unsupported mode: $MODE" >&2
        exit 1
        ;;
esac

if [[ "$MODE" == *"+speculative" ]]; then
    export LMCACHE_SPECULATIVE_PREFETCH_ENABLED=True
    export LMCACHE_SPECULATIVE_PREFETCH_LAYERWISE_BASELINE=True
    export LMCACHE_SPECULATIVE_PREFETCH_NAIVE_PHASE2=True
    export LMCACHE_SPECULATIVE_PREFETCH_GPU_MEMORY_BUDGET_GB="${SPEC_GPU_BUDGET_GB:-10.0}"
    export LMCACHE_INTERNAL_API_SERVER_ENABLED=True
    # Worker 0 binds to internal_api_server_port_start + 1
    export LMCACHE_INTERNAL_API_SERVER_PORT_START=$((PREFETCH_PORT - 1))
else
    export LMCACHE_SPECULATIVE_PREFETCH_ENABLED=False
fi

echo "[serve_vllm] Starting mode=$MODE model=$MODEL gpu=$GPU_ID port=$PORT kv_dir=$KV_CACHE_ABS"
exec vllm serve "${COMMON_VLLM_ARGS[@]}" \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

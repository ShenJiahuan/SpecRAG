# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import json

# Third Party
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

router = APIRouter()

# Cache so we only probe AutoProcessor once per tokenizer identity.
_chat_template_resolved: set = set()


def _ensure_chat_template(tokenizer) -> None:
    """Try to populate ``tokenizer.chat_template`` from AutoProcessor.

    vLLM's ``resolve_hf_chat_template`` falls back to the processor's
    chat_template when the tokenizer itself doesn't have one (common for
    Mistral models).  We replicate the same logic here so that
    ``apply_chat_template`` works in the prefetch path.
    """
    tok_id = id(tokenizer)
    if tok_id in _chat_template_resolved:
        return
    _chat_template_resolved.add(tok_id)

    model_name = getattr(tokenizer, "name_or_path", None)
    if not model_name:
        return
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name)
        ct = getattr(processor, "chat_template", None)
        if ct is not None:
            tokenizer.chat_template = ct
            logger.info(
                "Set chat_template from AutoProcessor for %s", model_name
            )
    except Exception as e:
        logger.warning(
            "Failed to load AutoProcessor for %s: %s", model_name, e
        )


def _compute_chat_template_suffix(tokenizer) -> List[int]:
    """Compute the trailing end-of-turn token suffix that
    ``apply_chat_template`` appends after the user content.

    We apply the template to an empty user message and compare with a
    non-empty one.  The common tail is the suffix that needs to be
    stripped when we only want ``<prefix> + <content>`` (no end-of-turn).
    """
    empty = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=True,
        add_generation_prompt=False,
    )
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test probe string"}],
        tokenize=True,
        add_generation_prompt=False,
    )
    suffix_len = 0
    max_check = min(len(empty), len(probe))
    for i in range(1, max_check + 1):
        if empty[-i] == probe[-i]:
            suffix_len = i
        else:
            break
    return list(empty[-suffix_len:]) if suffix_len > 0 else []


class SpeculativePrefetchRequest(BaseModel):
    """Request body for the speculative prefetch endpoint."""

    request_id: str
    # Pre-joined prompt prefix text (system_prompt + context with
    # separators), exactly as it will appear in the final prompt.
    # The server wraps it in the chat template, tokenizes once, then
    # splits by the configured separator — guaranteeing identical
    # tokenization to the retrieve path.
    prompt_prefix: Optional[str] = None
    # Legacy fields kept for backward compatibility.
    chunks: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    # Number of layers to prefetch. None falls back to config default.
    # 0 means all layers.
    num_prefetch_layers: Optional[int] = None


@router.post("/v1/speculative_prefetch")
async def speculative_prefetch(
    request: Request,
    body: SpeculativePrefetchRequest,
):
    """Speculatively prefetch KV caches for candidate chunks.

    This endpoint is called by the LangChain RAG pipeline during ANN search,
    before the final prompt is ready. It receives raw chunk text, tokenizes
    it server-side, and triggers asynchronous KV cache prefetch from disk
    to GPU memory.

    The prefetched data is correlated with a future chat completion request
    via the ``request_id`` field.

    Args:
        request: The FastAPI request object containing application state.
        body: The request body with chunk texts and a correlation request_id.

    Returns:
        PlainTextResponse with status "accepted" on success, or an error.
    """
    lmcache_adapter = request.app.state.lmcache_adapter
    lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)

    if not lmcache_engine:
        error_info = {
            "error": "speculative_prefetch API is unavailable",
            "message": "LMCache engine not configured.",
        }
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=503,
        )

    if not lmcache_engine.config.speculative_prefetch_enabled:
        error_info = {
            "error": "speculative_prefetch is disabled",
            "message": "Set speculative_prefetch_enabled=True to use this endpoint.",
        }
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=400,
        )

    try:
        # Tokenize chunk texts using the tokenizer from token_database
        tokenizer = getattr(
            getattr(lmcache_engine, "token_database", None),
            "tokenizer", None,
        )
        if tokenizer is None:
            # Fall back to engine or adapter level
            tokenizer = getattr(lmcache_engine, "tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(lmcache_adapter, "tokenizer", None)

        if tokenizer is None:
            error_info = {
                "error": "tokenizer not available",
                "message": (
                    "No tokenizer found on engine or adapter. "
                    "Cannot tokenize chunk texts."
                ),
            }
            return PlainTextResponse(
                content=json.dumps(error_info, indent=2),
                media_type="application/json",
                status_code=500,
            )

        # Some models (e.g. Mistral) don't ship chat_template on the
        # tokenizer, but the AutoProcessor has one.  Mirror vLLM's
        # fallback logic so apply_chat_template works for prefetch.
        if getattr(tokenizer, "chat_template", None) is None:
            _ensure_chat_template(tokenizer)

        def _tokenize_and_dispatch() -> int:
            """Tokenize the request body and dispatch to
            ``lmcache_engine.speculative_prefetch``. Returns the number
            of chunks tokenized. Used inline for the chunkwise path and
            on a background thread for the layerwise baseline path.
            """
            local_chunk_tokens_list: List[List[int]] = []

            if body.prompt_prefix is not None:
                # New path: client sends the pre-joined prompt prefix
                # (system_prompt + context with separators). We wrap
                # it in the chat template and tokenize once, then pass
                # the full token sequence as a single entry.
                # speculative_prefetch() will call process_tokens →
                # _text_based_split, producing identical chunks to the
                # retrieve path.
                full_template_tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": body.prompt_prefix}],
                    tokenize=True,
                    add_generation_prompt=False,
                )
                suffix = _compute_chat_template_suffix(tokenizer)
                if (
                    suffix
                    and len(full_template_tokens) > len(suffix)
                    and full_template_tokens[-len(suffix):] == suffix
                ):
                    prefix_tokens = full_template_tokens[:-len(suffix)]
                else:
                    prefix_tokens = full_template_tokens
                local_chunk_tokens_list.append(prefix_tokens)

                logger.info(
                    "Speculative prefetch %s: prompt_prefix tokenized "
                    "(%d tokens)",
                    body.request_id,
                    len(prefix_tokens),
                )
            else:
                # Legacy path: individual chunks + optional system_prompt
                if body.system_prompt is not None:
                    full_template_tokens = tokenizer.apply_chat_template(
                        [{"role": "user", "content": body.system_prompt}],
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                    suffix = _compute_chat_template_suffix(tokenizer)
                    if (
                        suffix
                        and len(full_template_tokens) > len(suffix)
                        and full_template_tokens[-len(suffix):] == suffix
                    ):
                        sys_tokens = full_template_tokens[:-len(suffix)]
                    else:
                        sys_tokens = full_template_tokens
                    local_chunk_tokens_list.append(sys_tokens)

                if body.chunks is not None:
                    for text in body.chunks:
                        local_chunk_tokens_list.append(
                            tokenizer.encode(text)
                        )

            lmcache_engine.speculative_prefetch(
                request_id=body.request_id,
                chunk_tokens_list=local_chunk_tokens_list,
                num_prefetch_layers=body.num_prefetch_layers,
            )
            return len(local_chunk_tokens_list)

        # Layerwise baseline: offload tokenization + setup to a
        # background thread so the HTTP ack returns after just a
        # network RTT. Moves ~10-15 ms of resolve_keys + batched_allocate
        # + precompute_h2d off the API handler's response path —
        # reduces client-observed prefetch_time (and, indirectly, the
        # GIL pressure that bleeds into the client-side FAISS probe
        # loop). Errors in the background task are logged server-side
        # but cannot be surfaced to the client; retrieve_layer's
        # fallback-to-disk path handles a missing spec_result.
        if lmcache_engine.config.speculative_prefetch_layerwise_baseline:
            def _async_setup() -> None:
                try:
                    _tokenize_and_dispatch()
                except Exception as exc:
                    logger.error(
                        "speculative_prefetch %s (async layerwise) "
                        "failed: %s",
                        body.request_id, str(exc),
                    )
            lmcache_engine.storage_manager.worker_pool.submit(
                _async_setup
            )
            success_info = {
                "status": "accepted",
                "request_id": body.request_id,
                "async": True,
            }
            return PlainTextResponse(
                content=json.dumps(success_info, indent=2),
                media_type="application/json",
            )

        num_chunks = _tokenize_and_dispatch()

        success_info = {
            "status": "accepted",
            "request_id": body.request_id,
            "num_chunks": num_chunks,
        }
        return PlainTextResponse(
            content=json.dumps(success_info, indent=2),
            media_type="application/json",
        )
    except Exception as e:
        logger.error("speculative_prefetch failed: %s", str(e))
        error_info = {
            "error": "speculative_prefetch failed",
            "message": str(e),
        }
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=500,
        )


@router.get("/v1/speculative_prefetch_stats/{request_id}")
async def speculative_prefetch_stats(request: Request, request_id: str):
    """Return speculative prefetch hit-rate stats for a given request_id.

    The stats become available after the corresponding ``retrieve_layer()``
    call has classified chunks into speculative hits and disk misses.
    """
    lmcache_adapter = request.app.state.lmcache_adapter
    lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)

    if not lmcache_engine:
        return PlainTextResponse(
            content=json.dumps({"error": "LMCache engine not configured."}),
            media_type="application/json",
            status_code=503,
        )

    stats = lmcache_engine.speculative_stats.pop(request_id, None)
    if stats is None:
        return PlainTextResponse(
            content=json.dumps({
                "error": "not_found",
                "message": f"No stats for request_id={request_id}. "
                           "Stats are available only after inference completes.",
            }),
            media_type="application/json",
            status_code=404,
        )

    stats["request_id"] = request_id
    return PlainTextResponse(
        content=json.dumps(stats),
        media_type="application/json",
    )


class SpeculativePrefetchTokensRequest(BaseModel):
    """Request body for the token-based speculative prefetch endpoint."""

    chunk_tokens_list: List[List[int]]
    request_id: str
    # Number of layers to prefetch. None falls back to config default.
    # 0 means all layers.
    num_prefetch_layers: Optional[int] = None


@router.post("/v1/speculative_prefetch_tokens")
async def speculative_prefetch_tokens(
    request: Request,
    body: SpeculativePrefetchTokensRequest,
):
    """Speculatively prefetch KV caches using pre-tokenized chunk data.

    Similar to ``/v1/speculative_prefetch``, but accepts already-tokenized
    chunk data instead of raw text. This avoids the need for a tokenizer
    on the server side and is useful when the client has already tokenized
    the chunks.

    Args:
        request: The FastAPI request object containing application state.
        body: The request body with pre-tokenized chunks and a correlation
            request_id.

    Returns:
        PlainTextResponse with status "accepted" on success, or an error.
    """
    lmcache_adapter = request.app.state.lmcache_adapter
    lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)

    if not lmcache_engine:
        error_info = {
            "error": "speculative_prefetch API is unavailable",
            "message": "LMCache engine not configured.",
        }
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=503,
        )

    if not lmcache_engine.config.speculative_prefetch_enabled:
        error_info = {
            "error": "speculative_prefetch is disabled",
            "message": "Set speculative_prefetch_enabled=True to use this endpoint.",
        }
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=400,
        )

    try:
        lmcache_engine.speculative_prefetch(
            request_id=body.request_id,
            chunk_tokens_list=body.chunk_tokens_list,
            num_prefetch_layers=body.num_prefetch_layers,
        )

        success_info = {
            "status": "accepted",
            "request_id": body.request_id,
            "num_chunks": len(body.chunk_tokens_list),
        }
        return PlainTextResponse(
            content=json.dumps(success_info, indent=2),
            media_type="application/json",
        )
    except Exception as e:
        logger.error("speculative_prefetch_tokens failed: %s", str(e))
        error_info = {
            "error": "speculative_prefetch_tokens failed",
            "message": str(e),
        }
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=500,
        )

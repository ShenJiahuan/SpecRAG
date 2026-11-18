"""Run RAG inference with FAISS IVF search and SpecRAG speculative KV prefetch."""

import argparse
import concurrent.futures
import json
import pickle
import random
import sqlite3
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import faiss
import numpy as np
import requests as http_requests
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

MODEL_DEFAULT_CHUNK_SEPARATORS: Dict[str, str] = {
    "Qwen/Qwen3-4B": "<|file_sep|>",
    "Qwen/Qwen3-8B": "<|file_sep|>",
    "Qwen/Qwen3-14B": "<|file_sep|>",
    "Qwen/Qwen3-32B": "<|file_sep|>",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "<SPECIAL_999>",
    "zai-org/GLM-4-32B-0414": "<|end_of_image|>",
}


def resolve_default_chunk_separator(model_name: str) -> str:
    if model_name in MODEL_DEFAULT_CHUNK_SEPARATORS:
        return MODEL_DEFAULT_CHUNK_SEPARATORS[model_name]
    raise ValueError(
        f"No default atomic chunk separator registered for model {model_name!r}. "
        f"Pass --chunk-separator explicitly with an atomic single-token string."
    )


def load_queries_from_file(path: Path, num_queries: int) -> List[str]:
    """Load questions from a JSONL (.jsonl) or JSON array (.json) query file."""
    suffix = path.suffix.lower()
    queries: List[str] = []

    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                question = data.get("question_text") or data.get("question") or ""
                if question:
                    queries.append(question)
                    if len(queries) >= num_queries:
                        break
        return queries

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
        for entry in data:
            question = entry.get("question") or entry.get("question_text") or ""
            if question:
                queries.append(question)
                if len(queries) >= num_queries:
                    break
        return queries

    raise ValueError(f"Unsupported query file format '{suffix}'. Use .jsonl or .json.")


class SqliteDocstore:
    """Thread-safe disk-backed docstore using SQLite."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self._db_path, check_same_thread=False)
        return self._local.conn

    def search(self, search: str):
        row = self._conn().execute(
            "SELECT page_content, metadata FROM docs WHERE id = ?", (search,)
        ).fetchone()
        if row is None:
            return f"ID {search} not found."
        return Document(id=str(search), page_content=row[0], metadata=json.loads(row[1]))


def find_index_path(dataset: str) -> Path:
    data_dir = Path(config.DATA_DIR) / dataset
    candidates = [
        "index_opq.faiss_langchain",
        "index.faiss_langchain",
        "index_flat.faiss_langchain",
    ]
    for candidate in candidates:
        filepath = data_dir / candidate
        if filepath.exists():
            return filepath
    raise FileNotFoundError(
        f"Could not find any LangChain FAISS index in {data_dir}. "
        f"Checked: {', '.join(candidates)}"
    )


def format_single_doc(doc) -> str:
    passage_id = doc.metadata.get("passage_id") or "Unknown"
    source = doc.metadata.get("source") or "Unknown"
    return (
        f"--- Document ---\n"
        f"Passage ID: {passage_id}\n"
        f"Source: {source}\n"
        f"Content: {doc.page_content}"
    )


def format_docs(docs: List, separator: str) -> str:
    formatted_text = separator + "\n"
    for doc in docs:
        formatted_text += format_single_doc(doc)
        formatted_text += "\n" + separator + "\n"
    return formatted_text[:-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SpecRAG inference")
    parser.add_argument("--dataset", type=str, default="wiki-18",
                        help="Dataset folder under data/ (used when --index-path is omitted)")
    parser.add_argument("--index-path", type=str, default=None,
                        help="Direct path to the LangChain FAISS index directory")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen3-4B",
                        help="Generation model name served by vLLM")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of retrieved chunks (K)")
    parser.add_argument("--nprobe", type=int, default=1024,
                        help="Number of IVF clusters to scan")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a single query string")
    parser.add_argument("--query-file", type=str, default=None,
                        help="Path to query file (.jsonl or .json)")
    parser.add_argument("--num-queries", type=int, default=20,
                        help="Number of queries to load from --query-file")
    parser.add_argument("--query-shuffle-seed", type=int, default=42,
                        help="Random seed used to shuffle loaded queries")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Concurrent query workers for --query-file (default: 1)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeated evaluations in --query or interactive mode")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--embedding-base-url", type=str, default=None,
                        help="OpenAI-compatible embedding endpoint base URL")
    parser.add_argument("--chat-base-url", type=str, default=None,
                        help="OpenAI-compatible vLLM chat endpoint base URL")
    parser.add_argument("--chunk-separator", type=str, default=None,
                        help="Atomic single-token chunk separator (auto-resolved for supported models)")
    parser.add_argument("--speculative-prefetch", action="store_true",
                        help="Enable SpecRAG speculative KV cache prefetch during FAISS search")
    parser.add_argument("--prefetch-url", type=str, default=None,
                        help="LMCache prefetch endpoint (defaults to http://<chat-host>:7800/v1/speculative_prefetch)")
    parser.add_argument("--adaptive-convergence", action="store_true",
                        help="Use online top-K stability detector to trigger prefetch")
    parser.add_argument("--convergence-window", type=int, default=0,
                        help="Consecutive stable IVF steps to trigger prefetch (0 = max(2, int(0.1 * nprobe)))")
    parser.add_argument("--prefetch-after-ratio", type=float, default=None,
                        help="Fixed trigger ratio in (0, 1) when --adaptive-convergence is not set")
    parser.add_argument("--num-prefetch-layers", type=int, default=0,
                        help="Upper bound on prefetched layers (0 = all layers)")
    parser.add_argument("--warmup-first", action="store_true",
                        help="Materialize chunk KV caches onto disk before timed evaluation")
    parser.add_argument("--warmup-strategy", type=str, choices=["per-chunk", "combined"], default="combined",
                        help="Warmup prompt strategy (default: combined)")
    parser.add_argument("--warmup-search-repeats", type=int, default=3,
                        help="Warmup retrieval passes per query")
    parser.add_argument("--mode", type=str,
                        choices=["baseline", "cacheblend", "cacheblend+speculative", "epic", "epic+speculative"],
                        default=None,
                        help="Serving mode label recorded in output logs")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory to write JSON result logs (default: logs/)")
    parser.add_argument("--show-docs", action="store_true",
                        help="Print retrieved document chunks")
    return parser.parse_args()


class RAGPipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        if self.args.chunk_separator is None:
            self.args.chunk_separator = resolve_default_chunk_separator(self.args.llm_model)
        if self.args.speculative_prefetch and not self.args.adaptive_convergence and self.args.prefetch_after_ratio is None:
            self.args.adaptive_convergence = True

        self._init_index_path()
        self._init_embeddings()
        self._init_vectorstore()
        self._init_llm()
        self._init_prompt()
        self._init_log_dir()

    def _init_index_path(self):
        if self.args.index_path:
            self.index_path = Path(self.args.index_path)
            if not self.index_path.exists():
                raise FileNotFoundError(f"Index path does not exist: {self.index_path}")
        else:
            self.index_path = find_index_path(self.args.dataset)
        print(f"[init] Using index: {self.index_path}")

    def _init_embeddings(self):
        embedding_base_url = self.args.embedding_base_url or config.OPENAI_EMBEDDING_BASE_URL
        print(f"[init] Embedding endpoint: {embedding_base_url} ({config.EMBEDDING_MODEL})")
        self.embeddings_func = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            dimensions=config.EMBEDDING_DIMENSION,
            openai_api_base=embedding_base_url,
            openai_api_key=config.OPENAI_API_KEY,
            check_embedding_ctx_length=False,
        )

    def _init_vectorstore(self):
        index = faiss.read_index(str(self.index_path / "index.faiss"))
        with open(self.index_path / "index_mapping.pkl", "rb") as fh:
            index_to_docstore_id = pickle.load(fh)
        docstore = SqliteDocstore(str(self.index_path / "index.db"))

        self.vectorstore = LangchainFAISS(
            embedding_function=self.embeddings_func,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        scan_index = self.vectorstore.index
        while hasattr(scan_index, "index"):
            scan_index = faiss.downcast_index(scan_index.index)
        if hasattr(scan_index, "nprobe"):
            scan_index.nprobe = self.args.nprobe
            print(f"[init] Loaded {len(index_to_docstore_id)} chunks, nprobe={self.args.nprobe}")

    def _init_llm(self):
        chat_base_url = self.args.chat_base_url or config.OPENAI_CHAT_BASE_URL
        print(f"[init] LLM endpoint: {chat_base_url} ({self.args.llm_model})")
        self.llm = ChatOpenAI(
            model=self.args.llm_model,
            openai_api_base=chat_base_url,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=self.args.temperature,
            max_tokens=32,
            stream_usage=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    def _init_prompt(self):
        self.system_prefix = (
            "You are a helpful assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Do not explicitly mention the context if it is not required. "
            "Your answer MUST be no more than 10 words."
            "\n\nContext:\n"
        )
        template = self.system_prefix + "{context}\nQuestion:\n{question}\n\nAnswer:\n"
        self.prompt = PromptTemplate.from_template(template)

    def _init_log_dir(self):
        self.log_dir = Path(self.args.log_dir) if self.args.log_dir else (Path(__file__).parent.parent / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_mode(self) -> str:
        if self.args.mode:
            return self.args.mode
        return "cacheblend+speculative" if self.args.speculative_prefetch else "baseline"

    def _get_prefetch_url(self) -> str:
        if self.args.prefetch_url:
            return self.args.prefetch_url
        chat_base_url = self.args.chat_base_url or config.OPENAI_CHAT_BASE_URL
        parsed = urlparse(chat_base_url)
        host = parsed.hostname or "localhost"
        lmcache_base = urlunparse((parsed.scheme or "http", f"{host}:7800", "", "", "", ""))
        return lmcache_base.rstrip("/") + "/v1/speculative_prefetch"

    def _query_prefetch_stats(self, request_id: str) -> Optional[dict]:
        base_stats_url = self._get_prefetch_url().replace(
            "/v1/speculative_prefetch", "/v1/speculative_prefetch_stats"
        )
        for rid in (f"chatcmpl-{request_id}", request_id):
            try:
                resp = http_requests.get(f"{base_stats_url}/{rid}", timeout=3)
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                pass
        return None

    def _faiss_ids_to_docs(self, ids: List[int]) -> List[Document]:
        docs = []
        for fid in ids:
            if fid == -1:
                continue
            docstore_id = self.vectorstore.index_to_docstore_id[fid]
            docs.append(self.vectorstore.docstore.search(docstore_id))
        return docs

    def _search_with_prefetch(self, query_embedding: List[float]) -> tuple:
        request_id = str(uuid.uuid4())
        convergence_probe: Optional[int] = None
        prefetch_info: Dict[str, Any] = {}
        prefetch_url = self._get_prefetch_url()

        def _send_prefetch(prompt_prefix: str, req_id: str):
            t = time.time()
            payload: Dict[str, Any] = {
                "prompt_prefix": prompt_prefix,
                "request_id": req_id,
            }
            if self.args.num_prefetch_layers > 0:
                payload["num_prefetch_layers"] = self.args.num_prefetch_layers
            resp = http_requests.post(prefetch_url, json=payload, timeout=5)
            resp.raise_for_status()
            prefetch_info["prefetch_time"] = time.time() - t

        def on_intermediate(query_idx, distances, labels, probes_completed, total_probes, elapsed_ms=-1):
            try:
                nonlocal convergence_probe
                convergence_probe = probes_completed
                prefetch_info["probe_elapsed_ms"] = elapsed_ms
                valid_ids = [lid for lid in labels if lid >= 0]
                docs = self._faiss_ids_to_docs(valid_ids)
                if docs:
                    prefetch_info["_intermediate_docs"] = list(docs)
                    context = format_docs(docs, separator=self.args.chunk_separator)
                    prompt_prefix = self.system_prefix + context
                    t_worker = threading.Thread(
                        target=_send_prefetch,
                        args=(prompt_prefix, request_id),
                        daemon=True,
                    )
                    prefetch_info["_thread"] = t_worker
                    t_worker.start()
            except Exception:
                traceback.print_exc()

        cb = faiss.PyCallbackIntermediateResult(on_intermediate)
        nprobe = self.args.nprobe

        if self.args.adaptive_convergence:
            conv_win = self.args.convergence_window
            if conv_win <= 0:
                conv_win = max(2, int(nprobe * 0.1))
            params = faiss.SearchParametersIVF(
                nprobe=nprobe,
                intermediate_cb=cb,
                adaptive_convergence=True,
                convergence_window=conv_win,
                min_probe_fraction=0.0,
            )
        else:
            ratio = self.args.prefetch_after_ratio if self.args.prefetch_after_ratio is not None else 0.5
            cb_after = max(1, int(nprobe * ratio))
            params = faiss.SearchParametersIVF(
                nprobe=nprobe,
                intermediate_cb=cb,
                intermediate_cb_after_probe=cb_after,
            )

        vector = np.array([query_embedding], dtype=np.float32)
        _, indices = self.vectorstore.index.search(vector, self.args.top_k, params=params)
        docs = self._faiss_ids_to_docs(indices[0].tolist())

        actual_convergence = getattr(params, "convergence_probe_out", -1)
        if actual_convergence >= 0:
            prefetch_info["actual_convergence_probe"] = actual_convergence

        effective_req_id = request_id if convergence_probe is not None else None
        return docs, effective_req_id, convergence_probe, prefetch_info

    def execute_warmup_query(self, query_text: str) -> None:
        """Materialize retrieved and speculated chunk KV caches onto disk."""
        dummy_question = query_text + " .."
        warmup_sets = []
        seen_doc_sets = set()

        def _doc_ids(doc_list):
            return tuple(d.metadata.get("passage_id", d.id) for d in doc_list)

        for _ in range(max(1, self.args.warmup_search_repeats)):
            query_embedding = self.embeddings_func.embed_query(query_text)
            if self.args.speculative_prefetch:
                docs, _, _, info = self._search_with_prefetch(query_embedding)
                t_worker = info.pop("_thread", None)
                if t_worker is not None:
                    t_worker.join(timeout=10)
                spec_docs = info.pop("_intermediate_docs", None) or []
                for candidate in (spec_docs, docs):
                    if candidate and _doc_ids(candidate) not in seen_doc_sets:
                        seen_doc_sets.add(_doc_ids(candidate))
                        warmup_sets.append(candidate)
            else:
                docs = self.vectorstore.similarity_search_by_vector(query_embedding, k=self.args.top_k)
                if docs and _doc_ids(docs) not in seen_doc_sets:
                    seen_doc_sets.add(_doc_ids(docs))
                    warmup_sets.append(docs)

        stream_kwargs = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        for warm_docs in warmup_sets:
            if self.args.warmup_strategy == "per-chunk":
                for doc in warm_docs:
                    prompt_val = self.prompt.invoke({
                        "context": format_docs([doc], separator=self.args.chunk_separator),
                        "question": dummy_question,
                    })
                    for _ in self.llm.stream(prompt_val, **stream_kwargs):
                        pass
            else:
                prompt_val = self.prompt.invoke({
                    "context": format_docs(warm_docs, separator=self.args.chunk_separator),
                    "question": dummy_question,
                })
                for _ in self.llm.stream(prompt_val, **stream_kwargs):
                    pass

    def execute_query(self, query_text: str, verbose: bool = True, skip_save: bool = False) -> dict:
        t0 = time.time()
        query_embedding = self.embeddings_func.embed_query(query_text)
        embedding_time = time.time() - t0

        t1 = time.time()
        request_id = None
        convergence_probe = None
        prefetch_info: Dict[str, Any] = {}
        if self.args.speculative_prefetch:
            docs, request_id, convergence_probe, prefetch_info = self._search_with_prefetch(query_embedding)
        else:
            docs = self.vectorstore.similarity_search_by_vector(query_embedding, k=self.args.top_k)
        faiss_time = time.time() - t1

        if self.args.show_docs and verbose:
            for idx, doc in enumerate(docs, 1):
                print(f"  [{idx}] {doc.metadata.get('passage_id', doc.id)}: {doc.page_content[:120]}...")

        t_fmt = time.time()
        formatted_context = format_docs(docs, separator=self.args.chunk_separator)
        prompt_value = self.prompt.invoke({"context": formatted_context, "question": query_text})
        prompt_time = time.time() - t_fmt

        extra_body: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
        if request_id:
            extra_body["request_id"] = request_id
        if skip_save:
            extra_body["kv_transfer_params"] = {"lmcache.skip_save": True}

        t2 = time.time()
        ttft = None
        response_chunks = []
        input_tokens = None
        output_tokens = None

        for chunk in self.llm.stream(prompt_value, extra_body=extra_body):
            if ttft is None:
                ttft = time.time() - t2
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            response_chunks.append(text)
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                input_tokens = chunk.usage_metadata.get("input_tokens", input_tokens)
                output_tokens = chunk.usage_metadata.get("output_tokens", output_tokens)

        llm_time = time.time() - t2
        response = "".join(response_chunks)

        prefetch_thread = prefetch_info.pop("_thread", None)
        if prefetch_thread is not None:
            prefetch_thread.join(timeout=10)
        prefetch_time = prefetch_info.pop("prefetch_time", None)
        prefetch_info.pop("_intermediate_docs", None)

        prefetch_hit_rate = None
        if self.args.speculative_prefetch and request_id:
            stats = self._query_prefetch_stats(request_id)
            if stats:
                prefetch_hit_rate = stats.get("hit_rate")
                if stats.get("num_prefetch_layers") is not None:
                    prefetch_info["server_prefetch_layers"] = stats["num_prefetch_layers"]
                if stats.get("layerwise_layers_issued") is not None:
                    prefetch_info["layerwise_layers_issued"] = stats["layerwise_layers_issued"]
                if stats.get("e2e_time_ms") is not None:
                    prefetch_info["server_e2e_ms"] = stats["e2e_time_ms"]

        e2e_ttft = (embedding_time + faiss_time + prompt_time + ttft) if ttft is not None else None
        result = {
            "query": query_text,
            "retrieved_docstore_ids": [d.id for d in docs],
            "response": response.strip(),
            "embedding_time": embedding_time,
            "faiss_time": faiss_time,
            "prompt_time": prompt_time,
            "llm_ttft": ttft,
            "e2e_ttft": e2e_ttft,
            "llm_time": llm_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "request_id": request_id,
            "prefetch_time": prefetch_time,
            "convergence_probe": convergence_probe,
            "prefetch_hit_rate": prefetch_hit_rate,
            "prefetch_info": prefetch_info,
        }

        if verbose:
            self._print_single_result(result)
        return result

    def _print_single_result(self, r: dict) -> None:
        print(f"  Answer        : {r['response']}")
        print(f"  Embedding     : {r['embedding_time'] * 1000:.1f} ms")
        print(f"  FAISS Search  : {r['faiss_time'] * 1000:.1f} ms")
        if r.get("convergence_probe") is not None:
            print(f"  Trigger Probe : {r['convergence_probe']}/{self.args.nprobe}")
        pinfo = r.get("prefetch_info", {})
        if pinfo.get("server_prefetch_layers") is not None:
            print(f"  Prefetched L  : {pinfo['server_prefetch_layers']} layers")
        if r.get("prefetch_hit_rate") is not None:
            print(f"  Spec Hit Rate : {r['prefetch_hit_rate']:.1f}%")
        if r["llm_ttft"] is not None:
            print(f"  LLM TTFT      : {r['llm_ttft'] * 1000:.1f} ms")
            print(f"  End-to-End TTFT: {r['e2e_ttft'] * 1000:.1f} ms")

    def _save_log(self, results: List[dict]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = self.args.llm_model.split("/")[-1]
        mode = self._get_mode()
        filename = f"{timestamp}_{self.args.dataset}_{model_short}_{mode}_k{self.args.top_k}_nprobe{self.args.nprobe}.json"
        log_path = self.log_dir / filename

        n = len(results)
        llm_ttfts = [r["llm_ttft"] for r in results if r["llm_ttft"] is not None]
        e2e_ttfts = [r["e2e_ttft"] for r in results if r["e2e_ttft"] is not None]
        hit_rates = [r["prefetch_hit_rate"] for r in results if r["prefetch_hit_rate"] is not None]
        triggers = [r["convergence_probe"] for r in results if r["convergence_probe"] is not None]
        prefetched_l = [
            r["prefetch_info"]["server_prefetch_layers"]
            for r in results
            if r.get("prefetch_info", {}).get("server_prefetch_layers") is not None
        ]

        summary = {
            "num_queries": n,
            "avg_embedding_ms": (sum(r["embedding_time"] for r in results) / n * 1000) if n else 0.0,
            "avg_faiss_ms": (sum(r["faiss_time"] for r in results) / n * 1000) if n else 0.0,
            "avg_llm_ttft_ms": (sum(llm_ttfts) / len(llm_ttfts) * 1000) if llm_ttfts else None,
            "avg_e2e_ttft_ms": (sum(e2e_ttfts) / len(e2e_ttfts) * 1000) if e2e_ttfts else None,
            "avg_trigger_probe": (sum(triggers) / len(triggers)) if triggers else None,
            "avg_prefetched_layers": (sum(prefetched_l) / len(prefetched_l)) if prefetched_l else None,
            "avg_prefetch_hit_rate": (sum(hit_rates) / len(hit_rates)) if hit_rates else None,
        }

        payload = {
            "timestamp": timestamp,
            "config": {
                "dataset": self.args.dataset,
                "index_path": str(self.index_path),
                "llm_model": self.args.llm_model,
                "mode": mode,
                "top_k": self.args.top_k,
                "nprobe": self.args.nprobe,
                "batch_size": self.args.batch_size,
                "speculative_prefetch": self.args.speculative_prefetch,
                "adaptive_convergence": self.args.adaptive_convergence,
            },
            "summary": summary,
            "queries": results,
        }

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"\n--- Summary ({mode}, {n} queries) ---")
        if summary["avg_e2e_ttft_ms"] is not None:
            print(f"Average End-to-End TTFT : {summary['avg_e2e_ttft_ms']:.1f} ms "
                  f"(Embed {summary['avg_embedding_ms']:.1f} ms + "
                  f"FAISS {summary['avg_faiss_ms']:.1f} ms + "
                  f"LLM TTFT {summary['avg_llm_ttft_ms']:.1f} ms)")
        if summary["avg_trigger_probe"] is not None:
            print(f"Average Trigger Probe   : {summary['avg_trigger_probe']:.1f}/{self.args.nprobe}")
        if summary["avg_prefetched_layers"] is not None:
            print(f"Average Prefetched L    : {summary['avg_prefetched_layers']:.1f}")
        if summary["avg_prefetch_hit_rate"] is not None:
            print(f"Average Spec Hit Rate   : {summary['avg_prefetch_hit_rate']:.1f}%")
        print(f"Results saved to: {log_path}")
        return log_path

    def run(self):
        if self.args.query:
            if self.args.warmup_first:
                print("[warmup] Materializing KV cache for query...")
                self.execute_warmup_query(self.args.query)
                time.sleep(5)
            results = []
            for i in range(max(1, self.args.repeat)):
                print(f"\n[Run {i + 1}/{max(1, self.args.repeat)}] {self.args.query}")
                results.append(self.execute_query(self.args.query, skip_save=True))
            self._save_log(results)
            return

        if self.args.query_file:
            queries = load_queries_from_file(Path(self.args.query_file), self.args.num_queries)
            if not queries:
                raise RuntimeError(f"No queries loaded from {self.args.query_file}")
            random.Random(self.args.query_shuffle_seed).shuffle(queries)

            if self.args.warmup_first:
                print(f"[warmup] Materializing KV caches across {len(queries)} queries...")
                for idx, q in enumerate(queries, 1):
                    print(f"  Warmup {idx}/{len(queries)}: {q[:60]}...")
                    self.execute_warmup_query(q)
                time.sleep(5)

            if self.args.batch_size <= 1:
                results = []
                for idx, q in enumerate(queries, 1):
                    print(f"\n[Query {idx}/{len(queries)}] {q}")
                    results.append(self.execute_query(q, verbose=True, skip_save=True))
            else:
                print(f"\n[eval] Running {len(queries)} queries with concurrency={self.args.batch_size}...")
                results_map: Dict[int, dict] = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.batch_size) as pool:
                    fut_to_idx = {
                        pool.submit(self.execute_query, q, False, True): i
                        for i, q in enumerate(queries)
                    }
                    for fut in concurrent.futures.as_completed(fut_to_idx):
                        results_map[fut_to_idx[fut]] = fut.result()
                results = [results_map[i] for i in sorted(results_map)]

            self._save_log(results)
            return

        print("Interactive mode (type 'quit' to exit).")
        while True:
            try:
                q = input("\nQuery: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"q", "quit", "exit"}:
                break
            if q:
                self.execute_query(q)


def main():
    args = parse_args()
    pipeline = RAGPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()

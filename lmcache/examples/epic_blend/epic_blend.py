# SPDX-License-Identifier: Apache-2.0
"""End-to-end EPIC blending example.

This script demonstrates EPIC's position-independent caching with
V-based importance scoring. It:
1. Sends a first request to populate the KV cache
2. Sends a second request sharing some chunks with the first
3. The shared chunks are loaded from cache and blended using EPIC

Usage:
    LMCACHE_BLENDING_ALGORITHM=epic python examples/epic_blend/epic_blend.py

Environment variables:
    LMCACHE_BLENDING_ALGORITHM: Must be "epic" for EPIC blending
    LMCACHE_EPIC_RECOMP_RATIO: Recomputation ratio (default: 0.16)
"""

# Standard
from dataclasses import asdict
import argparse
import contextlib
import os
import time

# Third Party
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def setup_environment_variables(
    use_disk: bool = False,
    blend_special_str: str = " # # ",
    recomp_ratio: float = 0.16,
):
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"

    # EPIC-specific config
    os.environ["LMCACHE_BLENDING_ALGORITHM"] = "epic"
    os.environ["LMCACHE_EPIC_RECOMP_RATIO"] = str(recomp_ratio)

    if use_disk:
        os.environ["LMCACHE_LOCAL_CPU"] = "False"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
    else:
        os.environ["LMCACHE_LOCAL_CPU"] = "True"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
        enforce_eager=True,
        max_num_batched_tokens=2048,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[int],
    sampling_params: SamplingParams,
    req_str: str,
):
    start = time.time()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt}, sampling_params=sampling_params
    )
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    elapsed = time.time() - start
    print(f"Generation took {elapsed:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def parse_args():
    parser = argparse.ArgumentParser(description="EPIC blending example")
    parser.add_argument(
        "-d", "--use-disk", action="store_true",
        help="Use disk as backend (default: False)",
    )
    parser.add_argument(
        "-b", "--blend-special-str", default="# #",
        help="Special separators between chunks (default: '# #')",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-4B",
    )
    parser.add_argument(
        "--recomp-ratio", type=float, default=0.16,
        help="EPIC recomputation ratio (default: 0.16)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    lmcache_connector = "LMCacheConnectorV1"
    model = args.model

    setup_environment_variables(
        args.use_disk, args.blend_special_str, args.recomp_ratio
    )

    tokenizer = AutoTokenizer.from_pretrained(model)

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        blend_special_str = tokenizer.encode(
            os.getenv("LMCACHE_BLEND_SPECIAL_STR")
        )

        warmup_prompt = tokenizer.encode("Hello, world!")

        first_prompt = tokenizer.encode(
            "Shanghai Jiao Tong University (SJTU) is a public university "
            "in Shanghai, China." * 8
        )
        second_prompt = tokenizer.encode(
            "SJTU was founded as Nanyang Public School in 1896." * 8
        )
        third_prompt = tokenizer.encode(
            "It is affiliated with the Ministry of Education of China." * 8
        )
        fourth_prompt = tokenizer.encode("The university is part of")
        fifth_prompt = tokenizer.encode(
            "It is affiliated with the Ministry of Education of P.R.China." * 8
        )

        query1 = (
            first_prompt
            + blend_special_str
            + second_prompt
            + blend_special_str
            + fourth_prompt
        )
        query2 = (
            first_prompt
            + blend_special_str
            + fifth_prompt
            + blend_special_str
            + second_prompt
            + blend_special_str
            + fourth_prompt
        )
        query3 = (
            third_prompt
            + blend_special_str
            + fifth_prompt
            + blend_special_str
            + second_prompt
            + blend_special_str
            + fourth_prompt
        )

        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=100)

        print("=== EPIC Blending Example ===")
        print(f"Model: {model}")
        print(f"Recomp ratio: {args.recomp_ratio}")
        print()

        print_output(llm, warmup_prompt, sampling_params, "warmup")
        print_output(llm, query1, sampling_params, "first (populates cache)")

        time.sleep(1)
        print_output(llm, query2, sampling_params, "second (EPIC blending)")

        time.sleep(1)
        print_output(llm, query3, sampling_params, "third (EPIC blending)")


if __name__ == "__main__":
    main()

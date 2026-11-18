# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Iterable, List, Optional, Tuple, Union
import abc
import logging

# Third Party
from transformers import AutoTokenizer
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)

NONE_HASH: int


class TokenDatabase(metaclass=abc.ABCMeta):
    """TokenDatabase is used to convert input tokens into list of
    cache engine keys. There are multiple ways to implement this:

    - ChunkedTokenDatabase: It processes tokens into chunks and convert
    each chunk into a cache engine key using prefix hash.

    - SegmentTokenDatabase: It processes tokens into segments based on
    special separators and convert each segment into a cache engine key.
    """

    @abc.abstractmethod
    def __init__(
        self,
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheEngineMetadata] = None,
    ):
        global NONE_HASH

        vllm_is_available = True
        try:
            # Third Party
            from vllm.utils.hashing import sha256, sha256_cbor
        except ImportError:
            # sha256, sha256_cbor are available through vLLM only
            vllm_is_available = False

        hash_algorithm: str
        if config is not None:
            hash_algorithm = config.pre_caching_hash_algorithm
        else:  # Default value
            hash_algorithm = "builtin"  # fallback to builtin hash

        # Need to support vLLM hashing functions at a minimum
        self.hash_func = (
            sha256_cbor
            if hash_algorithm == "sha256_cbor" and vllm_is_available
            else sha256
            if hash_algorithm == "sha256" and vllm_is_available
            else hash
        )

        # NOTE: For centralized cache sharing, ensure PYTHONHASHSEED is
        # set consistently across all processes (e.g., export PYTHONHASHSEED=0).
        try:
            # Third Party
            from vllm.v1.core import kv_cache_utils

            kv_cache_utils.init_none_hash(self.hash_func)
            NONE_HASH = kv_cache_utils.NONE_HASH
        except (ImportError, AttributeError):
            NONE_HASH = 0

        self.metadata = metadata

    @abc.abstractmethod
    def process_tokens(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        request_configs: Optional[dict] = None,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, int]]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: The tokens to process.

        :param Optional[List[int]] hashes: The hashes to process. If provided,
            it will be used instead of tokens to generate cache engine keys.

        :param Optional[List[int]] offsets: The number of tokens in each chunk.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param bool make_key: Whether to make the cache engine key or not.
            If False, the hash value will be returned instead.

        :param Optional[dict] request_configs: The configs of the request.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.
        """

        raise NotImplementedError

    def _make_key_by_hash(
        self, chunk_hash: int, request_configs: Optional[dict] = None
    ):
        assert self.metadata is not None
        return CacheEngineKey(
            self.metadata.fmt,
            self.metadata.model_name,
            self.metadata.world_size,
            self.metadata.worker_id,
            chunk_hash,
            request_configs,
        )

    def _hash_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        prefix_hash: Optional[int] = None,
        extra_keys: Optional[list[Any]] = None,
    ) -> int:
        if isinstance(tokens, torch.Tensor):
            tokens_tuple = tuple(tokens.cpu().tolist())
        elif isinstance(tokens, list):
            tokens_tuple = tuple(tokens)
        else:
            raise ValueError(f"Unsupported tokens type: {type(tokens)}")

        # Ignore extra keys for now
        # Extra keys are for multi-modal inputs and
        # request specific metadata (e.g., LoRA ID).
        return self.hash_func((prefix_hash, tokens_tuple, extra_keys))


class ChunkedTokenDatabase(TokenDatabase):
    def __init__(
        self,
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheEngineMetadata] = None,
    ):
        super(ChunkedTokenDatabase, self).__init__(config, metadata)

        if config is not None:
            self.chunk_size = config.chunk_size
            self.save_unfull_chunk = config.save_unfull_chunk

            # Check for cross-process cache sharing setup
            # Standard
            import os

            if os.getenv("PYTHONHASHSEED") is None:
                if config.remote_url is not None:
                    logger.warning(
                        "Centralized cache sharing detected "
                        "but PYTHONHASHSEED not set. "
                        "For consistent caching, set: export PYTHONHASHSEED=0 "
                        "before the engine starts."
                    )
                if config.enable_pd:
                    logger.error(
                        "P/D Disaggregation detected "
                        "but PYTHONHASHSEED not set. "
                        "For consistent caching, set: export PYTHONHASHSEED=0 "
                        "before the engine starts. "
                        "This will cause incorrect KV cache transfer."
                    )
        else:  # Default values
            self.chunk_size = 256
            self.save_unfull_chunk = True

    def _get_init_hash(self) -> int:
        return NONE_HASH

    def _chunk_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
    ) -> Iterable[Union[torch.Tensor, List[int]]]:
        """
        Chunk the tokens into chunks of size self.chunk_size.

        :param tokens: the input tokens, with shape [seq_len]
            device: the target device after chunking

        :return: a generator of chunks of tokens, each with
                shape [chunk_size]
        """
        end = (
            len(tokens)
            if self.save_unfull_chunk
            else (len(tokens) - len(tokens) % self.chunk_size)
        )
        for i in range(0, end, self.chunk_size):
            yield tokens[i : i + self.chunk_size]

    def _prefix_hash(
        self,
        token_chunks: Iterable[Union[torch.Tensor, List[int]]],
    ) -> Iterable[int]:
        prefix_hash = self._get_init_hash()
        for token_chunk in token_chunks:
            prefix_hash = self._hash_tokens(token_chunk, prefix_hash)
            yield prefix_hash

    @_lmcache_nvtx_annotate
    def process_tokens(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        request_configs: Optional[dict] = None,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, int]]]:
        """Process the tokens/hashes and return the corresponding cache engine keys.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: The tokens to process.

        :param Optional[List[int]] hashes: The hashes to process. If provided,
            it will be used instead of tokens to generate cache engine keys.

        :param Optional[List[int]] offsets: The number of tokens in each chunk.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param bool make_key: Whether to make the cache engine key or not.
            If False, the hash value will be returned instead.

        :param Optional[dict] request_configs: The configs of the request.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()
        else:
            num_falses = 0

        if num_falses % self.chunk_size != 0:
            raise ValueError(
                "The number of Falses in the mask is not a multiple of the chunk size."
            )

        if tokens is not None:
            total_len = len(tokens)
            token_chunks = self._chunk_tokens(tokens)
            prefix_hashes = self._prefix_hash(token_chunks)
            for chunk_id, hash_val in enumerate(prefix_hashes):
                start_idx = chunk_id * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, total_len)
                if start_idx < num_falses:
                    continue
                else:
                    if make_key:
                        yield (
                            start_idx,
                            end_idx,
                            self._make_key_by_hash(hash_val, request_configs),
                        )
                    else:
                        yield start_idx, end_idx, hash_val
        elif hashes is not None:
            assert offsets is not None, (
                "If hashes are provided, offsets must also be provided."
            )
            start_idx = 0
            for hash_val, offset in zip(hashes, offsets, strict=False):
                end_idx = start_idx + offset
                if make_key:
                    yield (
                        start_idx,
                        end_idx,
                        self._make_key_by_hash(hash_val, request_configs),
                    )
                else:
                    yield start_idx, end_idx, hash_val
                start_idx = end_idx
        else:
            raise ValueError("Either tokens or hashes must be provided.")


class SegmentTokenDatabase(TokenDatabase):
    """
    Currently, we still use special separators to identify chunks.
    In the future, we might need to implement a fast substring match.
    """

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        super(SegmentTokenDatabase, self).__init__(config, metadata)

        self.tokenizer = AutoTokenizer.from_pretrained(metadata.model_name)
        self.sep_str = config.blend_special_str

    def _hash_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        prefix_hash: Optional[int] = None,
        extra_keys: Optional[list[Any]] = None,
    ) -> int:
        """Override to hash decoded TEXT instead of raw token IDs.

        BPE tokenization is context-sensitive: the same text produces
        different token sequences depending on surrounding context.
        By hashing the decoded text we ensure that chunks with identical
        text always get the same hash, regardless of how they were
        tokenized (standalone vs. part of a larger prompt).
        """
        if isinstance(tokens, torch.Tensor):
            token_list = tokens.cpu().tolist()
        elif isinstance(tokens, list):
            token_list = list(tokens)
        else:
            raise ValueError(f"Unsupported tokens type: {type(tokens)}")

        text = self.tokenizer.decode(token_list, skip_special_tokens=False)
        return self.hash_func((prefix_hash, text, extra_keys))

    def _text_based_split(
        self, tokens: torch.Tensor
    ) -> Iterable[Tuple[int, int]]:
        """Find separator positions by decoding tokens to text, then
        map text-level separator boundaries back to token boundaries.

        BPE tokenization is context-sensitive: the same separator string
        can produce different token sequences depending on surrounding
        characters.  Token-level sliding-window matching therefore fails
        when the BPE merges separator characters with their neighbours.
        This method avoids the problem by operating on decoded text.

        Yields (start_token_idx, end_token_idx) for each chunk, where
        the ranges exclude the separator tokens.
        """
        token_list = tokens.tolist()
        num_tokens = len(token_list)

        if not self.sep_str or num_tokens == 0:
            yield 0, num_tokens
            return

        # Build a cumulative character-length table so we can map from
        # character offset back to token index.
        # char_offsets[i] = number of decoded characters BEFORE token i.
        #
        # Use the Rust backend tokenizer's per-token decode (~1ms for 4k
        # tokens).  Unlike convert_ids_to_tokens (which returns raw BPE
        # pieces like 'ĠcafÃ©'), backend.decode applies the ByteLevel
        # decoder so lengths match the full-sequence decode for all
        # tokenizers including those with byte-fallback / NFC
        # normalization (e.g. Qwen).
        backend = getattr(self.tokenizer, '_tokenizer', None)
        char_offsets = [0] * (num_tokens + 1)
        if backend is not None:
            for i in range(num_tokens):
                char_offsets[i + 1] = char_offsets[i] + len(
                    backend.decode([token_list[i]],
                                   skip_special_tokens=False))
        else:
            for i in range(num_tokens):
                char_offsets[i + 1] = char_offsets[i] + len(
                    self.tokenizer.decode([token_list[i]],
                                          skip_special_tokens=False))

        # Decode full sequence and find separator positions in text.
        full_text = self.tokenizer.decode(
            token_list, skip_special_tokens=False
        )
        sep_len_chars = len(self.sep_str)

        # Collect all separator start positions in the decoded text.
        sep_positions: list[int] = []
        search_start = 0
        while True:
            pos = full_text.find(self.sep_str, search_start)
            if pos < 0:
                break
            sep_positions.append(pos)
            search_start = pos + sep_len_chars

        if not sep_positions:
            yield 0, num_tokens
            return

        def _char_to_token(char_pos: int) -> int:
            """Find the token index whose decoded text covers char_pos.
            Returns the first token i where char_offsets[i] <= char_pos
            < char_offsets[i+1].  If char_pos falls exactly on a token
            boundary, return that token index."""
            lo, hi = 0, num_tokens
            while lo < hi:
                mid = (lo + hi) // 2
                if char_offsets[mid + 1] <= char_pos:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        chunk_start_tok = 0
        for sep_char_pos in sep_positions:
            sep_end_char = sep_char_pos + sep_len_chars
            # Token where the separator starts
            sep_start_tok = _char_to_token(sep_char_pos)
            # Token right after the separator ends
            sep_end_tok = _char_to_token(sep_end_char)

            # If the separator start falls mid-token, that token
            # straddles content and separator — exclude it from
            # the preceding chunk.
            if char_offsets[sep_start_tok] < sep_char_pos:
                sep_start_tok = sep_start_tok  # already excluded (range is exclusive)

            # If the separator end falls mid-token, that token
            # straddles separator and content (e.g. BPE merged
            # ` #` + `\n` into one token).  Skip it so the next
            # chunk doesn't start with separator residue.
            if (sep_end_tok < num_tokens
                    and char_offsets[sep_end_tok] < sep_end_char):
                sep_end_tok += 1

            if sep_start_tok > chunk_start_tok:
                yield chunk_start_tok, sep_start_tok
            chunk_start_tok = sep_end_tok

        # Last chunk after the final separator.
        if chunk_start_tok < num_tokens:
            yield chunk_start_tok, num_tokens

    def process_tokens(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        request_configs: Optional[dict] = None,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, int]]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param Union[torch.Tensor, List[int]] tokens: The tokens to process.

        :param Optional[List[int]] hashes: The hashes to process. If provided,
            it will be used instead of tokens to generate cache engine keys.

        :param Optional[List[int]] offsets: The number of tokens in each chunk.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param bool make_key: Whether to make the cache engine key or not.
            If False, the hash value will be returned instead.

        :param Optional[dict] request_configs: The configs of the request.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key for the tokens.

        """

        if tokens is not None:
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long, device="cpu")
            else:
                tokens = tokens.to(device="cpu", dtype=torch.long)

            if mask is not None:
                num_falses = mask.numel() - mask.long().sum().item()
            else:
                num_falses = 0
            assert num_falses < len(tokens), (
                "The number of Falses in the mask shouldn't "
                "be less than the length of tokens."
            )

            for start_idx, end_idx in self._text_based_split(tokens):
                if start_idx < num_falses:
                    continue
                token_chunk = tokens[start_idx:end_idx]
                chunk_hash = self._hash_tokens(token_chunk)
                if logger.isEnabledFor(logging.DEBUG):
                    chunk_text = self.tokenizer.decode(
                        token_chunk.tolist(), skip_special_tokens=False
                    )
                    logger.debug(
                        "SegmentTokenDatabase chunk [%d:%d] hash=%d "
                        "text_preview=%.120r",
                        start_idx, end_idx, chunk_hash,
                        chunk_text,
                    )
                if make_key:
                    yield (
                        start_idx,
                        end_idx,
                        self._make_key_by_hash(
                            chunk_hash, request_configs
                        ),
                    )
                else:
                    yield start_idx, end_idx, chunk_hash
        elif hashes is not None:
            assert offsets is not None, (
                "If hashes are provided, offsets must also be provided."
            )
            start_idx = 0
            for hash_val, offset in zip(hashes, offsets, strict=False):
                end_idx = start_idx + offset
                if make_key:
                    yield (
                        start_idx,
                        end_idx,
                        self._make_key_by_hash(hash_val, request_configs),
                    )
                else:
                    yield start_idx, end_idx, hash_val
                start_idx = end_idx
        else:
            raise ValueError("Either tokens or hashes must be provided.")

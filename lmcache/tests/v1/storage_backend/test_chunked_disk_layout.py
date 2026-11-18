# SPDX-License-Identifier: Apache-2.0
"""Tests for the chunked (all-layers-in-one-file) disk layout."""

import asyncio
import ctypes
import os
import shutil
import tempfile
import threading
import time

import pytest
import torch

from lmcache.utils import (
    CacheEngineKey,
    ChunkedCacheEngineKey,
    DiskCacheMetadata,
    LayerCacheEngineKey,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend


# --------------- helpers ---------------


def _make_config(disk_path: str, max_disk_gb: float = 1.0):
    return LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_disk=disk_path,
        max_local_disk_size=max_disk_gb,
        lmcache_instance_id="test_chunked",
    )


def _make_key(chunk_id: int = 0) -> CacheEngineKey:
    return CacheEngineKey("vllm", "test_model", 1, 0, chunk_id)


def _alloc_memory_obj(shape=(2, 16, 128), dtype=torch.bfloat16):
    from lmcache.v1.memory_management import AdHocMemoryAllocator

    allocator = AdHocMemoryAllocator(device="cpu")
    return allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_2TD)


def _start_loop():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    return loop


# --------------- fixtures ---------------


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def backend(temp_dir, memory_allocator):
    loop = _start_loop()
    config = _make_config(temp_dir)
    cpu_backend = LocalCPUBackend(config, memory_allocator=memory_allocator)
    backend = LocalDiskBackend(
        config=config,
        loop=loop,
        local_cpu_backend=cpu_backend,
        dst_device="cpu",
    )
    yield backend
    backend.close()


# --------------- Key tests ---------------


class TestChunkedCacheEngineKey:
    def test_to_string_contains_chunked(self):
        k = _make_key(42)
        ck = k.to_chunked_key(80)
        s = ck.to_string()
        assert "@chunked" in s
        assert "42" not in s or "2a" in s  # hex

    def test_round_trip(self):
        k = _make_key(0xDEAD)
        ck = k.to_chunked_key(80)
        parsed = ChunkedCacheEngineKey.from_string(ck.to_string())
        assert parsed.chunk_hash == 0xDEAD
        assert parsed.fmt == "vllm"

    def test_hash_equality(self):
        k = _make_key(99)
        ck1 = k.to_chunked_key(80)
        ck2 = k.to_chunked_key(80)
        assert ck1 == ck2
        assert hash(ck1) == hash(ck2)

    def test_different_num_layers(self):
        k = _make_key(99)
        ck80 = k.to_chunked_key(80)
        ck40 = k.to_chunked_key(40)
        assert ck80 != ck40

    def test_not_equal_to_layer_key(self):
        k = _make_key(99)
        ck = k.to_chunked_key(80)
        lk = k.split_layers(1)[0]
        assert ck != lk


class TestDiskCacheMetadata:
    def test_default_values(self):
        m = DiskCacheMetadata("/tmp/x.pt", 100)
        assert m.chunked is False
        assert m.num_layers == 1

    def test_chunked_values(self):
        m = DiskCacheMetadata("/tmp/x.pt", 100, chunked=True, num_layers=80)
        assert m.chunked is True
        assert m.num_layers == 80


# --------------- Write tests ---------------


class TestChunkedWrite:
    def test_write_and_read_back(self, backend, temp_dir):
        """Write a chunked file and verify its contents."""
        num_layers = 4
        k = _make_key(1)
        ck = k.to_chunked_key(num_layers)
        shape = torch.Size([2, 16, 128])
        dtype = torch.bfloat16

        # Create per-layer MemoryObjs with distinct data
        objs = []
        for lid in range(num_layers):
            obj = _alloc_memory_obj(shape, dtype)
            obj.tensor.fill_(lid + 1.0)
            objs.append(obj)

        # Write chunked
        backend.async_save_chunked_to_disk(key=ck, memory_objs=objs)

        # Verify file exists
        path = backend._key_to_path(ck)
        assert os.path.exists(path)

        # Verify metadata
        assert ck in backend.dict
        meta = backend.dict[ck]
        assert meta.chunked is True
        assert meta.num_layers == num_layers

        # Read back and verify
        layer_size = 2 * 16 * 128 * 2  # shape * element_size(bfloat16)
        total_size = layer_size * num_layers
        assert meta.size == total_size

        buf = (ctypes.c_ubyte * total_size)()
        with open(path, "rb") as f:
            f.readinto(buf)

        for lid in range(num_layers):
            offset = lid * layer_size
            t = torch.frombuffer(
                bytearray(bytes(buf[offset : offset + layer_size])),
                dtype=dtype,
            ).reshape(shape)
            expected_val = lid + 1.0
            assert torch.allclose(
                t, torch.full(shape, expected_val, dtype=dtype)
            ), f"Layer {lid} data mismatch"


# --------------- Read tests ---------------


class TestChunkedRead:
    def _write_chunked_file(self, backend, chunk_id, num_layers, shape, dtype):
        """Helper: write a chunked file and return the key."""
        k = _make_key(chunk_id)
        ck = k.to_chunked_key(num_layers)
        objs = []
        for lid in range(num_layers):
            obj = _alloc_memory_obj(shape, dtype)
            obj.tensor.fill_(lid + 1.0)
            objs.append(obj)
        backend.async_save_chunked_to_disk(key=ck, memory_objs=objs)
        return ck

    def test_full_chunked_read(self, backend):
        """Read all layers from a chunked file."""
        num_layers = 4
        shape = torch.Size([2, 16, 128])
        dtype = torch.bfloat16
        ck = self._write_chunked_file(backend, 10, num_layers, shape, dtype)

        # Allocate read targets
        read_objs = [_alloc_memory_obj(shape, dtype) for _ in range(num_layers)]

        # Call synchronous read method directly
        result = backend.async_load_chunked_from_disk(
            path=backend._key_to_path(ck),
            key=ck,
            num_layers=num_layers,
            memory_objs=read_objs,
        )

        assert len(result) == num_layers
        for lid in range(num_layers):
            expected = lid + 1.0
            assert torch.allclose(
                result[lid].tensor,
                torch.full(shape, expected, dtype=dtype),
            ), f"Layer {lid} read mismatch"

    def test_partial_chunked_read(self, backend):
        """Read a subset of layers from a chunked file using pread."""
        num_layers = 8
        shape = torch.Size([2, 16, 128])
        dtype = torch.bfloat16
        ck = self._write_chunked_file(backend, 20, num_layers, shape, dtype)

        start_layer = 3
        end_layer = 6
        num_to_read = end_layer - start_layer

        read_objs = [_alloc_memory_obj(shape, dtype) for _ in range(num_to_read)]

        result = backend.async_load_chunked_partial_from_disk(
            path=backend._key_to_path(ck),
            key=ck,
            start_layer=start_layer,
            end_layer=end_layer,
            memory_objs=read_objs,
        )

        assert len(result) == num_to_read
        for i in range(num_to_read):
            expected = start_layer + i + 1.0
            assert torch.allclose(
                result[i].tensor,
                torch.full(shape, expected, dtype=dtype),
            ), f"Partial layer {start_layer + i} read mismatch"

    def test_contains_chunked_key(self, backend):
        """Verify contains() works with ChunkedCacheEngineKey."""
        num_layers = 4
        shape = torch.Size([2, 16, 128])
        dtype = torch.bfloat16
        ck = self._write_chunked_file(backend, 30, num_layers, shape, dtype)

        assert backend.contains(ck)
        # Per-layer key should NOT match
        lk = _make_key(30).split_layers(1)[0]
        assert not backend.contains(lk)

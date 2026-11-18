# SPDX-License-Identifier: Apache-2.0
"""Tests for StorageManager.batched_get_all_layers()"""

# Standard
import os
import shutil
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.storage_manager import StorageManager


NUM_LAYERS = 4
NUM_HEADS = 8
HEAD_SIZE = 16
HIDDEN_DIM = NUM_HEADS * HEAD_SIZE  # 128
CHUNK_SIZE = 16


def create_test_metadata():
    return LMCacheEngineMetadata(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        fmt="vllm",
        kv_dtype=torch.bfloat16,
        kv_shape=(NUM_LAYERS, 2, CHUNK_SIZE, NUM_HEADS, HEAD_SIZE),
    )


def create_test_key(layer_id: int, chunk_hash: int) -> CacheEngineKey:
    return CacheEngineKey("vllm", "test_model", layer_id, CHUNK_SIZE, chunk_hash)


@pytest.fixture
def temp_disk_path():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def storage_manager_with_disk(temp_disk_path, memory_allocator):
    """Create a StorageManager with LocalCPUBackend + LocalDiskBackend."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=CHUNK_SIZE,
        local_disk=temp_disk_path,
        max_local_disk_size=1.0,
        lmcache_instance_id="test_instance",
        use_layerwise=True,
    )
    metadata = create_test_metadata()
    event_manager = EventManager()

    sm = StorageManager(
        config=config,
        metadata=metadata,
        event_manager=event_manager,
    )
    sm.post_init()
    yield sm
    sm.close()


def _wait_disk_put_tasks(sm, max_wait=10.0):
    """Wait for all async disk put tasks to complete."""
    disk_backend = sm.storage_backends.get("LocalDiskBackend")
    if disk_backend is None:
        return
    elapsed = 0.0
    interval = 0.05
    while elapsed < max_wait:
        with disk_backend.disk_worker.put_lock:
            if len(disk_backend.disk_worker.put_tasks) == 0:
                return
        time.sleep(interval)
        elapsed += interval
    raise TimeoutError("Disk put tasks did not complete in time")


def _write_chunk_to_disk(sm, key, data_tensor):
    """Write a single chunk's KV data to disk via batched_put.

    data_tensor shape: [2, 1, CHUNK_SIZE, HIDDEN_DIM] (KV_2LTD single-layer)
    """
    mem_obj = sm.allocate(
        shape=data_tensor.shape,
        dtype=data_tensor.dtype,
        fmt=MemoryFormat.KV_2LTD,
        eviction=True,
        busy_loop=True,
    )
    assert mem_obj is not None, "Failed to allocate memory for test data"
    mem_obj.tensor.copy_(data_tensor)
    mem_obj.ref_count_up()
    sm.batched_put([key], [mem_obj])
    return mem_obj


class TestBatchedGetAllLayers:
    """Tests for StorageManager.batched_get_all_layers()"""

    def test_basic_load_all_layers(self, storage_manager_with_disk):
        """Test loading all layers of multiple chunks in one batch."""
        sm = storage_manager_with_disk

        num_chunks = 3
        chunk_hashes = [1000 + i for i in range(num_chunks)]

        # Create and store test data: [num_layers][num_chunks] keys
        keys_layer_major = []
        expected_data = {}  # (layer, chunk) -> tensor

        for layer_id in range(NUM_LAYERS):
            layer_keys = []
            for chunk_idx, chunk_hash in enumerate(chunk_hashes):
                key = create_test_key(layer_id, chunk_hash)
                layer_keys.append(key)

                # Create unique data per (layer, chunk)
                data = torch.randn(
                    2, 1, CHUNK_SIZE, HIDDEN_DIM, dtype=torch.bfloat16
                )
                data.fill_(float(layer_id * 100 + chunk_idx))
                _write_chunk_to_disk(sm, key, data)
                expected_data[(layer_id, chunk_idx)] = data

            keys_layer_major.append(layer_keys)

        # Wait for async disk writes to complete
        _wait_disk_put_tasks(sm)

        # Call batched_get_all_layers
        future = sm.batched_get_all_layers(
            keys_layer_major=keys_layer_major,
            location="LocalDiskBackend",
        )

        # Wait for result
        result = future.result(timeout=30)
        assert result is not None, "batched_get_all_layers returned None"
        assert len(result) == NUM_LAYERS * num_chunks, (
            f"Expected {NUM_LAYERS * num_chunks} memory objects, got {len(result)}"
        )

        # Verify data matches: results are flattened in layer-major order
        for layer_id in range(NUM_LAYERS):
            for chunk_idx in range(num_chunks):
                flat_idx = layer_id * num_chunks + chunk_idx
                mem_obj = result[flat_idx]
                assert mem_obj is not None, (
                    f"MemoryObj at layer={layer_id}, chunk={chunk_idx} is None"
                )
                expected = expected_data[(layer_id, chunk_idx)]
                assert torch.equal(mem_obj.tensor, expected), (
                    f"Data mismatch at layer={layer_id}, chunk={chunk_idx}"
                )

    def test_consistency_with_layerwise(self, storage_manager_with_disk):
        """Verify batched_get_all_layers returns same data as
        layerwise_batched_get."""
        sm = storage_manager_with_disk

        num_chunks = 2
        chunk_hashes = [2000 + i for i in range(num_chunks)]
        keys_layer_major = []
        expected_data = {}

        for layer_id in range(NUM_LAYERS):
            layer_keys = []
            for chunk_idx, chunk_hash in enumerate(chunk_hashes):
                key = create_test_key(layer_id, chunk_hash)
                layer_keys.append(key)
                data = torch.randn(
                    2, 1, CHUNK_SIZE, HIDDEN_DIM, dtype=torch.bfloat16
                )
                data.fill_(float(layer_id * 10 + chunk_hash))
                _write_chunk_to_disk(sm, key, data)
                expected_data[(layer_id, chunk_idx)] = data
            keys_layer_major.append(layer_keys)

        _wait_disk_put_tasks(sm)

        # Get via batched_get_all_layers
        all_layers_future = sm.batched_get_all_layers(
            keys_layer_major=keys_layer_major,
            location="LocalDiskBackend",
        )
        all_layers_result = all_layers_future.result(timeout=30)

        # Verify all_layers results are correct
        assert len(all_layers_result) == NUM_LAYERS * num_chunks

        # Verify data content against expected values
        for layer_id in range(NUM_LAYERS):
            for chunk_idx in range(num_chunks):
                flat_idx = layer_id * num_chunks + chunk_idx
                expected = expected_data[(layer_id, chunk_idx)]
                assert torch.equal(all_layers_result[flat_idx].tensor, expected), (
                    f"Data mismatch at layer={layer_id}, chunk={chunk_idx}"
                )

    def test_single_chunk_single_layer(self, storage_manager_with_disk):
        """Test edge case: single chunk, single layer."""
        sm = storage_manager_with_disk

        key = create_test_key(layer_id=0, chunk_hash=3000)
        data = torch.randn(2, 1, CHUNK_SIZE, HIDDEN_DIM, dtype=torch.bfloat16)
        data.fill_(42.0)
        _write_chunk_to_disk(sm, key, data)

        _wait_disk_put_tasks(sm)

        future = sm.batched_get_all_layers(
            keys_layer_major=[[key]],
            location="LocalDiskBackend",
        )
        result = future.result(timeout=30)
        assert len(result) == 1
        assert torch.equal(result[0].tensor, data)

    def test_with_pre_allocated_memory(self, storage_manager_with_disk):
        """Test loading with pre-allocated memory objects."""
        sm = storage_manager_with_disk

        num_chunks = 2
        chunk_hashes = [4000 + i for i in range(num_chunks)]
        keys_layer_major = []
        expected_data = {}

        for layer_id in range(NUM_LAYERS):
            layer_keys = []
            for chunk_idx, chunk_hash in enumerate(chunk_hashes):
                key = create_test_key(layer_id, chunk_hash)
                layer_keys.append(key)
                data = torch.randn(
                    2, 1, CHUNK_SIZE, HIDDEN_DIM, dtype=torch.bfloat16
                )
                data.fill_(float(layer_id * 100 + chunk_idx + 1))
                _write_chunk_to_disk(sm, key, data)
                expected_data[(layer_id, chunk_idx)] = data
            keys_layer_major.append(layer_keys)

        _wait_disk_put_tasks(sm)

        # Pre-allocate memory objects in layer-major format
        shape = torch.Size([2, 1, CHUNK_SIZE, HIDDEN_DIM])
        pre_allocated = []
        for layer_id in range(NUM_LAYERS):
            layer_objs = sm.batched_allocate(
                shape=shape,
                dtype=torch.bfloat16,
                batch_size=num_chunks,
                fmt=MemoryFormat.KV_2LTD,
                eviction=True,
                busy_loop=True,
            )
            assert layer_objs is not None, (
                f"Failed to pre-allocate for layer {layer_id}"
            )
            pre_allocated.append(layer_objs)

        future = sm.batched_get_all_layers(
            keys_layer_major=keys_layer_major,
            location="LocalDiskBackend",
            pre_allocated_mem_objs=pre_allocated,
        )
        result = future.result(timeout=30)
        assert len(result) == NUM_LAYERS * num_chunks

        for layer_id in range(NUM_LAYERS):
            for chunk_idx in range(num_chunks):
                flat_idx = layer_id * num_chunks + chunk_idx
                expected = expected_data[(layer_id, chunk_idx)]
                assert torch.equal(result[flat_idx].tensor, expected), (
                    f"Data mismatch at layer={layer_id}, chunk={chunk_idx}"
                )

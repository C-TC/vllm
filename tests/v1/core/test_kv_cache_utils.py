# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import importlib
from collections.abc import Callable
from typing import Any

import pytest
import torch

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256, sha256_cbor
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    VICTIM_POLICY_PURE_LRU,
    VICTIM_POLICY_WIRES_THREE_POOL,
    BlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    estimate_max_model_len,
    generate_block_hash_extra_keys,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_max_concurrency_for_kv_cache_config,
    get_request_block_hasher,
    hash_block_tokens,
    init_none_hash,
    is_kv_cache_spec_uniform,
    make_block_hash_with_group_id,
    tensor_data,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.metrics.stats import CachingMetrics, PrefixCacheStats
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn(request):
    hash_fn: Callable
    if "hash_fn" in request.fixturenames:
        hash_fn = request.getfixturevalue("hash_fn")
    else:
        hash_fn = sha256
    init_none_hash(hash_fn)


def make_request(
    request_id: str,
    prompt_token_ids: list[int] | None,
    block_size: int = 3,
    hash_fn: Callable = hash,
    mm_positions: list[PlaceholderRange] | None = None,
    mm_hashes: list[str] | None = None,
    cache_salt: str | None = None,
    prompt_embeds: torch.Tensor | None = None,
):
    mm_features = []
    if mm_positions is not None:
        for j, position in enumerate(mm_positions):
            identifier = mm_hashes[j] if mm_hashes else f"hash_{j}"
            mm_feature = MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                mm_position=position,
                identifier=identifier,
                modality="image",
            )
            mm_features.append(mm_feature)

    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=mm_features if mm_features else None,
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=None,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(block_size, hash_fn),
        prompt_embeds=prompt_embeds,
    )


def new_kv_cache_spec(
    block_size=16,
    num_kv_heads=2,
    head_size=64,
    dtype=torch.float32,
    page_size_padded=None,
    sliding_window=None,
    attention_chunk_size=None,
):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_size_padded,
        sliding_window=sliding_window,
        attention_chunk_size=attention_chunk_size,
    )


def new_sliding_window_spec(
    block_size=16,
    num_kv_heads=2,
    head_size=64,
    dtype=torch.float32,
    page_size_padded=None,
    sliding_window=1,
):
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_size_padded,
        sliding_window=sliding_window,
    )


def new_chunked_local_attention_spec(
    block_size=16,
    num_kv_heads=2,
    head_size=64,
    dtype=torch.float32,
    page_size_padded=None,
    attention_chunk_size=4,
):
    return ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_size_padded,
        attention_chunk_size=attention_chunk_size,
    )


def new_mamba_spec(
    block_size=16,
    shapes=((2, 512), (3, 32, 32)),
    dtypes=(torch.float32, torch.float32),
    num_speculative_blocks=2,
    mamba_cache_mode="none",
    page_size_padded=None,
):
    return MambaSpec(
        block_size=block_size,
        shapes=shapes,
        dtypes=dtypes,
        page_size_padded=page_size_padded,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_none_hash(monkeypatch, hash_fn):
    import vllm.v1.core.kv_cache_utils

    # case 1: PYTHONHASHSEED is not set, use random
    with monkeypatch.context() as m:
        m.delenv("PYTHONHASHSEED", raising=False)
        reloaded_kv_cache_utils = importlib.reload(vllm.v1.core.kv_cache_utils)
        reloaded_kv_cache_utils.init_none_hash(hash_fn)
        assert reloaded_kv_cache_utils.NONE_HASH is not None
        assert isinstance(reloaded_kv_cache_utils.NONE_HASH, bytes)
        assert reloaded_kv_cache_utils.NONE_HASH != b""

    # case 2: PYTHONHASHSEED is set, use the seed and hash_fn
    with monkeypatch.context() as m:
        m.setenv("PYTHONHASHSEED", "python hash seed")
        reloaded_kv_cache_utils = importlib.reload(vllm.v1.core.kv_cache_utils)
        reloaded_kv_cache_utils.init_none_hash(hash_fn)
        assert reloaded_kv_cache_utils.NONE_HASH is not None
        assert isinstance(reloaded_kv_cache_utils.NONE_HASH, bytes)
        assert hash_fn("python hash seed") == reloaded_kv_cache_utils.NONE_HASH


def test_kv_cache_block():
    # Test KVCacheBlock initialization
    block = KVCacheBlock(block_id=0)
    assert block.block_id == 0
    assert block.ref_cnt == 0
    assert block.block_hash is None

    # Test reference count manipulation
    block.ref_cnt += 1
    assert block.ref_cnt == 1
    block.ref_cnt -= 1
    assert block.ref_cnt == 0

    # Test block hash setting and resetting
    block_hash = make_block_hash_with_group_id(BlockHash(b"abc"), 0)
    block.block_hash = block_hash
    assert block.block_hash == block_hash

    block.reset_hash()
    assert block.block_hash is None


def test_kv_cache_block_uses_slots():
    block = KVCacheBlock(block_id=0)

    # Slots eliminate per-instance __dict__, saving ~264 bytes per block.
    # At 100K+ blocks this avoids tens of MB of overhead and GC pressure.
    assert not hasattr(block, "__dict__")

    # Verify that slots actually prevent dynamic attribute assignment.
    with pytest.raises(AttributeError):
        block.unexpected_field = True


def test_free_kv_cache_block_queue_initialization():
    # Test with a single block
    block = KVCacheBlock(block_id=0)
    queue = FreeKVCacheBlockQueue([block])
    assert queue.num_free_blocks == 1
    assert queue.fake_free_list_head.next_free_block is block
    assert queue.fake_free_list_tail.prev_free_block is block


def test_free_kv_cache_block_queue_operations():
    # Create a list of KVCacheBlock objects
    blocks = [KVCacheBlock(block_id=i) for i in range(5)]

    # Create a FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue(blocks)

    # Check initial state
    assert queue.num_free_blocks == 5
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert queue.fake_free_list_tail.prev_free_block is blocks[4]

    # Pop the first block
    block1 = queue.popleft()
    assert block1 == blocks[0]
    assert queue.num_free_blocks == 4
    assert queue.fake_free_list_head.next_free_block is blocks[1]
    assert queue.fake_free_list_tail.prev_free_block is blocks[4]

    # Remove a block from the middle
    block_to_remove = blocks[2]
    queue.remove(block_to_remove)
    assert queue.num_free_blocks == 3
    assert blocks[1].next_free_block is blocks[3]
    assert blocks[3].prev_free_block is blocks[1]

    # Append a block back
    queue.append(block_to_remove)
    assert queue.num_free_blocks == 4
    assert queue.fake_free_list_tail.prev_free_block is block_to_remove
    assert block_to_remove.prev_free_block is blocks[4]
    assert block_to_remove.next_free_block is queue.fake_free_list_tail

    # Pop blocks until empty
    for _ in range(4):
        queue.popleft()
    assert queue.num_free_blocks == 0
    assert queue.fake_free_list_head.next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is queue.fake_free_list_head

    # Attempt to pop from an empty queue
    with pytest.raises(ValueError) as e:
        queue.popleft()
    assert str(e.value) == "No free blocks available"


def test_free_kv_cache_block_queue_append_n():
    # Create an empty FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue([])
    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
    # Append 0 block
    # fake_head->fake_tail
    queue.append_n([])
    assert queue.num_free_blocks == 0
    assert queue.fake_free_list_head.next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is queue.fake_free_list_head
    # Append 1 block
    # fake_head->b0->fake_tail
    queue.append_n(blocks[0:1])
    assert queue.num_free_blocks == 1
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert blocks[0].prev_free_block is queue.fake_free_list_head
    assert blocks[0].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[0]
    # Append 2 blocks
    # fake_head->b0->b4->b5->fake_tail
    queue.append_n(blocks[4:6])
    assert queue.num_free_blocks == 3
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert blocks[0].prev_free_block is queue.fake_free_list_head
    assert blocks[0].next_free_block is blocks[4]
    assert blocks[4].prev_free_block is blocks[0]
    assert blocks[4].next_free_block is blocks[5]
    assert blocks[5].prev_free_block is blocks[4]
    assert blocks[5].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[5]
    # Append 3 blocks
    # fake_head->b0->b4->b5->b1->b2->b3->fake_tail
    queue.append_n(blocks[1:4])
    assert queue.num_free_blocks == 6
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert blocks[0].prev_free_block is queue.fake_free_list_head
    assert blocks[0].next_free_block is blocks[4]
    assert blocks[4].prev_free_block is blocks[0]
    assert blocks[4].next_free_block is blocks[5]
    assert blocks[5].prev_free_block is blocks[4]
    assert blocks[5].next_free_block is blocks[1]
    assert blocks[1].prev_free_block is blocks[5]
    assert blocks[1].next_free_block is blocks[2]
    assert blocks[2].prev_free_block is blocks[1]
    assert blocks[2].next_free_block is blocks[3]
    assert blocks[3].prev_free_block is blocks[2]
    assert blocks[3].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[3]

    # Create an empty FreeKVCacheBlockQueue
    invalid_queue = FreeKVCacheBlockQueue([])
    # set prev_free_block to None and this will cause assertion in append_n
    invalid_queue.fake_free_list_tail.prev_free_block = None
    with pytest.raises(AssertionError):
        # Append 1 block
        # fake_head->fake_tail
        invalid_queue.append_n(blocks[0:1])
    assert invalid_queue.num_free_blocks == 0
    assert (
        invalid_queue.fake_free_list_head.next_free_block
        == invalid_queue.fake_free_list_tail
    )


def test_free_kv_cache_block_queue_popleft_n():
    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
    # Create an empty FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue(
        [blocks[1], blocks[3], blocks[5], blocks[4], blocks[0], blocks[2]]
    )
    assert queue.num_free_blocks == 6
    assert queue.fake_free_list_head.next_free_block is blocks[1]
    assert blocks[1].prev_free_block is queue.fake_free_list_head
    assert blocks[1].next_free_block is blocks[3]
    assert blocks[3].prev_free_block is blocks[1]
    assert blocks[3].next_free_block is blocks[5]
    assert blocks[5].prev_free_block is blocks[3]
    assert blocks[5].next_free_block is blocks[4]
    assert blocks[4].prev_free_block is blocks[5]
    assert blocks[4].next_free_block is blocks[0]
    assert blocks[0].prev_free_block is blocks[4]
    assert blocks[0].next_free_block is blocks[2]
    assert blocks[2].prev_free_block is blocks[0]
    assert blocks[2].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[2]

    # Pop 0 block
    # fake_head->b1->b3->b5->b4->b0->b2->fake_tail
    assert len(queue.popleft_n(0)) == 0
    assert queue.num_free_blocks == 6
    # Pop 1 block
    # fake_head->b3->b5->b4->b0->b2->fake_tail
    result_blocks = queue.popleft_n(1)
    assert queue.num_free_blocks == 5
    assert len(result_blocks) == 1
    assert result_blocks[0] is blocks[1]
    for block in result_blocks:
        assert block.prev_free_block is None
        assert block.next_free_block is None
    # Pop 2 blocks
    # fake_head->b4->b0->b2->fake_tail
    result_blocks = queue.popleft_n(2)
    assert len(result_blocks) == 2
    assert queue.num_free_blocks == 3
    assert result_blocks[0] is blocks[3]
    assert result_blocks[1] is blocks[5]
    for block in result_blocks:
        assert block.prev_free_block is None
        assert block.next_free_block is None
    # Pop 3 blocks
    # fake_head->fake_tail
    result_blocks = queue.popleft_n(3)
    assert len(result_blocks) == 3
    assert queue.num_free_blocks == 0
    assert result_blocks[0] is blocks[4]
    assert result_blocks[1] is blocks[0]
    assert result_blocks[2] is blocks[2]
    for block in result_blocks:
        assert block.prev_free_block is None
        assert block.next_free_block is None


def test_free_kv_cache_block_queue_lifecycle_priority_popleft_n():
    """WIRES Phase E2: popleft_n prefers lifecycle_hint='no' > 'may' > 'must'.

    Exercises 3-priority eviction order:
      - "no" blocks pulled first (eviction-safe)
      - "may" blocks (default) pulled next (LRU within class)
      - "must" blocks pulled last resort
    Within each class the existing front-to-back LRU order is preserved.
    """
    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
    # Layout (LRU order): b0=may b1=must b2=no b3=may b4=must b5=no
    blocks[0].lifecycle_hint = "may"
    blocks[1].lifecycle_hint = "must"
    blocks[2].lifecycle_hint = "no"
    blocks[3].lifecycle_hint = "may"
    blocks[4].lifecycle_hint = "must"
    blocks[5].lifecycle_hint = "no"
    queue = FreeKVCacheBlockQueue(blocks)
    assert queue.num_free_blocks == 6

    # Pop 1 → should pull b2 (first "no" in LRU order)
    out = queue.popleft_n(1)
    assert out == [blocks[2]]
    assert queue.num_free_blocks == 5

    # Pop 2 → b5 (last remaining "no") then b0 (first "may")
    out = queue.popleft_n(2)
    assert out == [blocks[5], blocks[0]]
    assert queue.num_free_blocks == 3

    # Pop 2 → b3 ("may") then b1 (first "must")
    out = queue.popleft_n(2)
    assert out == [blocks[3], blocks[1]]
    assert queue.num_free_blocks == 1

    # Pop last → b4 (only "must" left)
    out = queue.popleft_n(1)
    assert out == [blocks[4]]
    assert queue.num_free_blocks == 0


def test_free_kv_cache_block_queue_default_priority_is_legacy_fifo():
    """All-default ("may") blocks must reproduce legacy FIFO popleft_n order."""

    blocks = [KVCacheBlock(block_id=i) for i in range(4)]
    queue = FreeKVCacheBlockQueue(blocks)
    out = queue.popleft_n(4)
    assert out == blocks
    assert queue.num_free_blocks == 0


def test_free_kv_cache_block_queue_get_all_free_blocks():
    # Create a list of KVCacheBlock objects
    blocks = [KVCacheBlock(block_id=i) for i in range(5)]

    # Create a FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue(blocks)

    # Check all blocks are correctly retrieved
    assert queue.get_all_free_blocks() == blocks

    # Pop a block and check again
    queue.popleft()
    assert queue.get_all_free_blocks() == blocks[1:]

    # Remove a block and check again
    block_to_remove = blocks[2]
    queue.remove(block_to_remove)
    assert queue.get_all_free_blocks() == blocks[1:2] + blocks[3:]

    # Append a block back and check again
    queue.append(block_to_remove)
    assert queue.get_all_free_blocks() == blocks[1:2] + blocks[3:] + [block_to_remove]


# -- WIRES docs/v2/32 §2 — 3-pool victim policy + pure_lru baseline mode -----


def test_three_pool_no_pressure_lru_within_may_matches_legacy():
    """Under no eviction pressure, all-default-may blocks behave as plain LRU.

    Reproduces docs/v2/32 §2.2 property: "when `no` and `may` pools have
    enough blocks, `must` blocks are never touched. Under no pressure,
    behaviour is exactly LRU within may (matches legacy)."
    """
    blocks = [KVCacheBlock(block_id=i) for i in range(8)]
    queue = FreeKVCacheBlockQueue(blocks, mode=VICTIM_POLICY_WIRES_THREE_POOL)
    assert queue.num_free_blocks == 8
    assert queue.num_free_blocks_in_pool("may") == 8
    assert queue.num_free_blocks_in_pool("no") == 0
    assert queue.num_free_blocks_in_pool("must") == 0
    assert queue.popleft_n(3) == blocks[:3]
    assert queue.must_pool_evicted_count == 0


def test_three_pool_priority_no_then_may_then_must():
    """popleft_n consumes pools in priority order regardless of insert order."""
    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
    blocks[0].lifecycle_hint = "must"
    blocks[1].lifecycle_hint = "must"
    blocks[2].lifecycle_hint = "may"
    blocks[3].lifecycle_hint = "no"
    blocks[4].lifecycle_hint = "may"
    blocks[5].lifecycle_hint = "no"
    queue = FreeKVCacheBlockQueue(blocks, mode=VICTIM_POLICY_WIRES_THREE_POOL)
    assert queue.num_free_blocks_in_pool("no") == 2
    assert queue.num_free_blocks_in_pool("may") == 2
    assert queue.num_free_blocks_in_pool("must") == 2

    # Pop 3 → both no (in insertion order), then first may
    out = queue.popleft_n(3)
    assert out == [blocks[3], blocks[5], blocks[2]]
    assert queue.must_pool_evicted_count == 0

    # Pop 2 → remaining may, then first must (signal fires)
    out = queue.popleft_n(2)
    assert out == [blocks[4], blocks[0]]
    assert queue.must_pool_evicted_count == 1

    # Pop last → second must
    out = queue.popleft_n(1)
    assert out == [blocks[1]]
    assert queue.must_pool_evicted_count == 2


def test_three_pool_must_pool_evicted_signal_batches():
    """must_pool_evicted_count records total must-blocks taken in one popleft_n."""
    blocks = [KVCacheBlock(block_id=i) for i in range(4)]
    for b in blocks:
        b.lifecycle_hint = "must"
    queue = FreeKVCacheBlockQueue(blocks, mode=VICTIM_POLICY_WIRES_THREE_POOL)
    queue.popleft_n(3)
    # 3 must-blocks taken in one call → counter += 3
    assert queue.must_pool_evicted_count == 3


def test_three_pool_update_block_hint_moves_pools_when_in_queue():
    """Phase B: update_block_hint flips a free block's pool membership."""
    blocks = [KVCacheBlock(block_id=i) for i in range(3)]
    blocks[0].lifecycle_hint = "may"
    blocks[1].lifecycle_hint = "may"
    blocks[2].lifecycle_hint = "may"
    queue = FreeKVCacheBlockQueue(blocks, mode=VICTIM_POLICY_WIRES_THREE_POOL)

    # Promote block 1 to must — it should leave may pool, land in must tail.
    queue.update_block_hint(blocks[1], "must")
    assert blocks[1].lifecycle_hint == "must"
    assert queue.num_free_blocks_in_pool("may") == 2
    assert queue.num_free_blocks_in_pool("must") == 1

    # Pop 2 → both may (block 0, block 2); must (block 1) untouched.
    out = queue.popleft_n(2)
    assert out == [blocks[0], blocks[2]]
    assert queue.must_pool_evicted_count == 0

    # Pop the last → block 1 from must.
    out = queue.popleft_n(1)
    assert out == [blocks[1]]
    assert queue.must_pool_evicted_count == 1


def test_three_pool_update_block_hint_noop_for_in_use_block():
    """Hint update on a block not in the queue (in-use) just sets metadata."""
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    queue = FreeKVCacheBlockQueue(blocks, mode=VICTIM_POLICY_WIRES_THREE_POOL)
    # Pop block 0 → it's now "in-use" (out of queue).
    popped = queue.popleft_n(1)
    assert popped == [blocks[0]]
    # Update hint on the out-of-queue block — should not crash, just set field.
    queue.update_block_hint(blocks[0], "must")
    assert blocks[0].lifecycle_hint == "must"
    assert queue.num_free_blocks == 1  # only block 1 still in queue


def test_three_pool_update_block_hint_idempotent_on_same_value():
    """update_block_hint is no-op when current hint == new hint."""
    blocks = [KVCacheBlock(block_id=i) for i in range(1)]
    queue = FreeKVCacheBlockQueue(blocks, mode=VICTIM_POLICY_WIRES_THREE_POOL)
    queue.update_block_hint(blocks[0], "may")  # already may
    assert queue.num_free_blocks_in_pool("may") == 1


def test_three_pool_append_n_routes_per_block_hint():
    """Bulk append_n splits a mixed-hint batch across pools."""
    blocks = [KVCacheBlock(block_id=i) for i in range(4)]
    blocks[0].lifecycle_hint = "must"
    blocks[1].lifecycle_hint = "no"
    blocks[2].lifecycle_hint = "may"
    blocks[3].lifecycle_hint = "must"
    queue = FreeKVCacheBlockQueue([], mode=VICTIM_POLICY_WIRES_THREE_POOL)
    assert queue.num_free_blocks == 0
    queue.append_n(blocks)
    assert queue.num_free_blocks_in_pool("no") == 1
    assert queue.num_free_blocks_in_pool("may") == 1
    assert queue.num_free_blocks_in_pool("must") == 2


def test_three_pool_remove_after_external_hint_mutation_does_not_corrupt():
    """If lifecycle_hint is mutated externally (without going through
    update_block_hint), remove() still finds the block in its original pool.
    Pool membership is tracked by block_id, not by reading the hint at
    remove time — this is the docs/v2/32 §2.5 invariant.
    """
    blocks = [KVCacheBlock(block_id=i) for i in range(3)]
    queue = FreeKVCacheBlockQueue(blocks, mode=VICTIM_POLICY_WIRES_THREE_POOL)
    # External mutation (the kind segment_actions used to do pre-Phase B):
    blocks[1].lifecycle_hint = "must"  # block stays in may pool!
    # remove() must still find block 1 in may pool.
    queue.remove(blocks[1])
    assert queue.num_free_blocks == 2
    assert queue.num_free_blocks_in_pool("may") == 2


def test_pure_lru_mode_ignores_lifecycle_hint():
    """`pure_lru` mode reproduces stock vLLM single-FIFO behaviour."""
    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
    blocks[0].lifecycle_hint = "must"
    blocks[1].lifecycle_hint = "must"
    blocks[2].lifecycle_hint = "no"
    blocks[3].lifecycle_hint = "may"
    blocks[4].lifecycle_hint = "must"
    blocks[5].lifecycle_hint = "no"
    queue = FreeKVCacheBlockQueue(blocks, mode=VICTIM_POLICY_PURE_LRU)
    # All blocks land in may pool regardless of hint.
    assert queue.num_free_blocks_in_pool("may") == 6
    assert queue.num_free_blocks_in_pool("no") == 0
    assert queue.num_free_blocks_in_pool("must") == 0
    # popleft_n returns blocks in insertion order (legacy FIFO).
    assert queue.popleft_n(6) == blocks
    # must_pool_evicted_count never increments (no must pool concept).
    assert queue.must_pool_evicted_count == 0


def test_pure_lru_mode_update_block_hint_does_not_move_pools():
    """`pure_lru` treats lifecycle_hint as metadata only — never moves blocks."""
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    queue = FreeKVCacheBlockQueue(blocks, mode=VICTIM_POLICY_PURE_LRU)
    queue.update_block_hint(blocks[0], "must")
    # Field updated but block stays in may pool.
    assert blocks[0].lifecycle_hint == "must"
    assert queue.num_free_blocks_in_pool("may") == 2
    assert queue.num_free_blocks_in_pool("must") == 0
    # Eviction order is still pure FIFO (block 0 first).
    assert queue.popleft_n(1) == [blocks[0]]


def test_victim_policy_env_var_default_is_wires_three_pool(monkeypatch):
    """Absent env var → wires_three_pool is the default."""
    monkeypatch.delenv("WIRES_KVCACHE_VICTIM_POLICY", raising=False)
    queue = FreeKVCacheBlockQueue([])
    assert queue.mode == VICTIM_POLICY_WIRES_THREE_POOL


def test_victim_policy_env_var_pure_lru(monkeypatch):
    """Setting env to pure_lru opts into baseline mode."""
    monkeypatch.setenv("WIRES_KVCACHE_VICTIM_POLICY", VICTIM_POLICY_PURE_LRU)
    queue = FreeKVCacheBlockQueue([])
    assert queue.mode == VICTIM_POLICY_PURE_LRU


def test_victim_policy_env_var_unknown_raises(monkeypatch):
    monkeypatch.setenv("WIRES_KVCACHE_VICTIM_POLICY", "bogus_policy")
    with pytest.raises(ValueError, match="Unknown WIRES_KVCACHE_VICTIM_POLICY"):
        FreeKVCacheBlockQueue([])


# -- WIRES Phase C: lazy TTL sweep on must pool head -------------------------


def _make_must_blocks(n: int) -> list[KVCacheBlock]:
    blocks = []
    for i in range(n):
        b = KVCacheBlock(block_id=i)
        b.lifecycle_hint = "must"
        blocks.append(b)
    return blocks


def test_phase_c_ttl_disabled_zero_means_no_sweep(monkeypatch):
    """structured_ttl_ns=0 disables the TTL sweep (must blocks stay forever)."""
    blocks = _make_must_blocks(3)
    queue = FreeKVCacheBlockQueue(
        blocks, mode=VICTIM_POLICY_WIRES_THREE_POOL, structured_ttl_ns=0
    )
    # Even after a "long" wait, no demotion fires.
    for b in blocks:
        b.last_promoted_ns = 1  # ancient timestamp
    demoted = queue._sweep_ttl_must()
    assert demoted == 0
    assert queue.num_free_blocks_in_pool("must") == 3
    assert queue.ttl_demoted_count == 0


def test_phase_c_ttl_expired_blocks_demoted_to_may_lru_end():
    """Expired must blocks demote to may pool HEAD (LRU end), so they
    evict before may's existing entries on the next popleft_n."""
    must_blocks = _make_must_blocks(2)
    may_block = KVCacheBlock(block_id=42)  # default may
    queue = FreeKVCacheBlockQueue(
        must_blocks + [may_block],
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        structured_ttl_ns=1_000_000_000,  # 1s
    )
    assert queue.num_free_blocks_in_pool("must") == 2
    assert queue.num_free_blocks_in_pool("may") == 1

    # Make both must blocks "expired" by backdating their promotion timestamp.
    for b in must_blocks:
        b.last_promoted_ns = 1  # ancient

    # Trigger sweep via popleft_n.
    out = queue.popleft_n(1)
    assert queue.ttl_demoted_count == 2
    assert queue.num_free_blocks_in_pool("must") == 0
    assert queue.num_free_blocks_in_pool("may") == 2  # 2 demoted in front of original may
    # First popped should be the FIRST demoted block (LRU within may)
    # since prepend_many places them at may's HEAD in source order.
    assert out == [must_blocks[0]]
    for blk in out:
        assert blk.lifecycle_hint == "may"


def test_phase_c_ttl_sweep_stops_at_first_non_expired():
    """Sweep walks from head and stops at first non-expired block."""
    must_blocks = _make_must_blocks(4)
    queue = FreeKVCacheBlockQueue(
        must_blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        structured_ttl_ns=1_000_000_000,
    )
    # Expire the first 2; leave 3 + 4 fresh.
    must_blocks[0].last_promoted_ns = 1
    must_blocks[1].last_promoted_ns = 1
    # 2 and 3 keep the timestamp from constructor (now-ish) → fresh.

    demoted = queue._sweep_ttl_must()
    assert demoted == 2
    assert queue.num_free_blocks_in_pool("must") == 2
    assert queue.num_free_blocks_in_pool("may") == 2


def test_phase_c_ttl_promotion_restarts_clock():
    """update_block_hint(may->must) stamps last_promoted_ns to now."""
    blocks = [KVCacheBlock(block_id=i) for i in range(1)]  # default may
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        structured_ttl_ns=1_000_000_000,
    )
    blocks[0].last_promoted_ns = 1  # ancient (irrelevant — block is in may)
    queue.update_block_hint(blocks[0], "must")
    # Promotion to must just stamped a fresh timestamp; block won't expire.
    assert blocks[0].last_promoted_ns > 1_000_000_000  # > 1 sec since epoch ns
    demoted = queue._sweep_ttl_must()
    assert demoted == 0
    assert queue.num_free_blocks_in_pool("must") == 1


def test_phase_c_ttl_no_pressure_no_sweep_when_must_empty():
    """popleft_n with no must blocks doesn't waste sweep work."""
    blocks = [KVCacheBlock(block_id=i) for i in range(3)]  # default may
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        structured_ttl_ns=1_000_000_000,
    )
    # popleft a few; should never increment ttl_demoted_count.
    queue.popleft_n(2)
    assert queue.ttl_demoted_count == 0


def test_phase_c_ttl_pure_lru_mode_skips_sweep():
    """pure_lru mode treats must pool as nonexistent; no sweep work."""
    blocks = [KVCacheBlock(block_id=i) for i in range(3)]
    for b in blocks:
        b.lifecycle_hint = "must"  # ignored in pure_lru
        b.last_promoted_ns = 1  # ancient
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_PURE_LRU,
        structured_ttl_ns=1_000_000_000,
    )
    demoted = queue._sweep_ttl_must()
    assert demoted == 0


def test_phase_c_ttl_env_default_300s(monkeypatch):
    """Default structured TTL is 300s = 300 * 1e9 ns when env unset."""
    monkeypatch.delenv("WIRES_KVCACHE_STRUCTURED_TTL_MS", raising=False)
    queue = FreeKVCacheBlockQueue([])
    assert queue._structured_ttl_ns == 300 * 1_000_000_000


def test_phase_c_ttl_env_override(monkeypatch):
    monkeypatch.setenv("WIRES_KVCACHE_STRUCTURED_TTL_MS", "60000")
    queue = FreeKVCacheBlockQueue([])
    assert queue._structured_ttl_ns == 60 * 1_000_000_000


def test_phase_c_ttl_env_zero_disables_sweep(monkeypatch):
    monkeypatch.setenv("WIRES_KVCACHE_STRUCTURED_TTL_MS", "0")
    queue = FreeKVCacheBlockQueue([])
    assert queue._structured_ttl_ns == 0


def test_phase_c_ttl_env_invalid_raises(monkeypatch):
    monkeypatch.setenv("WIRES_KVCACHE_STRUCTURED_TTL_MS", "bogus")
    with pytest.raises(ValueError, match="must be an integer"):
        FreeKVCacheBlockQueue([])
    monkeypatch.setenv("WIRES_KVCACHE_STRUCTURED_TTL_MS", "-100")
    with pytest.raises(ValueError, match="must be >= 0"):
        FreeKVCacheBlockQueue([])


# -- WIRES Phase D: access-based promotion + per-class TTL -------------------


def test_phase_d_access_promote_threshold_1_promotes_after_first_hit():
    """Default threshold=1: any cache hit on a may block promotes it to must
    with source_class='unstructured'."""
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]  # default may
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        access_promotion_threshold=1,
    )
    assert queue.num_free_blocks_in_pool("may") == 2
    # First access promotes block[0].
    promoted = queue.try_access_promote(blocks[0])
    assert promoted is True
    assert blocks[0].lifecycle_hint == "must"
    assert blocks[0].source_class == "unstructured"
    assert queue.num_free_blocks_in_pool("must") == 1
    assert queue.num_free_blocks_in_pool("may") == 1
    assert queue.access_promoted_count == 1


def test_phase_d_access_promote_higher_threshold_requires_multiple_hits(
    monkeypatch,
):
    """threshold=3: takes 3 hits to promote."""
    blocks = [KVCacheBlock(block_id=0)]
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        access_promotion_threshold=3,
    )
    assert queue.try_access_promote(blocks[0]) is False  # count=1
    assert queue.try_access_promote(blocks[0]) is False  # count=2
    assert queue.try_access_promote(blocks[0]) is True   # count=3 → fire
    assert blocks[0].lifecycle_hint == "must"
    assert blocks[0]._access_count == 3


def test_phase_d_access_promote_threshold_0_disables(monkeypatch):
    """threshold=0 disables Phase D entirely."""
    blocks = [KVCacheBlock(block_id=0)]
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        access_promotion_threshold=0,
    )
    for _ in range(5):
        assert queue.try_access_promote(blocks[0]) is False
    assert blocks[0].lifecycle_hint == "may"
    assert blocks[0]._access_count == 0  # not even incremented when disabled


def test_phase_d_access_promote_skips_already_must():
    """Blocks already in must pool aren't re-promoted; access count stays 0
    (we exit before incrementing if hint != may)."""
    blocks = [KVCacheBlock(block_id=0)]
    blocks[0].lifecycle_hint = "must"
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        access_promotion_threshold=1,
    )
    # Increment access count, but no promote because hint is already must.
    promoted = queue.try_access_promote(blocks[0])
    assert promoted is False
    assert blocks[0].lifecycle_hint == "must"  # unchanged
    assert queue.access_promoted_count == 0


def test_phase_d_access_promote_pure_lru_mode_noop():
    """pure_lru mode never promotes via access (lifecycle_hint is metadata)."""
    blocks = [KVCacheBlock(block_id=0)]
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_PURE_LRU,
        access_promotion_threshold=1,
    )
    assert queue.try_access_promote(blocks[0]) is False
    assert blocks[0].lifecycle_hint == "may"
    assert queue.access_promoted_count == 0


def test_phase_d_per_class_ttl_structured_short_unstructured_long():
    """Phase C3: ttl_at_promotion is snapshotted PER block based on its
    source_class at promotion time. Structured block uses
    structured_ttl_ns; unstructured uses bootstrap (n_samples < 50)."""
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        structured_ttl_ns=1_000_000,                    # 1 ms
        unstructured_bootstrap_ttl_ns=10_000_000_000,   # 10 s bootstrap
    )
    # Promote block[0] structured, block[1] unstructured.
    queue.update_block_hint(blocks[0], "must", source_class="structured")
    queue.update_block_hint(blocks[1], "must", source_class="unstructured")
    assert blocks[0].source_class == "structured"
    assert blocks[1].source_class == "unstructured"
    assert blocks[0].ttl_at_promotion_ns == 1_000_000  # structured constant
    assert blocks[1].ttl_at_promotion_ns == 10_000_000_000  # bootstrap (no samples)
    assert queue.num_free_blocks_in_pool("must") == 2

    # Backdate blocks[0] so it's expired by its 1ms TTL.
    blocks[0].last_promoted_ns = 1  # ancient
    # blocks[1] keeps its fresh timestamp from update_block_hint.

    # Sweep should demote ONLY the structured block (per-block deadline
    # uses block's own ttl_at_promotion_ns).
    demoted = queue._sweep_ttl_must()
    assert demoted == 1
    assert queue.num_free_blocks_in_pool("must") == 1
    assert queue.num_free_blocks_in_pool("may") == 1
    must_remaining = queue._pools["must"].head.next_free_block
    assert must_remaining is blocks[1]


def test_phase_d_access_count_resets_on_block_eviction():
    """When a block's hash is reset (cache slot recycled for new content),
    _access_count zeroes so the new entry starts fresh."""
    block = KVCacheBlock(block_id=0)
    block._access_count = 5
    block.reset_hash()
    assert block._access_count == 0


def test_phase_d_threshold_env_default_1(monkeypatch):
    monkeypatch.delenv("WIRES_KVCACHE_ACCESS_PROMOTION_THRESHOLD", raising=False)
    queue = FreeKVCacheBlockQueue([])
    assert queue.access_promotion_threshold == 1


def test_phase_d_threshold_env_override(monkeypatch):
    monkeypatch.setenv("WIRES_KVCACHE_ACCESS_PROMOTION_THRESHOLD", "5")
    queue = FreeKVCacheBlockQueue([])
    assert queue.access_promotion_threshold == 5


def test_phase_c3_unstructured_bootstrap_ttl_env_default(monkeypatch):
    monkeypatch.delenv("WIRES_KVCACHE_UNSTRUCTURED_BOOTSTRAP_TTL_MS", raising=False)
    queue = FreeKVCacheBlockQueue([])
    # Bootstrap default = 60s (per docs/v2/32 §2.4.1 + Q answer).
    assert queue._unstructured_bootstrap_ttl_ns == 60 * 1_000_000_000


def test_phase_c3_unstructured_bootstrap_ttl_env_override(monkeypatch):
    monkeypatch.setenv("WIRES_KVCACHE_UNSTRUCTURED_BOOTSTRAP_TTL_MS", "120000")
    queue = FreeKVCacheBlockQueue([])
    assert queue._unstructured_bootstrap_ttl_ns == 120 * 1_000_000_000


def test_phase_d_update_block_hint_invalid_source_class_raises():
    blocks = [KVCacheBlock(block_id=0)]
    queue = FreeKVCacheBlockQueue(blocks, mode=VICTIM_POLICY_WIRES_THREE_POOL)
    with pytest.raises(ValueError, match="source_class"):
        queue.update_block_hint(blocks[0], "must", source_class="bogus")


# -- WIRES Phase C3: EMA estimator for unstructured prefix-reuse interval ----


def test_phase_c3_ema_bootstrap_default_used_until_threshold():
    """Until n_samples >= bootstrap_samples, ttl_at_promotion uses
    bootstrap default (60s by default; configurable)."""
    queue = FreeKVCacheBlockQueue(
        [],
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        unstructured_bootstrap_ttl_ns=12_345_678,  # arbitrary
        bootstrap_samples=3,
    )
    # Before any samples, snapshot returns bootstrap.
    assert queue._unstructured_ttl_at_promotion_ns() == 12_345_678
    # Feed 2 samples — still below threshold of 3.
    queue._feed_unstructured_sample(1_000_000_000)
    queue._feed_unstructured_sample(2_000_000_000)
    assert queue._n_samples_unstructured == 2
    assert queue._unstructured_ttl_at_promotion_ns() == 12_345_678  # still bootstrap
    # 3rd sample crosses threshold.
    queue._feed_unstructured_sample(1_500_000_000)
    assert queue._n_samples_unstructured == 3
    snap = queue._unstructured_ttl_at_promotion_ns()
    # Should be T̂_u + k * sigma_u (positive value).
    assert snap > 0
    assert snap != 12_345_678  # left bootstrap


def test_phase_c3_ema_estimator_converges_toward_input_mean():
    """EMA T̂_u tracks input mean over enough samples."""
    queue = FreeKVCacheBlockQueue(
        [],
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        bootstrap_samples=1,
        ema_alpha=0.5,  # fast convergence for test
    )
    # Feed constant 1000ns 50 times — T̂_u should converge to ~1000.
    for _ in range(50):
        queue._feed_unstructured_sample(1000)
    # Allow some slack for floating-point and EMA convergence.
    assert 950 <= queue.T_hat_u_ns <= 1050
    # Var should be near zero (constant input).
    assert queue.sigma_u_ns < 50


def test_phase_c3_ema_first_sample_seeds_estimator():
    """First sample seeds T̂_u directly (avoid slow climb from bootstrap)."""
    queue = FreeKVCacheBlockQueue(
        [],
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        unstructured_bootstrap_ttl_ns=60_000_000_000,  # 60s
    )
    queue._feed_unstructured_sample(5_000_000_000)  # 5s
    assert queue.T_hat_u_ns == 5_000_000_000
    assert queue._n_samples_unstructured == 1


def test_phase_c3_unstructured_promotion_snapshots_estimator_value():
    """update_block_hint(must, source=unstructured) snapshots T̂_u + k·ε at
    promotion time, NOT a constant — and then doesn't change retroactively."""
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        bootstrap_samples=1,
        ema_alpha=1.0,  # any sample fully replaces estimator
        unstructured_k=2.0,
    )
    # Feed an estimator value: T̂_u = 1000, Var = 0 → sigma = 0.
    queue._feed_unstructured_sample(1000)
    # Promote block[0] — snapshot should be T̂ + k*sigma = 1000 + 0 = 1000.
    queue.update_block_hint(blocks[0], "must", source_class="unstructured")
    assert blocks[0].ttl_at_promotion_ns == 1000
    # Now feed wildly different sample to shift T̂_u.
    queue._feed_unstructured_sample(1_000_000_000_000)
    # Already-promoted block's snapshot is unchanged.
    assert blocks[0].ttl_at_promotion_ns == 1000
    # New promotion picks up the new value.
    queue.update_block_hint(blocks[1], "must", source_class="unstructured")
    assert blocks[1].ttl_at_promotion_ns >= 1_000_000_000_000


# -- WIRES Phase C4: α-shrinkage capacity boundary ---------------------------


def test_phase_c4_alpha_default_one_when_no_pressure():
    """No capacity pressure → α stays at 1.0."""
    blocks = [KVCacheBlock(block_id=i) for i in range(10)]
    queue = FreeKVCacheBlockQueue(
        blocks, mode=VICTIM_POLICY_WIRES_THREE_POOL, total_capacity=100
    )
    # 10 blocks in may, 90 capacity unused → tons of headroom.
    alpha = queue._recompute_alpha()
    assert alpha == 1.0


def test_phase_c4_alpha_shrinks_under_must_pool_pressure():
    """When must pool size exceeds N_must_target = B - may - in_use,
    α drops below 1 to scale down deadlines."""
    blocks = [KVCacheBlock(block_id=i) for i in range(50)]
    for b in blocks:
        b.lifecycle_hint = "must"
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        total_capacity=10,  # 10 capacity but 50 must blocks!
        overflow_shrink_floor=0.1,
    )
    # in_use = 10 - 50 - 1 = -41 → clamped to 0.
    # N_must_target = 10 - 0 - 0 = 10. must_size = 50.
    # raw_alpha = 10/50 = 0.2 → above floor (0.1) → α = 0.2.
    alpha = queue._recompute_alpha()
    assert 0.1 < alpha < 1.0


def test_phase_c4_alpha_clamped_at_floor():
    """When N_must_target ≤ 0, α clamps to the floor."""
    blocks = [KVCacheBlock(block_id=i) for i in range(20)]
    for b in blocks:
        b.lifecycle_hint = "may"
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        total_capacity=20,
        overflow_shrink_floor=0.3,
    )
    # may=20, in_use = 20-20-1 = -1 clamped 0. N_must_target = 20-20-0 = 0
    # → clamps to floor.
    alpha = queue._recompute_alpha()
    assert alpha == 0.3


def test_phase_c4_sweep_uses_alpha_to_scale_deadline():
    """A vanilla block whose unscaled TTL would NOT have expired but
    α-scaled TTL HAS expired must be demoted.

    M16: α applies only to vanilla (unstructured-class) blocks; hinted
    (structured-class) blocks use the raw constant T_h backstop. This
    test exercises the vanilla branch by promoting blocks with
    ``source_class="unstructured"`` and forcing the bootstrap TTL via
    ``bootstrap_samples=1`` + ``ema_alpha=1.0`` so the snapshot is
    deterministic.

    To force α<1 we need must_pool_size > B - may_size - in_use, i.e.
    must_size > total_capacity - 0 (no may, no real in_use in this
    test). So set total_capacity LESS than the must pool size."""
    blocks = [KVCacheBlock(block_id=i) for i in range(5)]
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        structured_ttl_ns=10_000_000_000,  # 10s; ignored on vanilla path
        unstructured_bootstrap_ttl_ns=10_000_000_000,  # 10s vanilla TTL
        bootstrap_samples=1,
        ema_alpha=1.0,
        unstructured_k=0.0,  # snapshot = T̂_v + 0 = bootstrap value
        total_capacity=2,  # 5 in must vs B=2 → forces α=0.4
        overflow_shrink_floor=0.0001,
    )
    for blk in blocks:
        queue.update_block_hint(blk, "must", source_class="unstructured")
    # α = 2 / 5 = 0.4 → scaled deadline = promoted_at + 0.4 * 10s = +4s.
    # Backdate 5s to push past scaled deadline.
    for blk in blocks:
        blk.last_promoted_ns -= 5_000_000_000
    demoted = queue._sweep_ttl_must()
    assert demoted == 5
    # α should reflect the pressure.
    assert 0.3 < queue.alpha_shrinkage < 0.5


def test_m16_sweep_skips_alpha_on_hinted_blocks():
    """M16: hinted (structured-class) must blocks ignore α — their
    constant T_h backstop is never shortened by capacity pressure.

    Same pressure setup as test_phase_c4_sweep_uses_alpha_to_scale_deadline
    but with structured-class promotion: α = 0.4 would shorten T_h
    from 10s to 4s, but the M16 fix ignores α for structured blocks
    and uses the raw 10s. Backdating 5s should NOT demote them."""
    blocks = [KVCacheBlock(block_id=i) for i in range(5)]
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        structured_ttl_ns=10_000_000_000,  # 10s
        total_capacity=2,  # forces α ~ 0.4
        overflow_shrink_floor=0.0001,
    )
    for blk in blocks:
        queue.update_block_hint(blk, "must", source_class="structured")
    # Backdate 5s — under M16 hinted blocks survive (deadline = +10s
    # not +4s). Under the old uniform-α code these would be demoted.
    for blk in blocks:
        blk.last_promoted_ns -= 5_000_000_000
    demoted = queue._sweep_ttl_must()
    assert demoted == 0
    # α itself is still computed (telemetry), it just isn't applied.
    assert 0.3 < queue.alpha_shrinkage < 0.5
    # Backdating an additional 6s pushes past the raw 10s deadline →
    # all five demote even without α.
    for blk in blocks:
        blk.last_promoted_ns -= 6_000_000_000
    demoted = queue._sweep_ttl_must()
    assert demoted == 5


def test_phase_c4_alpha_pure_lru_mode_always_one():
    """pure_lru mode skips α-shrinkage entirely."""
    blocks = [KVCacheBlock(block_id=i) for i in range(5)]
    queue = FreeKVCacheBlockQueue(
        blocks, mode=VICTIM_POLICY_PURE_LRU, total_capacity=2
    )
    alpha = queue._recompute_alpha()
    assert alpha == 1.0


def test_phase_c3_env_knobs(monkeypatch):
    monkeypatch.setenv("WIRES_KVCACHE_UNSTRUCTURED_K", "2.5")
    monkeypatch.setenv("WIRES_KVCACHE_UNSTRUCTURED_EMA_ALPHA", "0.1")
    monkeypatch.setenv("WIRES_KVCACHE_UNSTRUCTURED_BOOTSTRAP_SAMPLES", "100")
    monkeypatch.setenv("WIRES_KVCACHE_OVERFLOW_SHRINK_FLOOR", "0.5")
    queue = FreeKVCacheBlockQueue([])
    # Env override pins both the legacy mirror attribute and the
    # adaptive-k current value to the env value (see M14).
    assert queue._unstructured_k == 2.5
    assert queue._unstructured_k_override == 2.5
    assert queue.unstructured_k_current == 2.5
    assert queue._ema_alpha == 0.1
    assert queue._bootstrap_samples == 100
    assert queue._overflow_shrink_floor == 0.5


# -- M14: p_h EMA + adaptive k -----------------------------------------------


def test_m14_p_h_starts_at_default_init():
    """No samples yet -> p_h sits at default init (0.5) and adaptive
    k clamps to the floor (1.0)."""
    queue = FreeKVCacheBlockQueue([], mode=VICTIM_POLICY_WIRES_THREE_POOL)
    assert queue.p_h_hat == 0.5
    assert queue.unstructured_k_current == 1.0
    assert queue.p_h_ema_sample_count == 0


def test_m14_p_h_ema_halflife_property():
    """After `halflife` samples of indicator=1 from p_h_init=0.5,
    EMA reaches halfway (~0.75). Asymptote: feeding a Bernoulli mix
    with mean 0.8 over many halflives drives p_h to ~0.8."""
    queue = FreeKVCacheBlockQueue(
        [],
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        ph_init=0.5,
        ph_ema_halflife=100,
    )
    for _ in range(100):
        queue._feed_p_h_sample(1)
    assert 0.74 < queue.p_h_hat < 0.76
    assert queue.p_h_ema_sample_count == 100

    queue2 = FreeKVCacheBlockQueue(
        [],
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        ph_init=0.5,
        ph_ema_halflife=100,
    )
    pattern = [1, 1, 1, 1, 0]  # 80% ones
    for i in range(2000):
        queue2._feed_p_h_sample(pattern[i % 5])
    assert 0.75 < queue2.p_h_hat < 0.85


def test_m14_k_derivation_reference_points():
    """`_derive_k_from_ph` matches the closed-form values from paper §3.6."""
    derive = FreeKVCacheBlockQueue._derive_k_from_ph
    assert derive(0.5) == pytest.approx(1.0)
    assert derive(0.8) == pytest.approx(2.0, rel=1e-6)
    assert 3.9 < derive(0.94) < 4.1
    assert 9.9 < derive(0.99) <= 10.0
    # p = 0.997 -> sqrt(332) ~ 18.2 -> clamped to 10.
    assert derive(0.997) == 10.0
    # p = 0.1 -> sqrt(0.111) ~ 0.33 -> clamped to floor 1.0.
    assert derive(0.1) == 1.0


def test_m14_k_clamp_boundaries():
    """k clamps at [1, 10] independently of p_h pathology."""
    derive = FreeKVCacheBlockQueue._derive_k_from_ph
    assert derive(0.001) == 1.0
    assert derive(1e-9) == 1.0
    assert derive(1.0 - 1e-9) == 10.0
    assert derive(0.999999) == 10.0


def test_m14_env_override_pins_k(monkeypatch):
    """`WIRES_KVCACHE_UNSTRUCTURED_K=7.0` pins k=7 regardless of EMA
    state, before and after samples arrive."""
    monkeypatch.setenv("WIRES_KVCACHE_UNSTRUCTURED_K", "7.0")
    queue = FreeKVCacheBlockQueue([], mode=VICTIM_POLICY_WIRES_THREE_POOL)
    assert queue.unstructured_k_current == 7.0
    for _ in range(500):
        queue._feed_p_h_sample(1)
    assert queue.unstructured_k_current == 7.0
    for _ in range(500):
        queue._feed_p_h_sample(0)
    assert queue.unstructured_k_current == 7.0
    assert queue._unstructured_k == 7.0


def test_m14_kwarg_override_pins_k():
    """Constructor kwarg `unstructured_k=` overrides env + adaptive."""
    queue = FreeKVCacheBlockQueue(
        [], mode=VICTIM_POLICY_WIRES_THREE_POOL, unstructured_k=3.5
    )
    assert queue.unstructured_k_current == 3.5
    for _ in range(200):
        queue._feed_p_h_sample(1)
    assert queue.unstructured_k_current == 3.5


def test_m14_adaptive_k_tracks_p_h():
    """Without an override, k tracks p_h: feed lots of ones -> p_h rises
    -> k climbs above the floor of 1."""
    queue = FreeKVCacheBlockQueue(
        [], mode=VICTIM_POLICY_WIRES_THREE_POOL, ph_ema_halflife=10
    )
    assert queue.unstructured_k_current == 1.0
    for _ in range(100):
        queue._feed_p_h_sample(1)
    assert queue.p_h_hat > 0.99
    assert queue.unstructured_k_current > 9.0


def test_m14_runner_driven_hinted_demote_samples_p_h():
    """`update_block_hint(may|no, structured)` on a must-pool block
    feeds p_h with `1[_must_hit_count > 0]`."""
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        ph_ema_halflife=1,
        ph_init=0.5,
    )
    queue.update_block_hint(blocks[0], "must", source_class="structured")
    queue.update_block_hint(blocks[1], "must", source_class="structured")
    blocks[0]._must_hit_count = 1
    queue.update_block_hint(blocks[0], "may", source_class="structured")
    queue.update_block_hint(blocks[1], "no", source_class="structured")
    assert queue.p_h_ema_sample_count == 2
    # halflife=1 -> alpha = 1 - 0.5 = 0.5. From 0.5 with samples [1, 0]:
    # p1 = 0.5*0.5 + 0.5*1 = 0.75 ; p2 = 0.5*0.75 + 0.5*0 = 0.375.
    assert queue.p_h_hat == pytest.approx(0.375, rel=1e-9)


def test_m14_t_h_backstop_demote_does_not_sample():
    """`_sweep_ttl_must` does NOT feed p_h - only runner-driven
    demotes count, per paper §3.6."""
    blocks = [KVCacheBlock(block_id=i) for i in range(3)]
    queue = FreeKVCacheBlockQueue(
        blocks,
        mode=VICTIM_POLICY_WIRES_THREE_POOL,
        structured_ttl_ns=1_000_000_000,
        ph_ema_halflife=10,
    )
    for blk in blocks:
        queue.update_block_hint(blk, "must", source_class="structured")
    initial_n = queue.p_h_ema_sample_count
    for blk in blocks:
        blk.last_promoted_ns -= 5_000_000_000
    demoted = queue._sweep_ttl_must()
    assert demoted == 3
    assert queue.p_h_ema_sample_count == initial_n


def test_m14_unstructured_demote_does_not_sample_p_h():
    """A vanilla (unstructured-class) demote must not feed p_h."""
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    queue = FreeKVCacheBlockQueue(
        blocks, mode=VICTIM_POLICY_WIRES_THREE_POOL
    )
    queue.update_block_hint(blocks[0], "must", source_class="unstructured")
    initial_n = queue.p_h_ema_sample_count
    queue.update_block_hint(blocks[0], "may", source_class="unstructured")
    assert queue.p_h_ema_sample_count == initial_n


def test_m14_ph_env_knobs(monkeypatch):
    """`WIRES_PH_INIT` and `WIRES_PH_EMA_HALFLIFE` resolve as expected."""
    monkeypatch.setenv("WIRES_PH_INIT", "0.94")
    monkeypatch.setenv("WIRES_PH_EMA_HALFLIFE", "50")
    queue = FreeKVCacheBlockQueue([], mode=VICTIM_POLICY_WIRES_THREE_POOL)
    assert queue._p_h_init == 0.94
    assert queue.p_h_hat == 0.94
    assert queue._p_h_ema_halflife == 50
    assert 3.9 < queue.unstructured_k_current < 4.1


def test_m14_ph_init_validation():
    """ph_init must be strictly in (0, 1)."""
    with pytest.raises(ValueError):
        FreeKVCacheBlockQueue([], ph_init=0.0)
    with pytest.raises(ValueError):
        FreeKVCacheBlockQueue([], ph_init=1.0)
    with pytest.raises(ValueError):
        FreeKVCacheBlockQueue([], ph_init=-0.1)


def test_m14_halflife_validation():
    """ph_ema_halflife must be >= 1."""
    with pytest.raises(ValueError):
        FreeKVCacheBlockQueue([], ph_ema_halflife=0)


def test_m14_must_hit_count_resets_on_promotion():
    """`_stamp_must_promotion` resets `_must_hit_count` so each fresh
    must-residency starts the indicator at 0."""
    block = KVCacheBlock(block_id=0)
    queue = FreeKVCacheBlockQueue(
        [block], mode=VICTIM_POLICY_WIRES_THREE_POOL
    )
    queue.update_block_hint(block, "must", source_class="structured")
    block._must_hit_count = 7
    queue.update_block_hint(block, "may", source_class="structured")
    queue.update_block_hint(block, "must", source_class="structured")
    assert block._must_hit_count == 0


def test_generate_block_hash_extra_keys():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(20)],
        mm_positions=[
            PlaceholderRange(offset=0, length=5),
            PlaceholderRange(offset=10, length=5),
        ],
        mm_hashes=["hash1", "hash2"],
    )

    # Test with no extra keys
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 0, 5, 0)
    assert extra_keys == (("hash1", 0),)
    assert next_mm_idx == 1

    # Test with partial overlap
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 3, 8, 0)
    assert extra_keys == (("hash1", -3),)
    assert next_mm_idx == 1

    # Test with no overlap
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 6, 10, 0)
    assert extra_keys is None
    assert next_mm_idx == 1

    # Test with multiple extra keys
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 0, 15, 0)
    assert extra_keys == (("hash1", 0), ("hash2", 10))
    assert next_mm_idx == 2


def test_generate_block_hash_extra_keys_no_mm_inputs():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=None,
        mm_hashes=None,
    )

    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 0, 5, 0)
    assert extra_keys is None
    assert next_mm_idx == 0


def test_generate_block_hash_extra_keys_cache_salt():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=None,
        mm_hashes=None,
        cache_salt="salt",
    )

    # salt is added for the first token
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 1, 0)
    assert extra_keys == ("salt",)
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 10, 0)
    assert extra_keys == ("salt",)

    # no salt added for other tokens
    extra_keys, _ = generate_block_hash_extra_keys(request, 1, 2, 0)
    assert extra_keys is None
    extra_keys, _ = generate_block_hash_extra_keys(request, 6, 10, 0)
    assert extra_keys is None

    # works together with other extra keys
    request_mm = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(20)],
        mm_positions=[
            PlaceholderRange(offset=0, length=5),
        ],
        mm_hashes=["hash1"],
        cache_salt="salt",
    )

    # Test with no extra keys
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request_mm, 0, 5, 0)
    assert extra_keys == (("hash1", 0), "salt")
    assert next_mm_idx == 1


def test_generate_block_hash_extra_keys_prompt_embeds():
    prompt_embeds = torch.randn(10, 3)
    request = make_request(
        request_id="0",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds,
    )

    # Test with prompt embeds for the first block
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 5, 0)
    expected_embeds = prompt_embeds[0:5]
    expected_hash = hashlib.sha256(kv_cache_utils.tensor_data(expected_embeds)).digest()
    assert extra_keys == (expected_hash,)

    # Test with prompt embeds for the second block
    extra_keys, _ = generate_block_hash_extra_keys(request, 5, 10, 0)
    expected_embeds = prompt_embeds[5:10]
    expected_hash = hashlib.sha256(kv_cache_utils.tensor_data(expected_embeds)).digest()
    assert extra_keys == (expected_hash,)


def test_generate_block_hash_extra_keys_prompt_embeds_cached(monkeypatch):
    prompt_embeds = torch.randn(10, 3)
    request = make_request(
        request_id="0",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds,
        block_size=20,
    )

    num_tensor_data_calls = 0
    original_tensor_data = kv_cache_utils.tensor_data

    def counting_tensor_data(tensor: torch.Tensor):
        nonlocal num_tensor_data_calls
        num_tensor_data_calls += 1
        return original_tensor_data(tensor)

    monkeypatch.setattr(kv_cache_utils, "tensor_data", counting_tensor_data)

    extra_keys_1, _ = generate_block_hash_extra_keys(request, 0, 5, 0)
    extra_keys_2, _ = generate_block_hash_extra_keys(request, 0, 5, 0)
    assert extra_keys_1 == extra_keys_2
    assert num_tensor_data_calls == 1


def test_generate_block_hash_extra_keys_different_prompt_embeds():
    prompt_embeds1 = torch.randn(10, 3)
    prompt_embeds2 = torch.randn(10, 3)
    request1 = make_request(
        request_id="0",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds1,
    )
    request2 = make_request(
        request_id="1",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds2,
    )

    extra_keys1, _ = generate_block_hash_extra_keys(request1, 0, 5, 0)
    extra_keys2, _ = generate_block_hash_extra_keys(request2, 0, 5, 0)
    assert extra_keys1 != extra_keys2


def test_generate_block_hash_extra_keys_lora():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
    )

    request.lora_request = LoRARequest(
        lora_name="test_lora_adapter", lora_int_id=1, lora_path="/path/to/lora"
    )

    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 3, 0)
    assert extra_keys == ("test_lora_adapter",)

    request.lora_request = None
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 3, 0)
    assert extra_keys is None


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_hash_block_tokens(hash_fn):
    parent_block_hash = BlockHash(b"123")
    curr_block_token_ids = (1, 2, 3)
    extra_keys = ("key1", "key2")

    block_hash = hash_block_tokens(
        hash_fn, parent_block_hash, curr_block_token_ids, extra_keys
    )
    expected = hash_fn((parent_block_hash, curr_block_token_ids, extra_keys))
    assert block_hash == expected


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_request_block_hasher(hash_fn):
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        block_size=3,
        hash_fn=hash_fn,
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash1", "hash2"],
    )

    block_hashes = request.block_hashes
    assert len(block_hashes) == 2
    assert block_hashes[0] == hash_fn(
        (kv_cache_utils.NONE_HASH, (0, 1, 2), (("hash1", 0),))
    )
    assert block_hashes[1] == hash_fn((block_hashes[0], (3, 4, 5), (("hash2", 0),)))


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_hash_tokens_different_mm_input(hash_fn):
    request1 = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        block_size=3,
        hash_fn=hash_fn,
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash1", "hash2"],
    )
    request2 = make_request(
        request_id="1",
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash3", "hash2"],
    )
    block_hashes1 = request1.block_hashes
    block_hashes2 = request2.block_hashes
    assert block_hashes1[0] != block_hashes2[0]
    assert block_hashes1[1] != block_hashes2[1]


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_hash_request_tokens_no_mm_inputs(hash_fn):
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        block_size=3,
        hash_fn=hash_fn,
        mm_positions=None,
        mm_hashes=None,
    )

    block_hashes = request.block_hashes

    assert len(block_hashes) == 2
    assert block_hashes[0] == hash_fn((kv_cache_utils.NONE_HASH, (0, 1, 2), None))
    assert block_hashes[1] == hash_fn((block_hashes[0], (3, 4, 5), None))


def _stats(requests: int, queries: int, hits: int) -> PrefixCacheStats:
    return PrefixCacheStats(requests=requests, queries=queries, hits=hits)


def test_metrics():
    """
    Test the prefix caching metrics.
    """
    metrics = CachingMetrics(max_recent_requests=5)
    assert metrics.hit_rate == 0.0

    metrics.observe(_stats(1, 20, 9))
    # 9 / 20 = 0.45
    assert metrics.hit_rate == 0.45

    metrics.observe(_stats(4, 80, 16))

    # 25 / 100 = 0.25
    assert metrics.hit_rate == 0.25

    metrics.observe(_stats(1, 10, 2))

    # Remove (20, 9) and add (10, 2): 18 / 90 = 0.2
    assert metrics.aggregated_requests == 5
    assert metrics.aggregated_query_total == 90
    assert metrics.aggregated_query_hit == 18
    assert metrics.hit_rate == 0.2

    metrics.reset()
    assert metrics.hit_rate == 0.0
    assert metrics.aggregated_requests == 0
    assert metrics.aggregated_query_total == 0
    assert metrics.aggregated_query_hit == 0
    assert not metrics.query_queue


def test_metrics_empty_stats():
    """
    Test the prefix caching metrics with empty stats.
    """
    metrics = CachingMetrics(max_recent_requests=5)
    metrics.observe(_stats(0, 0, 0))
    metrics.observe(_stats(1, 20, 9))
    metrics.observe(_stats(0, 0, 0))
    metrics.observe(_stats(4, 80, 16))
    metrics.observe(_stats(0, 0, 0))
    metrics.observe(_stats(1, 10, 2))
    # Remove (20, 9) and add (10, 2): 18 / 90 = 0.2
    assert metrics.aggregated_requests == 5
    assert metrics.aggregated_query_total == 90
    assert metrics.aggregated_query_hit == 18
    assert metrics.hit_rate == 0.2

    # Only the latest added stats preserved 10 / 20 = 0.5
    metrics.observe(_stats(11, 20, 10))
    assert metrics.aggregated_requests == 11
    assert metrics.aggregated_query_total == 20
    assert metrics.aggregated_query_hit == 10
    assert metrics.hit_rate == 0.5

    # Only the latest added stats preserved 30 / 40 = 0.75
    metrics.observe(_stats(22, 40, 30))
    assert metrics.aggregated_requests == 22
    assert metrics.aggregated_query_total == 40
    assert metrics.aggregated_query_hit == 30
    assert metrics.hit_rate == 0.75


def test_get_kv_cache_configs_multiple_workers():
    model_config = ModelConfig(max_model_len=16)
    vllm_config = VllmConfig(model_config=model_config)

    ref_kv_cache_spec = new_kv_cache_spec()
    same_kv_cache_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
    ]

    # Basic case. All things are the same.
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        same_kv_cache_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
    ]

    # Different available memory. This is the case for TP.
    # Use the smallest memory available.
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        same_kv_cache_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 20,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
    ]

    # Different KV cache specs. This is the case for PP.
    different_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
        },
        {
            "layer2": new_kv_cache_spec(),
            "layer3": new_kv_cache_spec(),
        },
    ]

    # Different workers have different layers.
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        different_layer_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1"], new_kv_cache_spec()),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer3"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer2", "layer3"], new_kv_cache_spec()),
            ],
        ),
    ]

    # Some layers are the same, some are different. This is the case for TP+PP
    tp_pp_kv_cache_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer3": new_kv_cache_spec(),
        },
        {
            "layer3": new_kv_cache_spec(),
        },
    ]

    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        tp_pp_kv_cache_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer3"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer3"], ref_kv_cache_spec),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer3"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer3"], ref_kv_cache_spec),
            ],
        ),
    ]

    # Different workers have different types of layers. This is the case for
    # hybrid models + PP.
    different_type_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer3": new_sliding_window_spec(),
            "layer4": new_sliding_window_spec(),
        },
    ]
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        different_type_layer_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
                KVCacheGroupSpec([], new_sliding_window_spec()),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer3"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer4"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec([], ref_kv_cache_spec),
                KVCacheGroupSpec(["layer3", "layer4"], new_sliding_window_spec()),
            ],
        ),
    ]

    # When divided into multiple KVCacheGroups, need to ensure the number of
    # layers per group is similar.
    different_type_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_sliding_window_spec(),
            "layer3": new_sliding_window_spec(),
        },
        {
            "layer4": new_kv_cache_spec(),
            "layer5": new_sliding_window_spec(),
            "layer6": new_sliding_window_spec(),
        },
    ]
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        different_type_layer_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 10,
            ref_kv_cache_spec.page_size_bytes * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    shared_by=["layer1", "layer2", "layer3"],
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1"], ref_kv_cache_spec),
                KVCacheGroupSpec(["layer2"], new_sliding_window_spec()),
                KVCacheGroupSpec(["layer3"], new_sliding_window_spec()),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    shared_by=["layer4", "layer5", "layer6"],
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer4"], ref_kv_cache_spec),
                KVCacheGroupSpec(["layer5"], new_sliding_window_spec()),
                KVCacheGroupSpec(["layer6"], new_sliding_window_spec()),
            ],
        ),
    ]

    # Have conflicting layers. Need to raise an error.
    conflicting_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
        },
        {
            "layer1": new_sliding_window_spec(),
        },
    ]
    with pytest.raises(AssertionError):
        get_kv_cache_configs(
            vllm_config,
            conflicting_layer_specs,
            [
                ref_kv_cache_spec.page_size_bytes * 2 * 10,
                ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ],
        )


@pytest.mark.parametrize(
    "asymmetric_memory",
    [False, True],
    ids=["symmetric", "asymmetric"],
)
def test_get_kv_cache_configs_pp_sharding(asymmetric_memory):
    model_config = ModelConfig(max_model_len=512)
    vllm_config = VllmConfig(model_config=model_config)

    ref_kv_cache_spec = new_kv_cache_spec()
    pp_kv_cache_specs = [
        {"layer1": ref_kv_cache_spec},
        {"layer2": ref_kv_cache_spec},
    ]

    expected_num_blocks = model_config.max_model_len // ref_kv_cache_spec.block_size + 1
    avail_memory = ref_kv_cache_spec.page_size_bytes * expected_num_blocks

    # With per-worker validation, each worker only needs memory for its own
    # layers. Worker 2 having more memory shouldn't affect worker 1's config.
    available_memory = (
        [avail_memory, avail_memory * 2] if asymmetric_memory else [avail_memory] * 2
    )

    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        pp_kv_cache_specs,
        available_memory,
    )

    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=expected_num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * expected_num_blocks,
                    shared_by=["layer1"],
                ),
            ],
            kv_cache_groups=[KVCacheGroupSpec(["layer1"], ref_kv_cache_spec)],
        ),
        KVCacheConfig(
            num_blocks=expected_num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * expected_num_blocks,
                    shared_by=["layer2"],
                ),
            ],
            kv_cache_groups=[KVCacheGroupSpec(["layer2"], ref_kv_cache_spec)],
        ),
    ]


def test_project_kv_cache_groups_to_worker():
    spec_a = new_kv_cache_spec()
    spec_b = new_kv_cache_spec(num_kv_heads=4)

    global_groups = [
        KVCacheGroupSpec(["layer1", "layer2", "layer3"], spec_a),
    ]
    worker_spec = {"layer1": spec_a, "layer2": spec_a}
    projected = kv_cache_utils._project_kv_cache_groups_to_worker(
        global_groups, worker_spec
    )
    assert len(projected) == 1
    assert projected[0].layer_names == ["layer1", "layer2"]
    assert projected[0].kv_cache_spec is spec_a

    projected = kv_cache_utils._project_kv_cache_groups_to_worker(
        global_groups, {"layer4": spec_a}
    )
    assert len(projected) == 1
    assert projected[0].layer_names == []
    assert projected[0].kv_cache_spec is spec_a

    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"layer1": spec_a, "layer2": spec_b, "layer3": spec_a},
    )
    global_groups_uniform = [
        KVCacheGroupSpec(["layer1", "layer2", "layer3"], uniform_spec),
    ]
    projected = kv_cache_utils._project_kv_cache_groups_to_worker(
        global_groups_uniform, {"layer1": spec_a, "layer3": spec_a}
    )
    assert len(projected) == 1
    assert projected[0].layer_names == ["layer1", "layer3"]
    proj_spec = projected[0].kv_cache_spec
    assert isinstance(proj_spec, UniformTypeKVCacheSpecs)
    assert set(proj_spec.kv_cache_specs.keys()) == {"layer1", "layer3"}


def test_merge_kv_cache_spec():
    same_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32),
        new_kv_cache_spec(num_kv_heads=32),
    ]
    merged_layer_spec = same_layer_specs[0].merge(same_layer_specs)
    assert merged_layer_spec.block_size == 16
    assert merged_layer_spec.num_kv_heads == 32
    assert merged_layer_spec.head_size == 64
    assert merged_layer_spec.dtype == torch.float32
    assert merged_layer_spec.sliding_window is None

    different_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32),
        new_kv_cache_spec(num_kv_heads=16),
    ]
    with pytest.raises(AssertionError):
        different_layer_specs[0].merge(different_layer_specs)

    full_spec = new_kv_cache_spec(num_kv_heads=32)
    different_type_layer_specs = [
        full_spec,
        SlidingWindowSpec(
            block_size=full_spec.block_size,
            num_kv_heads=full_spec.num_kv_heads,
            head_size=full_spec.head_size,
            dtype=full_spec.dtype,
            sliding_window=1,
        ),
    ]
    with pytest.raises(AssertionError):
        different_type_layer_specs[0].merge(different_type_layer_specs)
    with pytest.raises(AssertionError):
        different_type_layer_specs[1].merge(different_type_layer_specs)

    different_sliding_window_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=2),
    ]
    with pytest.raises(ValueError):
        different_sliding_window_layer_specs[0].merge(
            different_sliding_window_layer_specs
        )

    same_sliding_window_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
    ]
    merged_layer_spec = same_sliding_window_layer_specs[0].merge(
        same_sliding_window_layer_specs
    )
    assert merged_layer_spec.sliding_window == 1

    same_sliding_window_layer_spec_with_none = [
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=None),
    ]
    merged_layer_spec = same_sliding_window_layer_spec_with_none[0].merge(
        same_sliding_window_layer_spec_with_none
    )
    assert merged_layer_spec.sliding_window == 1


def test_is_kv_cache_spec_uniform():
    kv_cache_spec = {
        "layer_1": new_kv_cache_spec(num_kv_heads=32),
        "layer_2": new_kv_cache_spec(num_kv_heads=32),
    }
    assert is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_kv_cache_spec(num_kv_heads=32),
        "layer_2": new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
    }
    assert is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_kv_cache_spec(num_kv_heads=32),
        "layer_2": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
    }
    assert not is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
        "layer_2": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
    }
    assert is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
        "layer_2": new_sliding_window_spec(num_kv_heads=32, sliding_window=2),
    }
    assert not is_kv_cache_spec_uniform(kv_cache_spec)


@pytest.mark.parametrize(
    ("model_id", "max_model_len", "want_estimated_max_len"),
    [
        ("Qwen/Qwen1.5-7B", 16385, 16384),
        ("Qwen/Qwen1.5-7B", 16383, 16383),
    ],
)
def test_estimate_max_model_len(model_id, max_model_len, want_estimated_max_len):
    # Create a VllmConfig
    model_config = ModelConfig(
        model_id,
        runner="generate",
        dtype="float16",
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=32768,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )

    # Create KV cache specs
    kv_cache_spec = {}
    for i in range(32):
        layer_name = f"layer_{i}"
        kv_cache_spec[layer_name] = FullAttentionSpec(
            block_size=16,
            num_kv_heads=32,
            head_size=128,
            dtype=torch.float16,
        )
    # Estimate the maximum model length, 16384 model_len need 8GB
    estimated_max_len = estimate_max_model_len(
        vllm_config, kv_cache_spec, 8 * GiB_bytes
    )
    assert estimated_max_len == want_estimated_max_len


def test_get_max_concurrency_for_kv_cache_config():
    # Create a VllmConfig
    model_id = "Qwen/Qwen1.5-7B"
    max_model_len = 16384
    model_config = ModelConfig(
        model_id,
        runner="generate",
        dtype="float16",
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )

    full_attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=32,
        head_size=128,
        dtype=torch.float16,
    )

    sliding_window_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=32,
        head_size=128,
        dtype=torch.float16,
        sliding_window=1024,
    )

    kv_cache_config_full_attention = KVCacheConfig(
        num_blocks=int(1024 * 1.5),
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)], full_attention_spec),
        ],
    )
    max_concurrency_full_attention = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_full_attention
    )
    assert max_concurrency_full_attention == 1.5

    kv_cache_config_sliding_window = KVCacheConfig(
        num_blocks=129 * 3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)], sliding_window_spec),
        ],
    )
    max_concurrency_sliding_window = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_sliding_window
    )
    assert max_concurrency_sliding_window == 3

    kv_cache_config_hybrid_model = KVCacheConfig(
        num_blocks=(1024 + 129) * 3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)], full_attention_spec),
            KVCacheGroupSpec(
                [f"layer_{i}" for i in range(32, 64)], sliding_window_spec
            ),
        ],
    )
    max_concurrency_hybrid_model = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_hybrid_model
    )
    assert max_concurrency_hybrid_model == 3


def test_allocate_with_lookahead():
    """Verify that lookahead tokens correctly affect block allocation"""
    block_size = 4
    config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(size=100, shared_by=["layer1"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer1"], new_kv_cache_spec(block_size=block_size)),
        ],
    )

    request = make_request(
        request_id="0",
        prompt_token_ids=[],
        block_size=block_size,
        mm_positions=None,
        mm_hashes=None,
    )

    # Test case 1: Requires additional lookahead tokens
    kv_cache_manager = KVCacheManager(
        kv_cache_config=config, max_model_len=100, hash_block_size=block_size
    )
    blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=3,
        num_lookahead_tokens=2,  # Total required: 3+2=5 tokens
    )
    assert len(blocks.get_block_ids()[0]) == 2  # ceil(5/4)=2 blocks

    # Test case 2: With precomputed blocks
    kv_cache_manager = KVCacheManager(
        kv_cache_config=config, max_model_len=100, hash_block_size=block_size
    )
    # required_blocks = ceil((3 + 2) /4) = 2
    blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=3,
        num_lookahead_tokens=2,
    )
    assert len(blocks.get_block_ids()[0]) == 2

    # Test case 3: With precomputed blocks
    # required_blocks = ceil((3 + 4) / 4) = 2
    kv_cache_manager = KVCacheManager(
        kv_cache_config=config, max_model_len=100, hash_block_size=block_size
    )
    blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=3,
        num_lookahead_tokens=4,
    )
    assert len(blocks.get_block_ids()[0]) == 2


def test_get_kv_cache_config_one_worker():
    # pass max_model_len to pass check_enough_kv_cache_memory
    model_config = ModelConfig(max_model_len=16)
    vllm_config = VllmConfig(model_config=model_config)

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    # all layers are full attention -> single group
    kv_cache_specs_full = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }
    kv_cache_config_full = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_full], [mem_per_block_per_layer * 2 * 32]
    )[0]
    print(kv_cache_config_full)
    assert kv_cache_config_full == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_2"]),
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())],
    )

    # all layers are sliding window -> single group
    kv_cache_specs_sliding = {
        "layer_1": new_sliding_window_spec(),
        "layer_2": new_sliding_window_spec(),
    }
    kv_cache_config_sliding = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_sliding], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_sliding == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_2"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2"], new_sliding_window_spec())
        ],
    )

    # full + sliding, but disable_hybrid_kv_cache_manager
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_2"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1", "layer_2"], new_kv_cache_spec(sliding_window=1)
            ),
        ],
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False

    # full + sliding, with hybrid_kv_cache_manager
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 64, shared_by=["layer_1", "layer_2"]
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1"], new_kv_cache_spec()),
            KVCacheGroupSpec(["layer_2"], new_sliding_window_spec()),
        ],
    )

    # 2 full + 4 sliding, 2 layers per group
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
        "layer_3": new_sliding_window_spec(),
        "layer_4": new_sliding_window_spec(),
        "layer_5": new_sliding_window_spec(),
        "layer_6": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_1", "layer_3", "layer_4"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_2", "layer_5", "layer_6"],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec()),
            KVCacheGroupSpec(["layer_3", "layer_5"], new_sliding_window_spec()),
            KVCacheGroupSpec(["layer_4", "layer_6"], new_sliding_window_spec()),
        ],
    )

    # 3 full + 7 sliding, pad to 3 full + 9 sliding
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
        "layer_3": new_kv_cache_spec(),
        "layer_4": new_sliding_window_spec(),
        "layer_5": new_sliding_window_spec(),
        "layer_6": new_sliding_window_spec(),
        "layer_7": new_sliding_window_spec(),
        "layer_8": new_sliding_window_spec(),
        "layer_9": new_sliding_window_spec(),
        "layer_10": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 3 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_1", "layer_4", "layer_5", "layer_6"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_2", "layer_7", "layer_8", "layer_9"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32, shared_by=["layer_3", "layer_10"]
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2", "layer_3"], new_kv_cache_spec()),
            KVCacheGroupSpec(
                ["layer_4", "layer_7", "layer_10"], new_sliding_window_spec()
            ),
            KVCacheGroupSpec(["layer_5", "layer_8"], new_sliding_window_spec()),
            KVCacheGroupSpec(["layer_6", "layer_9"], new_sliding_window_spec()),
        ],
    )

    # 6 full + 5 sliding, pad to 6 full + 6 sliding. This is a typical case for gpt-oss
    # eagle where there is only one more full attention layer than sliding window layers
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
        "layer_3": new_kv_cache_spec(),
        "layer_4": new_kv_cache_spec(),
        "layer_5": new_kv_cache_spec(),
        "layer_6": new_kv_cache_spec(),
        "layer_7": new_sliding_window_spec(),
        "layer_8": new_sliding_window_spec(),
        "layer_9": new_sliding_window_spec(),
        "layer_10": new_sliding_window_spec(),
        "layer_11": new_sliding_window_spec(),
    }

    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 6 * 32]
    )[0]
    print(kv_cache_config_hybrid)
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_1", "layer_7"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_2", "layer_8"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_3", "layer_9"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_4", "layer_10"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_5", "layer_11"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_6"],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5", "layer_6"],
                new_kv_cache_spec(),
            ),
            KVCacheGroupSpec(
                ["layer_7", "layer_8", "layer_9", "layer_10", "layer_11"],
                new_sliding_window_spec(),
            ),
        ],
    )

    # different hidden size but same type, use UniformTypeKVCacheSpecs
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(head_size=128),
        "layer_2": new_kv_cache_spec(head_size=64),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 3 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32 * 2, shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_2"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1", "layer_2"],
                UniformTypeKVCacheSpecs(
                    block_size=16, kv_cache_specs=kv_cache_specs_hybrid
                ),
            )
        ],
    )

    # Different hidden size and different type, align by different block size
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(head_size=64),
        "layer_2": new_sliding_window_spec(head_size=32),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32, shared_by=["layer_1", "layer_2"]
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1"], new_kv_cache_spec(head_size=64)),
            KVCacheGroupSpec(
                ["layer_2"], new_sliding_window_spec(head_size=32, block_size=32)
            ),
        ],
    )

    # different hidden size that cannot be aligned by using different block size
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(head_size=64),
        "layer_2": new_sliding_window_spec(head_size=96),
    }

    with pytest.raises(NotImplementedError):
        get_kv_cache_configs(
            vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
        )[0]

    # Test num_gpu_blocks_override
    vllm_config.cache_config.num_gpu_blocks_override = 16
    kv_cache_config_override_blocks = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_full], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_override_blocks == KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 16, shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 16, shared_by=["layer_2"]),
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())],
    )


def test_get_kv_cache_configs_attention_free():
    kv_cache_specs: dict[str, KVCacheSpec] = {}
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    kv_cache_configs = get_kv_cache_configs(vllm_config, [kv_cache_specs], [0])
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[],
        )
    ]


def test_generate_uniform_type_kv_cache_specs():
    # All layers are full attention, can be merged
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(head_size=128),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec == UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs=kv_cache_specs
    )

    # Full attention + sliding window, cannot be merged
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_sliding_window_spec(sliding_window=1),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec is None

    # different order of full attention + sliding window, cannot be merged
    kv_cache_specs = {
        "layer_1": new_sliding_window_spec(sliding_window=1),
        "layer_2": new_kv_cache_spec(),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec is None

    # Same-size sliding window, can be merged
    kv_cache_specs = {
        "layer_1": new_sliding_window_spec(sliding_window=1),
        "layer_2": new_sliding_window_spec(sliding_window=1, head_size=128),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec == UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs=kv_cache_specs
    )

    # different block sizes, cannot be merged
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(block_size=16),
        "layer_2": new_kv_cache_spec(block_size=32),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec is None


def test_generate_scheduler_kv_cache_config():
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(head_size=128),
    }
    kv_cache_configs = [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer_1", "layer_2"],
                    UniformTypeKVCacheSpecs(
                        block_size=16, kv_cache_specs=kv_cache_specs
                    ),
                ),
            ],
        )
    ]
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    assert scheduler_kv_cache_config == KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())],
    )


def new_mla_spec(cache_dtype_str=None):
    return MLAAttentionSpec(
        block_size=16,
        num_kv_heads=16,
        head_size=64,
        dtype=torch.float32,
        cache_dtype_str=cache_dtype_str,
    )


def test_merge_mla_spec():
    kv_cache_specs = [
        new_mla_spec(),
        new_mla_spec(),
    ]
    mla_spec = kv_cache_specs[0].merge(kv_cache_specs)
    assert mla_spec == new_mla_spec()

    kv_cache_specs = [
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
    ]
    mla_spec = kv_cache_specs[0].merge(kv_cache_specs)
    assert mla_spec == new_mla_spec(cache_dtype_str="fp8_ds_mla")

    kv_cache_specs = [
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
        new_mla_spec(cache_dtype_str=None),
    ]
    with pytest.raises(AssertionError):
        kv_cache_specs[0].merge(kv_cache_specs)

    kv_cache_specs = [
        new_kv_cache_spec(),
        new_mla_spec(),
    ]
    with pytest.raises(AssertionError):
        kv_cache_specs[0].merge(kv_cache_specs)

    kv_cache_specs = [
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
        new_kv_cache_spec(),
    ]
    with pytest.raises(AssertionError):
        kv_cache_specs[0].merge(kv_cache_specs)


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_request_block_hasher_with_prompt_embeds(hash_fn: Callable[[Any], bytes]):
    block_size = 3
    num_tokens = 2 * block_size
    prompt_token_ids = [_ for _ in range(num_tokens)]
    hidden_size = 5
    prompt_embeds = torch.randn((num_tokens, hidden_size))

    request = make_request(
        request_id="0",
        prompt_token_ids=prompt_token_ids,
        block_size=block_size,
        hash_fn=hash_fn,
        prompt_embeds=prompt_embeds,
    )

    block_hashes = request.block_hashes
    assert len(block_hashes) == 2

    block1_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[:block_size])
    ).digest()
    expected_hash1 = hash_fn(
        (
            kv_cache_utils.NONE_HASH,
            tuple(prompt_token_ids[:block_size]),
            (block1_embeds_hash,),
        )
    )
    assert block_hashes[0] == expected_hash1

    block2_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[block_size:num_tokens])
    ).digest()
    expected_hash2 = hash_fn(
        (
            block_hashes[0],
            tuple(prompt_token_ids[block_size:num_tokens]),
            (block2_embeds_hash,),
        )
    )
    assert block_hashes[1] == expected_hash2


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_request_with_prompt_embeds_and_mm_inputs(hash_fn: Callable[[Any], bytes]):
    block_size = 3
    num_tokens = 2 * block_size
    prompt_token_ids = [_ for _ in range(num_tokens)]
    hidden_size = 5
    prompt_embeds = torch.randn((num_tokens, hidden_size))

    request = make_request(
        request_id="0",
        prompt_token_ids=prompt_token_ids,
        block_size=block_size,
        hash_fn=hash_fn,
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash1", "hash2"],
        prompt_embeds=prompt_embeds,
    )

    block_hashes = request.block_hashes
    assert len(block_hashes) == 2

    block1_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[:block_size])
    ).digest()
    expected_hash1 = hash_fn(
        (
            kv_cache_utils.NONE_HASH,
            tuple(prompt_token_ids[:block_size]),
            (("hash1", 0), block1_embeds_hash),
        )
    )
    assert block_hashes[0] == expected_hash1

    block2_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[block_size:num_tokens])
    ).digest()
    expected_hash2 = hash_fn(
        (
            block_hashes[0],
            tuple(prompt_token_ids[block_size:num_tokens]),
            (("hash2", 0), block2_embeds_hash),
        )
    )
    assert block_hashes[1] == expected_hash2


def test_auto_fit_max_model_len():
    """Test that max_model_len=-1 auto-fits to available GPU memory."""
    # Create config with original_max_model_len=-1 to trigger auto-fit
    model_config = ModelConfig(max_model_len=1024)
    # Simulate the user passing -1 by setting original_max_model_len
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2  # 16KB per block per layer
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }

    # With enough memory, max_model_len stays at the derived max
    large_available_memory = mem_per_block_per_layer * 2 * 1024  # plenty of memory
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [large_available_memory]
    )
    assert vllm_config.model_config.max_model_len == 1024

    # Reset for next test
    model_config = ModelConfig(max_model_len=1024)
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)

    # With limited memory, max_model_len should be reduced
    # Need memory for at least max_model_len tokens
    # 32 blocks worth of memory for 2 layers = can fit 32*16=512 tokens
    limited_memory = mem_per_block_per_layer * 2 * 32
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [limited_memory]
    )
    # Should be reduced to fit in memory
    assert vllm_config.model_config.max_model_len < 1024
    assert vllm_config.model_config.max_model_len > 0


def test_auto_fit_max_model_len_with_hybrid():
    """Test that auto-fit works with hybrid KV cache specs."""
    # Create config with original_max_model_len=-1 to trigger auto-fit
    model_config = ModelConfig(max_model_len=8192)
    # Simulate the user passing -1 by setting original_max_model_len
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2  # 16KB per block per layer
    gamma = 2
    kv_cache_specs = {
        "layer_1": new_mamba_spec(num_speculative_blocks=gamma),
        "layer_2": new_kv_cache_spec(),
    }

    available_memory = mem_per_block_per_layer * (1024 // 16 + 1 + gamma)
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [available_memory]
    )
    assert vllm_config.model_config.max_model_len == 1024


def test_auto_fit_max_model_len_not_triggered():
    """Test that auto-fit is not triggered when original_max_model_len is not -1."""
    model_config = ModelConfig(max_model_len=16)
    # original_max_model_len should be None by default, not -1
    vllm_config = VllmConfig(model_config=model_config)

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }

    # This should work normally without auto-fit
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [mem_per_block_per_layer * 2 * 32]
    )
    assert vllm_config.model_config.max_model_len == 16


def test_unify_hybrid_kv_cache_specs():
    # 1. has_full_attention and has_sliding_window
    before_spec_1 = new_kv_cache_spec()
    before_spec_2 = new_sliding_window_spec(
        page_size_padded=32 * 1024, sliding_window=1024
    )
    kv_cache_spec = {
        "layer_1": before_spec_1,
        "layer_2": before_spec_2,
    }
    kv_cache_utils.unify_hybrid_kv_cache_specs(kv_cache_spec)
    expected_spec_1 = new_kv_cache_spec()
    expected_spec_2 = new_kv_cache_spec(page_size_padded=32 * 1024, sliding_window=1024)
    assert kv_cache_spec["layer_1"] == expected_spec_1
    assert kv_cache_spec["layer_2"] == expected_spec_2

    # 2. has_full_attention and has_chunked_local_attention
    before_spec_1 = new_kv_cache_spec()
    before_spec_2 = new_chunked_local_attention_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )
    kv_cache_spec = {
        "layer_1": before_spec_1,
        "layer_2": before_spec_2,
    }
    kv_cache_utils.unify_hybrid_kv_cache_specs(kv_cache_spec)
    expected_spec_1 = new_kv_cache_spec()
    expected_spec_2 = new_kv_cache_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )

    assert kv_cache_spec["layer_1"] == expected_spec_1
    assert kv_cache_spec["layer_2"] == expected_spec_2

    # 3. has_full_attention, has_sliding_window and has_chunked_local_attention
    before_spec_1 = new_kv_cache_spec()
    before_spec_2 = new_sliding_window_spec(
        page_size_padded=32 * 1024, sliding_window=1024
    )
    before_spec_3 = new_chunked_local_attention_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )
    kv_cache_spec = {
        "layer_1": before_spec_1,
        "layer_2": before_spec_2,
        "layer_3": before_spec_3,
    }
    kv_cache_utils.unify_hybrid_kv_cache_specs(kv_cache_spec)
    expected_spec_1 = new_kv_cache_spec()
    expected_spec_2 = new_kv_cache_spec(page_size_padded=32 * 1024, sliding_window=1024)
    expected_spec_3 = new_kv_cache_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )
    assert kv_cache_spec["layer_1"] == expected_spec_1
    assert kv_cache_spec["layer_2"] == expected_spec_2
    assert kv_cache_spec["layer_3"] == expected_spec_3

    # 4. No FullAttentionSpec, should not convert
    kv_cache_spec = {
        "layer_1": new_sliding_window_spec(sliding_window=1024),
        "layer_2": new_chunked_local_attention_spec(attention_chunk_size=512),
    }

    with pytest.raises(ValueError):
        kv_cache_utils.unify_hybrid_kv_cache_specs(kv_cache_spec)

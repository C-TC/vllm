# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-Cache Utilities."""

import copy
import hashlib
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, NewType, TypeAlias, overload

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.hashing import sha256_cbor, xxhash_cbor
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import format_gib
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request
from vllm.v1.utils import tensor_data
from vllm.v1.wires_engine_telemetry import (
    EVICT_REASON_LRU_PRESSURE,
    EVICT_REASON_MUST_PRESSURE_BACKSTOP,
    EVICT_REASON_TTL_EXPIRED,
)
from vllm.v1.wires_engine_telemetry import (
    emit_eviction as _wires_emit_eviction,
)
from vllm.v1.wires_telemetry import is_enabled as _wires_telemetry_enabled

# BlockHash represents the hash of a single KV-cache block used for
# prefix caching.  Treating it as a distinct type from `bytes` helps
# catch accidental misuse when passing around raw byte strings.
BlockHash = NewType("BlockHash", bytes)

# `BlockHashWithGroupId` combines a `BlockHash` with its KV cache group ID.
# It is represented as raw bytes for compactness and efficiency. The helper
# functions below pack/unpack the `BlockHash` and group id into/from the key.
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# ExternalBlockHash is used for reproducible prefix-cache block hashing.
# It's a union of `bytes` and `int` to keep backward compatibility
# after we default block hashing to use sha256 bytes.
ExternalBlockHash: TypeAlias = bytes | int


def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    """Pack a `BlockHash` and group id into a `BlockHashWithGroupId`.

    The group id is encoded using 4 bytes in big-endian order and appended to
    the block hash bytes.  This representation avoids creating tuples while
    still allowing us to recover both components when needed.
    """
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    """Extract the `BlockHash` from a `BlockHashWithGroupId`."""
    return BlockHash(key[:-4])


def get_group_id(key: BlockHashWithGroupId) -> int:
    """Extract the group id from a `BlockHashWithGroupId`."""
    return int.from_bytes(key[-4:], "big", signed=False)


def maybe_convert_block_hash(hash_bytes: BlockHash) -> ExternalBlockHash:
    if not envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES:
        return hash_bytes
    return int.from_bytes(hash_bytes, byteorder="big") & ((1 << 64) - 1)


logger = init_logger(__name__)

# The hash seed for the first block of any prefix block sequence.
#
# We use a random value to avoid hash collisions or PYTHONHASHSEED environment
# variable if set such that processes can share the seed if needed. This aligns
# with the behavior of Python's hash() function, which also uses a random seed
# if PYTHONHASHSEED is not set.
#
# The function `init_none_hash` initializes this variable globally.
NONE_HASH: BlockHash
_CBOR_HASH_FUNCTIONS = frozenset({sha256_cbor, xxhash_cbor})


def init_none_hash(hash_fn: Callable[[Any], bytes]):
    global NONE_HASH

    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None and hash_fn in _CBOR_HASH_FUNCTIONS:
        logger.warning(
            "PYTHONHASHSEED is not set. This will lead to non-reproducible "
            "block-hashes when using CBOR-based hash functions such as "
            "sha256_cbor or xxhash_cbor. Consider setting PYTHONHASHSEED to a "
            "fixed value for reproducibility."
        )

    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(32))
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed))


@dataclass(slots=True)
class KVCacheBlock:
    """KV-cache block metadata."""

    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    _block_hash: BlockHashWithGroupId | None = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # Whether the block is a null block that should never be cached.
    is_null: bool = False

    # WIRES Phase E2: per-block lifecycle hint driving 3-priority
    # eviction within the free queue. See docs/v2/31 §E2.
    #   "no"   — caller has signaled the block is safe to evict ASAP.
    #            Eviction picks these first; effectively LRU within "no".
    #   "may"  — DEFAULT. Standard LRU-aware behavior; evicted after "no"
    #            blocks are exhausted.
    #   "must" — caller has signaled the block is still live; engine
    #            avoids evicting these unless the pool is fully starved.
    # Set by:
    #   - request submission (per-request hint propagated to allocated blocks
    #     via vllm_xargs["lifecycle_hint"]; see kv_cache_manager.allocate_slots)
    #   - WIRES `/v1/coopt/segment_lifecycle_update` endpoint (deferred)
    #   - Monitor callback after CFG events (loop exit -> "no")
    lifecycle_hint: str = "may"

    # WIRES Phase C: monotonic ns timestamp recorded when this block
    # was last promoted into the must pool. 0 = never promoted (block
    # has always lived in no/may, or was just allocated). Used by the
    # lazy TTL sweep at the head of popleft_n. Phase C3+ uses
    # ``last_promoted_ns + ttl_at_promotion_ns < now`` as the demotion
    # deadline (M16 cleanup: α-shrinkage removed; capacity pressure is
    # handled via the `must_pool_evicted` backstop only). See
    # docs/v2/32 §2.4.
    last_promoted_ns: int = 0

    # WIRES Phase C3 (per docs/v2/32 §2.4.1 unstructured TTL design):
    # snapshot of the TTL value (in ns) computed at promotion time.
    # For structured-source promotions: equal to
    # WIRES_KVCACHE_STRUCTURED_TTL_MS converted to ns.
    # For unstructured-source promotions: equal to
    #   T̂_u + k * sqrt(Var_u)   (if n_samples_unstructured >= 50)
    #   bootstrap default 60s   (otherwise)
    # Snapshot semantics: subsequent EMA drift does NOT modify
    # already-promoted blocks. Avoids retroactive churn under
    # workload shifts.
    ttl_at_promotion_ns: int = 0

    # WIRES Phase C3: monotonic ns timestamp of the block's most
    # recent access (cache hit OR initial fill). Bumped in
    # ``BlockPool.touch()``. Used by the unstructured EMA estimator
    # to compute prefix-reuse intervals: when a must-pool
    # unstructured block is hit, ``interval = now - last_access_ns``
    # is fed into T̂_u / Var_u BEFORE last_access_ns is bumped to
    # ``now``. Demotion (rollback or TTL) does NOT clear this field.
    last_access_ns: int = 0

    # WIRES Phase D / M20: which promotion path put this block into the
    # must pool. Drives per-class TTL during the lazy sweep.
    #   "structured"   — explicit CFG-driven promotion via
    #                    update_segment_lifecycle_hint (driver/monitor
    #                    knows the segment will recur). TTL =
    #                    _structured_ttl_ns (default 5min, env
    #                    WIRES_KVCACHE_STRUCTURED_TTL_MS).
    #   "unstructured" — access-based promotion via block_pool.touch()
    #                    once _access_count crosses the threshold (the
    #                    natural LRU path, paper §2.3). This is the
    #                    DEFAULT, most blocks reach must this way.
    #                    TTL = _unstructured_ttl_at_promotion_ns()
    #                    (EMA-derived snapshot, bootstrap default 60s).
    #   "speculative"  — proactive may->must promotion driven by a
    #                    runner hint that anticipates future reuse
    #                    (paper §3.2 future-work extension; doc 32 §4
    #                    OQ6 lists the candidate patterns). Carries a
    #                    much shorter TTL backstop (_speculative_ttl_ns,
    #                    default 30s, env
    #                    WIRES_KVCACHE_SPECULATIVE_TTL_MS) so a wrong
    #                    speculation only wastes the must slot for a
    #                    bounded window. Class is TEMPORARY: on the
    #                    first cache hit, BlockPool.touch() upgrades
    #                    the block to "unstructured" and re-stamps it
    #                    with the EMA-based TTL (M20: speculation
    #                    confirmed -> behave like a normal access-
    #                    promoted block).
    # Refreshed each time the block is promoted; demotion does not
    # clear it (so a block re-promoted via access continues to use
    # the unstructured TTL).
    source_class: str = "unstructured"

    # WIRES Phase D: per-block hit count, used by access-based
    # promotion. Incremented in block_pool.touch() (one cache hit on
    # this block = one increment). Compared to
    # WIRES_KVCACHE_ACCESS_PROMOTION_THRESHOLD (default 1) to decide
    # whether to promote may -> must. Reset to 0 in reset_hash() so
    # a slot recycled for a different cache entry starts fresh.
    _access_count: int = 0

    # M14 (paper §3.6): per-block tally of cache hits that landed
    # while the block was residing in the must pool, scoped to the
    # block's CURRENT must-residency. Reset to 0 at every promotion
    # into must (`_stamp_must_promotion`) so each must-residency
    # starts fresh; bumped in `block_pool.touch()` whenever a hit
    # arrives on a block currently in the must pool.
    #
    # The runner-driven hinted-demote sample point reads this field
    # to derive the indicator `1[block hit at least once in must]`
    # that feeds `p̂_h`. Demotes induced by the constant T_h backstop
    # (`_sweep_ttl_must`) do NOT sample, so this field is only
    # consulted on the runner-driven path.
    _must_hit_count: int = 0

    # WIRES Phase E4 / M18 (paper/CODE_MISMATCH_NOTES.md): per-block
    # segment ids used by the block-pool touch / eviction hooks to
    # attribute cache events back to the SegmentRegistry. Empty tuple
    # ``()`` for non-WIRES blocks (the vast majority); appended to by
    # ``segment_actions.tag_blocks_with_segment_id`` when a
    # segment_prepare prefill returns. The block_pool hooks
    # short-circuit when this is empty, so non-WIRES request paths
    # pay zero telemetry cost.
    #
    # M18: this used to be a single ``str | None``. A physical KV
    # block (16 tokens by default) may overlap multiple segments
    # when segment boundaries don't align with block boundaries
    # (doc 32 §2.3 "min wins"); the previous single-valued schema
    # silently lost the earlier segment's reverse-index entry on
    # the second tag. The new tuple keeps the full set so
    # ``update_block_hints`` can recompute the min-priority hint
    # across all overlapping segments and the reverse map stays
    # symmetric. ``_segment_ids[0]`` is the "primary" tag (the
    # first segment to claim the block); a backward-compat
    # ``_segment_id`` property returns it for read-only consumers
    # that haven't migrated yet (delete in a follow-up after all
    # consumers migrate).
    _segment_ids: tuple[str, ...] = ()

    # T-45.7 (docs/v2/45 §3.4): writer attribution. Each time this
    # block is filled (chat completion prefill, segment_prepare
    # prefill, segment_refresh prefill, prefix_prepare prefill) the
    # caller stamps these three fields via
    # ``wires_engine_telemetry.stamp_block_writer``. The E4 cache_source
    # decision walks them at allocate time:
    #
    #   _last_writer_kind:   one of WRITER_KIND_* enum values
    #                        (chat_completion / segment_prepare /
    #                        segment_refresh / prefix_prepare); None
    #                        means "block has never been filled"
    #                        => cold_prefill.
    #   _last_writer_ts:     wall-clock epoch of the fill. Used with
    #                        WIRES_TELEMETRY_PRE_PREPARED_WINDOW_S to
    #                        decide pre_prepared vs peer_cached.
    #   _last_writer_request_id:
    #                        for chat_completion, the ``vllm_request_id``
    #                        (chatcmpl-...) of the request that wrote
    #                        the block. Used to distinguish self_hit
    #                        from peer_cached. For action-driven fills
    #                        carries the action endpoint string
    #                        (segment_prepare / etc.) so it's always
    #                        deterministic.
    #
    # All three default to None / 0.0 so non-WIRES blocks (e.g.
    # baseline lanes' fills) pay no construction cost beyond the slot.
    _last_writer_kind: str | None = None
    _last_writer_ts: float = 0.0
    _last_writer_request_id: str | None = None

    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHashWithGroupId):
        assert self.block_hash is None, (
            "The block already has a hash. This should not happen."
        )
        self._block_hash = block_hash

    @property
    def _segment_id(self) -> str | None:
        """M18 backward-compat shim: return the primary segment id.

        The schema migrated from ``_segment_id: str | None`` to
        ``_segment_ids: tuple[str, ...]`` so a single physical KV
        block can carry multiple overlapping segment tags (doc 32
        §2.3 "min wins"). Read-only consumers that haven't migrated
        yet can still call ``block._segment_id`` and get the first /
        "primary" tag; callers that need the FULL set must read
        ``block._segment_ids`` directly. Delete this property in a
        follow-up once all consumers migrate.
        """

        return self._segment_ids[0] if self._segment_ids else None

    def reset_hash(self):
        """Reset the block hash when the block is evicted.

        Also resets WIRES Phase D/C3 per-block state, since the slot
        is being recycled for new content; the previous content's
        hit history / promotion / TTL snapshot are no longer
        meaningful.
        """
        self._block_hash = None
        self._access_count = 0
        # M14: clear must-residency hit tally (the previous content's
        # in-must hits are no longer meaningful for the new content).
        self._must_hit_count = 0
        # M18: a recycled slot must lose its prior segment tags so the
        # new content's tag_blocks_with_segment_id calls don't union
        # against stale segment ids. The reverse-index in
        # SegmentRegistry self-heals stale entries via the
        # ``segment_id not in block._segment_ids`` skip; this clear
        # keeps memory bounded across slot reuse.
        self._segment_ids = ()
        # Phase C3: clear promotion snapshot, the next promotion
        # starts a fresh TTL window for the new content.
        self.last_promoted_ns = 0
        self.ttl_at_promotion_ns = 0
        # Note: last_access_ns is NOT reset here; the field tracks
        # the slot's latest physical access regardless of content,
        # and zero-after-allocation behaviour is preserved by the
        # default field initialiser only on construction.

    def __repr__(self) -> str:
        # Use block_id instead of KVCacheBlock object to avoid calling __repr__
        # on KVCacheBlock object recursively.
        prev_block_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"KVCacheBlock(block_id={self.block_id}, "
            f"ref_cnt={self.ref_cnt}, "
            f"_block_hash={self._block_hash!r}, "
            f"prev_free_block={prev_block_id}, "
            f"next_free_block={next_block_id})"
        )


class _PoolList:
    """Internal: one pool's doubly-linked list (fake-head/tail sentinel
    pattern). Not part of FreeKVCacheBlockQueue's public API.

    Operations are O(1); does not allocate Python objects per call,
    matching the performance contract of the legacy single-queue impl.
    """

    __slots__ = ("head", "tail", "size")

    def __init__(self) -> None:
        self.head = KVCacheBlock(block_id=-1)
        self.tail = KVCacheBlock(block_id=-1)
        self.head.next_free_block = self.tail
        self.tail.prev_free_block = self.head
        self.size = 0

    def append(self, block: "KVCacheBlock") -> None:
        last = self.tail.prev_free_block
        assert last is not None
        last.next_free_block = block
        block.prev_free_block = last
        block.next_free_block = self.tail
        self.tail.prev_free_block = block
        self.size += 1

    def append_many(self, blocks: list["KVCacheBlock"]) -> None:
        if not blocks:
            return
        last = self.tail.prev_free_block
        assert last is not None
        for b in blocks:
            b.prev_free_block = last
            last.next_free_block = b
            last = b
        last.next_free_block = self.tail
        self.tail.prev_free_block = last
        self.size += len(blocks)

    def prepend_many(self, blocks: list["KVCacheBlock"]) -> None:
        """Insert blocks at the LRU end (front). Used by the TTL sweep
        to drop demoted-from-must blocks at the front of the may pool
        so they evict before may's existing entries (next-to-evict
        semantics). See docs/v2/32 §2.4.

        DEPRECATED for the M17 sweep path; kept for backward compat
        with any caller that genuinely wants LRU-end prepend.
        ``insert_by_last_access`` / ``merge_sorted_by_last_access`` are
        the access-time-correct replacements.
        """
        if not blocks:
            return
        first = self.head.next_free_block
        assert first is not None
        # Splice [blocks...] between head and current first.
        prev = self.head
        for b in blocks:
            b.prev_free_block = prev
            prev.next_free_block = b
            prev = b
        prev.next_free_block = first
        first.prev_free_block = prev
        self.size += len(blocks)

    # M17 (CODE_MISMATCH_NOTES.md): last_access_ns-correct insertion.
    # ``head`` is the LRU end (pop direction); ``tail`` is the MRU end.
    # The pool invariant for the two helpers below is
    # ``head.next.last_access_ns <= ... <= tail.prev.last_access_ns``
    # (sorted ascending by access time). Used by the TTL sweep
    # (batched merge) and by the lazy-flush single-block move so
    # demoted blocks land at the LRU position corresponding to their
    # actual recent-access pattern, instead of always at the LRU end
    # (prepend_many) or MRU end (append). See docs/v2/32 §2.3
    # ("Demotion (must -> may): insert at LRU-correct position in may
    # pool by the block's `last_access_ns`").
    def insert_by_last_access(self, block: "KVCacheBlock") -> int:
        """Insert ``block`` at the position that keeps the pool sorted
        ascending by ``last_access_ns`` (LRU at head, MRU at tail).

        Walk from the MRU end (tail) backwards; the first cursor whose
        ``last_access_ns <= block.last_access_ns`` is the insertion
        predecessor. O(n) worst case; in practice short walks dominate
        because demoted blocks tend to be older than typical may
        entries (LRU-end target). Returns the number of cursor steps
        walked (used by ``FreeKVCacheBlockQueue`` for walk-depth
        telemetry per the M17 spec).
        """
        target = block.last_access_ns
        cursor = self.tail.prev_free_block
        assert cursor is not None
        steps = 0
        while cursor is not self.head and cursor.last_access_ns > target:
            steps += 1
            prev = cursor.prev_free_block
            assert prev is not None
            cursor = prev
        # Insert AFTER cursor (cursor is either the head sentinel or a
        # block whose last_access_ns <= block.last_access_ns).
        nxt = cursor.next_free_block
        assert nxt is not None
        block.prev_free_block = cursor
        block.next_free_block = nxt
        cursor.next_free_block = block
        nxt.prev_free_block = block
        self.size += 1
        return steps

    def merge_sorted_by_last_access(
        self, blocks_sorted_asc: list["KVCacheBlock"]
    ) -> int:
        """Merge a list of blocks (already sorted ascending by
        ``last_access_ns``) into the pool, preserving the
        sorted-by-access-time invariant. Two-pointer merge against the
        existing pool walking from head -> tail; O(k + n_existing)
        where k = len(blocks_sorted_asc), n_existing = self.size.

        Returns the cumulative cursor-walk distance used (each
        existing block stepped over counts as one step). The TTL sweep
        and any other batched demote path should use this rather than
        looping over ``insert_by_last_access`` to avoid the O(k * n)
        worst case of repeated single-block walks.
        """
        if not blocks_sorted_asc:
            return 0
        # Walk pool from head; each new block is spliced in at the
        # first cursor whose last_access_ns > new_block.last_access_ns
        # (which preserves the ascending order). cursor starts at the
        # block AFTER head; we hold prev = head and step prev <- cursor
        # for each cursor we pass.
        steps = 0
        prev = self.head
        cursor = self.head.next_free_block
        assert cursor is not None
        for new_block in blocks_sorted_asc:
            target = new_block.last_access_ns
            # Advance cursor past every existing block whose
            # last_access_ns <= target (those belong before the new
            # block to keep the pool ascending).
            while cursor is not self.tail and cursor.last_access_ns <= target:
                steps += 1
                prev = cursor
                cursor = cursor.next_free_block
                assert cursor is not None
            # Splice new_block between prev and cursor.
            new_block.prev_free_block = prev
            new_block.next_free_block = cursor
            prev.next_free_block = new_block
            cursor.prev_free_block = new_block
            # The new block becomes the predecessor for the next
            # incoming block; cursor stays pointing at the next
            # existing entry (or tail).
            prev = new_block
        self.size += len(blocks_sorted_asc)
        return steps

    def remove(self, block: "KVCacheBlock") -> None:
        prev = block.prev_free_block
        nxt = block.next_free_block
        assert prev is not None and nxt is not None, (
            f"remove() called on a block not in this pool: {block}"
        )
        prev.next_free_block = nxt
        nxt.prev_free_block = prev
        block.prev_free_block = None
        block.next_free_block = None
        self.size -= 1

    def popleft_one(self) -> "KVCacheBlock | None":
        first = self.head.next_free_block
        if first is self.tail or first is None:
            return None
        nxt = first.next_free_block
        assert nxt is not None
        self.head.next_free_block = nxt
        nxt.prev_free_block = self.head
        first.prev_free_block = None
        first.next_free_block = None
        self.size -= 1
        return first

    def iter_blocks(self) -> Iterator["KVCacheBlock"]:
        b = self.head.next_free_block
        while b is not None and b is not self.tail:
            yield b
            b = b.next_free_block


VICTIM_POLICY_WIRES_THREE_POOL = "wires_three_pool"
VICTIM_POLICY_PURE_LRU = "pure_lru"
SUPPORTED_VICTIM_POLICIES = (VICTIM_POLICY_WIRES_THREE_POOL, VICTIM_POLICY_PURE_LRU)

# WIRES Phase C: structured-source TTL_must default = 5 minutes.
# Per docs/v2/32 DL-8: structured TTL is a heuristic constant in v1
# (refined later via per-segment next-consumer ETA from monitor).
# Unstructured TTL is the cross-class fairness lever and is held until
# the expert-brief design lands (see docs/v2/33 OQ-Fairness).
_DEFAULT_STRUCTURED_TTL_MS = 300_000
# Phase C3 (docs/v2/32 §2.4.1): bootstrap TTL applied to
# unstructured-source promotions until the EMA estimator has at
# least N_BOOTSTRAP samples. After that the estimator drives TTL
# (snapshot at promotion: T̂_u + k * sqrt(Var_u)).
_DEFAULT_UNSTRUCTURED_TTL_BOOTSTRAP_MS = 60_000
# M20 (paper §3.2 future-work extension; CODE_MISMATCH_NOTES.md M20):
# speculative-source TTL backstop, default 30s. Speculative blocks are
# proactively promoted from may to must by a runner hint that
# anticipates future reuse; a much shorter TTL than structured (5min)
# or unstructured-bootstrap (60s) bounds the cost of a misprediction
# while still giving the speculated reuse a reasonable window. Class
# upgrades to "unstructured" on the first cache hit, after which the
# normal EMA-derived TTL takes over.
_DEFAULT_SPECULATIVE_TTL_MS = 30_000
_DEFAULT_UNSTRUCTURED_BOOTSTRAP_SAMPLES = 50
# Phase C3 EMA smoothing constant (~last 20 samples weighted).
_DEFAULT_UNSTRUCTURED_EMA_ALPHA = 0.05
# Phase C3 buffer thickness `k`. Per docs/v2/32 §3.1.4 (Cantelli
# correction): Pr(actual >= T̂ + k*ε) ≤ 1/(1+k²). With k=4 → ≤5.9%
# TTL-induced miss rate.
#
# M14 (paper §3.6, post-2026-05-13 rewrite): `k` is normally derived
# adaptively from `p̂_h` (engine-observed EMA of hinted-block survival)
# via `k = sqrt(p̂_h / (1 − p̂_h))` clamped to `[_K_MIN, _K_MAX]`. The
# WIRES_KVCACHE_UNSTRUCTURED_K env var, when set, overrides the
# derivation entirely (legacy fixed-k mode for A/B comparison). The
# default below is unused once adaptive k is active; it only takes
# effect via the env override.
_DEFAULT_UNSTRUCTURED_K = 4.0
_K_MIN = 1.0
_K_MAX = 10.0
# M14: initial p̂_h before any samples are observed. 0.5 → k = 1
# (the floor), giving conservative-but-nontrivial vanilla TTLs out
# of the gate. After WIRES_PH_EMA_HALFLIFE samples the EMA reflects
# observed behaviour.
_DEFAULT_PH_INIT = 0.5
# M14: EMA half-life in samples. After N = halflife samples of a
# constant indicator, the EMA reaches halfway between the initial
# value and the new constant. Translates to smoothing constant
# alpha_ema = 1 − 0.5**(1/N).
_DEFAULT_PH_EMA_HALFLIFE = 100
# Phase C4 α-shrinkage was removed in the M16 cleanup (2026-05-13
# follow-up). Capacity pressure on the must pool is now handled
# exclusively by the `must_pool_evicted` backstop in `popleft_n`
# (docs/v2/32 §2.2 step 3). The earlier α scaling in `_sweep_ttl_must`
# added code complexity with no paper claim (paper §3.6 does not
# mention α) and is gone.

# Phase D access-based promotion threshold (DL-OQ5 = 1).
_DEFAULT_ACCESS_PROMOTION_THRESHOLD = 1


def _resolve_ttl_ns_from_env(env_name: str, default_ms: int) -> int:
    """Resolve ``WIRES_KVCACHE_*_TTL_MS`` env var to nanoseconds.

    A value of 0 disables the TTL sweep for that source class
    (blocks of that class live in must until force-evicted by
    `popleft_n` step 3). Useful for A/B isolating per-class
    behaviour.
    """
    raw = os.environ.get(env_name)
    if raw is None:
        return default_ms * 1_000_000
    try:
        ms = int(raw)
    except ValueError as e:
        raise ValueError(f"{env_name} must be an integer (got {raw!r})") from e
    if ms < 0:
        raise ValueError(f"{env_name} must be >= 0 (got {ms})")
    return ms * 1_000_000


def _resolve_structured_ttl_ns_from_env() -> int:
    return _resolve_ttl_ns_from_env(
        "WIRES_KVCACHE_STRUCTURED_TTL_MS", _DEFAULT_STRUCTURED_TTL_MS
    )


def _resolve_unstructured_bootstrap_ttl_ns_from_env() -> int:
    return _resolve_ttl_ns_from_env(
        "WIRES_KVCACHE_UNSTRUCTURED_BOOTSTRAP_TTL_MS",
        _DEFAULT_UNSTRUCTURED_TTL_BOOTSTRAP_MS,
    )


def _resolve_speculative_ttl_ns_from_env() -> int:
    """M20: resolve ``WIRES_KVCACHE_SPECULATIVE_TTL_MS`` env var to ns.

    Default 30s. A value of 0 disables the speculative TTL sweep
    (speculative blocks then live in must until force-evicted by
    ``popleft_n`` step 3 or upgraded to unstructured by
    ``BlockPool.touch()`` on first reuse).
    """
    return _resolve_ttl_ns_from_env(
        "WIRES_KVCACHE_SPECULATIVE_TTL_MS", _DEFAULT_SPECULATIVE_TTL_MS
    )


def _resolve_float_from_env(env_name: str, default: float) -> float:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{env_name} must be a float (got {raw!r})") from e


def _resolve_int_from_env(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{env_name} must be an integer (got {raw!r})") from e


def _resolve_access_promotion_threshold_from_env() -> int:
    """Resolve ``WIRES_KVCACHE_ACCESS_PROMOTION_THRESHOLD`` env var.

    A value of 0 disables access-based promotion entirely (Phase D
    becomes inert; only explicit promotion via update_block_hint
    fires). Default 1 per DL-OQ5: any cached block hit at least
    once gets promoted to must.
    """
    raw = os.environ.get("WIRES_KVCACHE_ACCESS_PROMOTION_THRESHOLD")
    if raw is None:
        return _DEFAULT_ACCESS_PROMOTION_THRESHOLD
    try:
        threshold = int(raw)
    except ValueError as e:
        raise ValueError(
            f"WIRES_KVCACHE_ACCESS_PROMOTION_THRESHOLD must be an integer (got {raw!r})"
        ) from e
    if threshold < 0:
        raise ValueError(
            f"WIRES_KVCACHE_ACCESS_PROMOTION_THRESHOLD must be >= 0 (got {threshold})"
        )
    return threshold


def _resolve_victim_policy_from_env() -> str:
    """Resolve ``WIRES_KVCACHE_VICTIM_POLICY`` env var to a known mode.

    Defaults to ``wires_three_pool`` (the WIRES design, docs/v2/32 §2).
    Set to ``pure_lru`` to run baseline / related-work comparisons with
    stock vLLM single-FIFO behavior — lifecycle hints are ignored and
    every block flows through one queue in insertion order.
    """
    raw = os.environ.get("WIRES_KVCACHE_VICTIM_POLICY", VICTIM_POLICY_WIRES_THREE_POOL)
    if raw not in SUPPORTED_VICTIM_POLICIES:
        raise ValueError(
            f"Unknown WIRES_KVCACHE_VICTIM_POLICY={raw!r}; "
            f"supported values: {SUPPORTED_VICTIM_POLICIES}"
        )
    return raw


class FreeKVCacheBlockQueue:
    """Mode-dispatched eviction queue.

    Mode is resolved from ``WIRES_KVCACHE_VICTIM_POLICY`` (or the
    explicit ``mode=`` kwarg, primarily for testing):

      - ``wires_three_pool`` (default): 3-pool design per docs/v2/32 §2.
        Internally maintains three independent doubly-linked lists, one
        per ``lifecycle_hint`` value (``no`` / ``may`` / ``must``).
        ``popleft_n`` walks pools in priority order; each step is O(k)
        on its own pool — no full-queue scans.

      - ``pure_lru``: stock vLLM single-FIFO behavior. Every block is
        placed in the may pool regardless of its lifecycle_hint;
        ``popleft_n`` returns from may in insertion order. Lifecycle
        hints become advisory metadata (still tracked on the block,
        ignored by eviction). Use this mode for baseline / related-work
        comparisons that need stock vLLM's exact eviction policy.

    Public API is preserved relative to the legacy implementation:
      - ``popleft()``, ``popleft_n(n)``, ``append(block)``,
        ``append_n(blocks)``, ``remove(block)``, ``num_free_blocks``,
        ``get_all_free_blocks()``.
      - Backward-compat: blocks default ``lifecycle_hint = "may"`` →
        all legacy paths land in the may pool, reproducing legacy LRU
        exactly. Existing tests pass against either mode.

    New API (used by Phase B):
      - ``update_block_hint(block, new_hint)`` — flip a block's hint
        and move pools if it's currently in the queue. No-op in
        ``pure_lru`` mode (hint is metadata only).

    Telemetry: ``must_pool_evicted_count`` (docs/v2/32 §2.6) increments
    when ``popleft_n`` takes blocks from the must pool. Always 0 in
    ``pure_lru`` mode (no must pool concept).
    """

    POOL_ORDER: tuple[str, str, str] = ("no", "may", "must")

    def __init__(
        self,
        blocks: list[KVCacheBlock],
        mode: str | None = None,
        structured_ttl_ns: int | None = None,
        unstructured_bootstrap_ttl_ns: int | None = None,
        speculative_ttl_ns: int | None = None,
        access_promotion_threshold: int | None = None,
        unstructured_k: float | None = None,
        ema_alpha: float | None = None,
        bootstrap_samples: int | None = None,
        total_capacity: int | None = None,
        ph_init: float | None = None,
        ph_ema_halflife: int | None = None,
    ) -> None:
        self.mode: str = mode if mode is not None else _resolve_victim_policy_from_env()
        if self.mode not in SUPPORTED_VICTIM_POLICIES:
            raise ValueError(
                f"Unknown FreeKVCacheBlockQueue mode={self.mode!r}; "
                f"supported: {SUPPORTED_VICTIM_POLICIES}"
            )
        self._structured_ttl_ns: int = (
            structured_ttl_ns
            if structured_ttl_ns is not None
            else _resolve_structured_ttl_ns_from_env()
        )
        self._unstructured_bootstrap_ttl_ns: int = (
            unstructured_bootstrap_ttl_ns
            if unstructured_bootstrap_ttl_ns is not None
            else _resolve_unstructured_bootstrap_ttl_ns_from_env()
        )
        # M20: speculative-source TTL backstop (default 30s). See
        # CODE_MISMATCH_NOTES.md M20 for design rationale.
        self._speculative_ttl_ns: int = (
            speculative_ttl_ns
            if speculative_ttl_ns is not None
            else _resolve_speculative_ttl_ns_from_env()
        )
        self.access_promotion_threshold: int = (
            access_promotion_threshold
            if access_promotion_threshold is not None
            else _resolve_access_promotion_threshold_from_env()
        )
        # Phase C3 / M14: buffer thickness `k`.
        #
        # Resolution order (first match wins):
        #   1. Explicit constructor kwarg `unstructured_k` — overrides
        #      everything (used by tests / callers that want a fixed k).
        #   2. WIRES_KVCACHE_UNSTRUCTURED_K env var — legacy fixed-k
        #      override, kept for A/B comparison against adaptive k.
        #   3. Adaptive: derived per-call from `p̂_h` via
        #      `k = sqrt(p̂_h / (1 − p̂_h))` clamped to [_K_MIN, _K_MAX].
        #
        # `_unstructured_k_override` is the resolved fixed value when
        # cases 1 or 2 apply; `None` means adaptive. `_unstructured_k`
        # (legacy attribute name kept for back-compat with tests that
        # read it directly) mirrors the *currently effective* `k`,
        # refreshed each time `_feed_p_h_sample` runs.
        env_k_raw = os.environ.get("WIRES_KVCACHE_UNSTRUCTURED_K")
        if unstructured_k is not None:
            self._unstructured_k_override: float | None = float(unstructured_k)
        elif env_k_raw is not None:
            self._unstructured_k_override = _resolve_float_from_env(
                "WIRES_KVCACHE_UNSTRUCTURED_K", _DEFAULT_UNSTRUCTURED_K
            )
        else:
            self._unstructured_k_override = None
        # Mirror of the currently effective k (telemetry).
        if self._unstructured_k_override is not None:
            self._unstructured_k: float = self._unstructured_k_override
        else:
            # Initial value; refreshed below once p̂_h init lands.
            self._unstructured_k = _K_MIN
        self._ema_alpha: float = (
            ema_alpha
            if ema_alpha is not None
            else _resolve_float_from_env(
                "WIRES_KVCACHE_UNSTRUCTURED_EMA_ALPHA",
                _DEFAULT_UNSTRUCTURED_EMA_ALPHA,
            )
        )
        self._bootstrap_samples: int = (
            bootstrap_samples
            if bootstrap_samples is not None
            else _resolve_int_from_env(
                "WIRES_KVCACHE_UNSTRUCTURED_BOOTSTRAP_SAMPLES",
                _DEFAULT_UNSTRUCTURED_BOOTSTRAP_SAMPLES,
            )
        )
        # EMA running state. Start at 0 — until n_samples >=
        # bootstrap_samples, _unstructured_ttl_at_promotion_ns()
        # returns the bootstrap constant, NOT a function of these
        # state values, so initial values don't gate snapshot
        # quality. After bootstrap, EMA is built up purely from
        # observed samples.
        self._T_hat_u_ns: float = 0.0
        self._var_u_ns2: float = 0.0
        self._n_samples_unstructured: int = 0

        # M14: p̂_h EMA — engine-observed survival rate of hinted
        # blocks under runner control. Sampled exactly once per
        # runner-driven hinted-must demote (`update_block_hint` with
        # new_hint != "must" on a structured-class must block); the
        # T_h backstop sweep does NOT sample (its demotes signal
        # monitor failure, not survival).
        ph_init_resolved: float = (
            ph_init
            if ph_init is not None
            else _resolve_float_from_env("WIRES_PH_INIT", _DEFAULT_PH_INIT)
        )
        if not 0.0 < ph_init_resolved < 1.0:
            raise ValueError(f"ph_init must be in (0, 1), got {ph_init_resolved}")
        halflife_resolved: int = (
            ph_ema_halflife
            if ph_ema_halflife is not None
            else _resolve_int_from_env(
                "WIRES_PH_EMA_HALFLIFE", _DEFAULT_PH_EMA_HALFLIFE
            )
        )
        if halflife_resolved < 1:
            raise ValueError(f"ph_ema_halflife must be >= 1, got {halflife_resolved}")
        self._p_h_init: float = ph_init_resolved
        self._p_h_hat: float = ph_init_resolved
        self._p_h_ema_halflife: int = halflife_resolved
        # alpha = 1 − 0.5**(1/halflife): after `halflife` samples of
        # a constant indicator, EMA is halfway from init to that value.
        self._p_h_ema_alpha: float = 1.0 - (0.5 ** (1.0 / halflife_resolved))
        self._n_samples_p_h: int = 0
        # Refresh adaptive `k` to reflect the current p̂_h at init.
        if self._unstructured_k_override is None:
            self._unstructured_k = self._derive_k_from_ph(self._p_h_hat)

        # Total capacity B (block_pool passes this in). Retained for
        # telemetry / future use; α-shrinkage no longer reads it
        # (removed in the M16 cleanup, 2026-05-13 follow-up).
        self._total_capacity: int | None = total_capacity

        self._pools: dict[str, _PoolList] = {
            name: _PoolList() for name in self.POOL_ORDER
        }
        # Track which pool each block was placed in. Looking up by
        # block_id avoids reading ``block.lifecycle_hint`` at remove
        # time — that field can be mutated externally between insert
        # and remove (e.g. via segment hint update); pool membership
        # must remain consistent regardless.
        self._pool_of: dict[int, str] = {}
        now_ns = time.monotonic_ns()
        for block in blocks:
            pool_name = self._pool_for_block(block)
            if pool_name == "must":
                # Initial placement counts as promotion-time for TTL.
                # `source_class` is dataclass-defaulted to "unstructured"
                # (M16 cleanup: the default flipped from "structured" to
                # match paper §2.3 — explicit CFG promotion is the
                # special case, access-based is the dominant path).
                self._stamp_must_promotion(block, now_ns, block.source_class)
            self._pools[pool_name].append(block)
            self._pool_of[block.block_id] = pool_name

        # Telemetry counter (docs/v2/32 §2.6). Always 0 in pure_lru mode.
        self.must_pool_evicted_count: int = 0
        # Telemetry counter (docs/v2/32 §2.4). Cumulative count of blocks
        # demoted from must -> may by the lazy TTL sweep at popleft_n.
        self.ttl_demoted_count: int = 0
        # WIRES Phase D telemetry: cumulative count of blocks promoted
        # may -> must via access threshold (called by block_pool.touch).
        self.access_promoted_count: int = 0
        # WIRES Phase C3 telemetry: cumulative EMA samples fed; running
        # estimator state read-only via T_hat_u_ns / sigma_u_ns props.
        self.unstructured_ema_sample_count: int = 0
        # M14 telemetry: cumulative count of hinted-demote samples fed
        # into p̂_h. Together with `p_h_hat` and `unstructured_k_current`
        # gives the runner / metrics scrape full visibility into the
        # adaptive-k loop.
        self.p_h_ema_sample_count: int = 0
        # M20 telemetry (speculative source class):
        #   speculation_promotion_count: total speculative may->must
        #     promotions (incremented when a block enters the must pool
        #     with source_class == "speculative").
        #   speculation_hit_count: speculative blocks that got reused
        #     (cache-hit) before TTL fired (incremented in
        #     BlockPool.touch() when the speculative -> unstructured
        #     class upgrade happens).
        #   speculation_miss_count: speculative blocks demoted by the
        #     TTL sweep without ever being hit (incremented in
        #     _sweep_ttl_must when the demoted block has source_class ==
        #     "speculative").
        # Derived hit rate via the ``speculation_hit_rate`` property.
        self.speculation_promotion_count: int = 0
        self.speculation_hit_count: int = 0
        self.speculation_miss_count: int = 0

        # M19 (CODE_MISMATCH_NOTES.md): engine-side lazy hint flips.
        # Per-block pending pool moves are coalesced here and applied
        # at the entry of any cache op that reads pool state
        # (popleft / popleft_n / append / append_n / remove / get_all).
        # ``block.lifecycle_hint`` is ALWAYS updated eagerly (it is
        # just metadata read by other modules); only the doubly-linked
        # list move is deferred. Multiple flips on the same block
        # within one window collapse to the final
        # ``(block, new_hint, source_class)`` entry in this dict, which
        # is what motivates the design (a fan-out join hint-flipping
        # the same block N times costs N dict writes + 1 pool move at
        # the next cache op, not N pool moves). Storing the block
        # object directly avoids any O(num_free_blocks) lookup at
        # flush time.
        self._pending_hint_flips: dict[
            int, tuple[KVCacheBlock, str, str]
        ] = {}
        # M19 telemetry: cumulative count of pending flips actually
        # applied by ``_flush_pending_hint_flips`` (sum across all
        # flushes). When this grows much faster than the count of
        # distinct cache ops that triggered the flush, the lazy
        # batching is doing its job; when they grow at the same rate,
        # the batching window only ever sees one pending flip and the
        # deferral is overhead-only.
        self.lazy_flush_total_blocks: int = 0

        # M17 (CODE_MISMATCH_NOTES.md): walk-depth telemetry for the
        # last_access_ns-correct insertion path. Each call to
        # ``_PoolList.insert_by_last_access`` /
        # ``_PoolList.merge_sorted_by_last_access`` returns its cursor
        # step count; the FreeKVCacheBlockQueue accumulates the total
        # plus a coarse histogram so we can monitor whether the O(n)
        # walk is acceptable in practice. If the >=1000 bucket grows
        # under load, that's the trigger to switch to a sortedcontainers
        # SortedKeyList or a bucket-by-timestamp scheme.
        #   buckets[0]: walks  < 10 steps
        #   buckets[1]: walks  < 100 steps
        #   buckets[2]: walks  < 1000 steps
        #   buckets[3]: walks >= 1000 steps
        self.lru_insert_walk_steps_total: int = 0
        self.lru_insert_count: int = 0
        self.lru_insert_walk_depth_buckets: list[int] = [0, 0, 0, 0]

    @property
    def T_hat_u_ns(self) -> float:
        """Current EMA estimate of unstructured prefix-reuse interval (ns)."""
        return self._T_hat_u_ns

    @property
    def sigma_u_ns(self) -> float:
        """Current EMA-derived stddev of unstructured prefix-reuse interval (ns)."""
        # Var can drift slightly negative under floating-point noise.
        return (max(self._var_u_ns2, 0.0)) ** 0.5

    # --- M14: p̂_h EMA + adaptive k ---------------------------------------
    @property
    def p_h_hat(self) -> float:
        """Current EMA estimate of hinted-block survival rate (paper §3.6).

        Sampled exactly once per runner-driven hinted-must demote
        (`update_block_hint(may|no)` on a structured-class must block).
        """
        return self._p_h_hat

    @property
    def unstructured_k_current(self) -> float:
        """Current effective `k` for vanilla TTL `T_v(b) = T̂_v + k·ε_v`.

        Returns the env / kwarg override when set; otherwise the
        adaptive `k = sqrt(p̂_h / (1 − p̂_h))` clamped to
        [_K_MIN, _K_MAX] using the latest p̂_h.
        """
        if self._unstructured_k_override is not None:
            return self._unstructured_k_override
        return self._derive_k_from_ph(self._p_h_hat)

    @property
    def speculation_hit_rate(self) -> float | None:
        """M20: derived metric, hits / promotions for speculative blocks.

        Returns ``None`` when no speculative promotions have happened
        (avoid 0/0 noise in the cache_stats surface). When the rate
        falls below ~50% the speculation pattern probably isn't
        worthwhile per the v1 simplification rule (paper §3.2 future
        work / CODE_MISMATCH_NOTES.md M20).
        """
        if self.speculation_promotion_count <= 0:
            return None
        return self.speculation_hit_count / self.speculation_promotion_count

    # --- M5(a): telemetry counter snapshot --------------------------------
    def cache_stats_snapshot(self) -> dict[str, object]:
        """Return a flat dict of every cumulative cache-stats counter
        currently tracked by this queue (CODE_MISMATCH_NOTES.md M5(a)).

        The returned dict is JSON-serialisable and is the canonical
        shape consumed by ``wires_engine_telemetry.emit_cache_stats``
        (E5 stream, ``engine_cache_stats.jsonl``). Keys are stable
        across releases; new counters are added (never renamed) so
        offline analysis can grow without breaking back-fills.

        ``speculation_hit_rate`` is denominator-protected: returns
        ``None`` when ``speculation_promotion_count == 0`` (avoid
        0/0 noise in dashboards).

        Cheap by construction: a few attribute reads + a list copy
        for the bucket histogram. Safe to call from the engine tick
        when telemetry is enabled.
        """
        return {
            # Three-pool eviction / demote / promote (paper §3, docs/v2/32 §2.4 + §2.6).
            "must_pool_evicted_count": self.must_pool_evicted_count,
            "ttl_demoted_count": self.ttl_demoted_count,
            "access_promoted_count": self.access_promoted_count,
            # EMA estimator sample tallies (Phase C3, M14).
            "unstructured_ema_sample_count": self.unstructured_ema_sample_count,
            "p_h_ema_sample_count": self.p_h_ema_sample_count,
            # M20 speculative source class (CODE_MISMATCH_NOTES.md M20).
            "speculation_promotion_count": self.speculation_promotion_count,
            "speculation_hit_count": self.speculation_hit_count,
            "speculation_miss_count": self.speculation_miss_count,
            "speculation_hit_rate": self.speculation_hit_rate,
            # M19 lazy hint flips.
            "lazy_flush_total_blocks": self.lazy_flush_total_blocks,
            # M17 LRU walk-depth histogram + cumulative steps.
            "lru_insert_count": self.lru_insert_count,
            "lru_insert_walk_steps_total": self.lru_insert_walk_steps_total,
            "lru_insert_walk_depth_buckets": list(self.lru_insert_walk_depth_buckets),
        }

    @staticmethod
    def _derive_k_from_ph(p_h: float) -> float:
        """M14: `k = sqrt(p̂_h / (1 − p̂_h))` clamped to [_K_MIN, _K_MAX].

        Pure function so tests can assert behaviour independent of any
        queue state. Defensive clamp keeps `p_h` strictly inside (0, 1)
        so the division never blows up under floating-point noise.
        """
        eps = 1e-9
        p = min(max(p_h, eps), 1.0 - eps)
        k_raw = (p / (1.0 - p)) ** 0.5
        return min(_K_MAX, max(_K_MIN, k_raw))

    def _feed_p_h_sample(self, indicator: int) -> None:
        """M14: feed an indicator (0 or 1) into the p̂_h EMA and refresh
        the adaptive `k`.

        Called from `update_block_hint` exactly once per runner-driven
        hinted-must demote. T_h backstop demotes (driven by
        `_sweep_ttl_must`) do NOT call this — those are excluded from
        the survival statistic per paper §3.6.
        """
        if indicator not in (0, 1):
            raise ValueError(
                f"_feed_p_h_sample expects an indicator in (0, 1), got {indicator}"
            )
        a = self._p_h_ema_alpha
        self._p_h_hat = (1.0 - a) * self._p_h_hat + a * float(indicator)
        self._n_samples_p_h += 1
        self.p_h_ema_sample_count += 1
        # Refresh cached `k` only when adaptive (no env / kwarg override).
        if self._unstructured_k_override is None:
            self._unstructured_k = self._derive_k_from_ph(self._p_h_hat)

    def _feed_unstructured_sample(self, interval_ns: int) -> None:
        """Phase C3: update EMA with a new prefix-reuse interval sample.

        Called from BlockPool.touch when a must-pool block with
        source_class == 'unstructured' gets a cache hit. Bootstrap
        defaults stay in effect until n_samples >= _bootstrap_samples.
        """
        if interval_ns <= 0:
            return
        a = self._ema_alpha
        x = float(interval_ns)
        if self._n_samples_unstructured == 0:
            # First sample: seed estimator at observed value, var stays
            # at bootstrap (still considered unconverged).
            self._T_hat_u_ns = x
        else:
            self._T_hat_u_ns = (1.0 - a) * self._T_hat_u_ns + a * x
        delta = x - self._T_hat_u_ns
        self._var_u_ns2 = (1.0 - a) * self._var_u_ns2 + a * (delta * delta)
        self._n_samples_unstructured += 1
        self.unstructured_ema_sample_count += 1

    def _unstructured_ttl_at_promotion_ns(self) -> int:
        """Phase C3 + M14: snapshot value for ttl_at_promotion_ns at
        unstructured may->must promotion time.

        Returns the bootstrap constant until the EMA estimator has
        ``_bootstrap_samples`` interval samples; once converged,
        snapshots ``T̂_v + k · ε_v`` using the CURRENT effective k
        (env/kwarg override or adaptive derivation from p̂_h).
        """
        if self._n_samples_unstructured < self._bootstrap_samples:
            return self._unstructured_bootstrap_ttl_ns
        return int(self._T_hat_u_ns + self.unstructured_k_current * self.sigma_u_ns)

    def _pool_for_block(self, block: KVCacheBlock) -> str:
        """Resolve which pool ``block`` should land in given the active mode."""
        if self.mode == VICTIM_POLICY_PURE_LRU:
            return "may"
        pool_name = block.lifecycle_hint
        if pool_name not in self._pools:
            pool_name = "may"
        return pool_name

    # --- M19: engine-side lazy hint flips --------------------------------
    def _do_real_pool_move(
        self,
        block: KVCacheBlock,
        new_hint: str,
        source_class: str,
    ) -> None:
        """Apply a single deferred hint flip to a block currently sitting
        in a free pool: remove from the current pool, optionally stamp
        must-promotion metadata, and append into the destination pool.

        Caller (`_flush_pending_hint_flips`) is responsible for verifying
        the block is still in a pool. We re-check here because the
        block may have been popped (eviction) or removed (touch hit)
        between the deferred record time and now.
        """
        current_pool = self._pool_of.get(block.block_id)
        if current_pool is None:
            # Block has been evicted or pulled into in-use since the
            # flip was recorded; the eager update of ``lifecycle_hint``
            # is the only state that mattered, and it has already
            # happened. Nothing else to do.
            return
        # If the block is already in the right pool (e.g., back-to-back
        # flip ended up where it started), skip the move.
        if current_pool == new_hint:
            return
        self._pools[current_pool].remove(block)
        if new_hint == "must":
            # Promotion to must: stamp promoted_at + ttl_at_promotion +
            # source_class (Phase C3). The MRU-end append for must is
            # appropriate (newly promoted -> freshest), so we keep the
            # tail-append here. M17's LRU-position fix targets the
            # demote (must -> may) and the runner-driven non-must
            # transitions where access-time ordering matters for the
            # LRU eviction class.
            self._stamp_must_promotion(block, time.monotonic_ns(), source_class)
            self._pools[new_hint].append(block)
        else:
            # M17: demote (or sideways move) into the no/may pool;
            # insert at the LRU-position correct for this block's
            # actual recent-access pattern.
            steps = self._pools[new_hint].insert_by_last_access(block)
            self._record_lru_insert_walk(steps)
        self._pool_of[block.block_id] = new_hint

    def _record_lru_insert_walk(self, steps: int) -> None:
        """M17: bump the walk-depth telemetry counters for one LRU-
        position insertion (single-block) or one batched merge step.

        ``steps`` is the cursor-walk distance returned by
        ``_PoolList.insert_by_last_access`` /
        ``_PoolList.merge_sorted_by_last_access``. The bucket
        boundaries are <10 / <100 / <1000 / >=1000 to give a coarse
        sense of the walk-depth distribution without dragging in
        per-insert histogram libraries.
        """
        self.lru_insert_count += 1
        self.lru_insert_walk_steps_total += steps
        if steps < 10:
            self.lru_insert_walk_depth_buckets[0] += 1
        elif steps < 100:
            self.lru_insert_walk_depth_buckets[1] += 1
        elif steps < 1000:
            self.lru_insert_walk_depth_buckets[2] += 1
        else:
            self.lru_insert_walk_depth_buckets[3] += 1

    def _flush_pending_hint_flips(self) -> int:
        """Apply all deferred hint flips. Called at the entry of any
        cache op that depends on pool state being up to date
        (``popleft``, ``popleft_n``, ``append``, ``append_n``,
        ``remove``, ``get_all_free_blocks``).

        Returns the number of pending flips actually applied (always
        equal to ``len(self._pending_hint_flips)`` at entry; useful as
        a per-call counter for tests).
        """
        if not self._pending_hint_flips:
            return 0
        # Snapshot + clear BEFORE iterating so any nested cache op
        # (defensive: none exists today, but cheap to guarantee) sees
        # an empty pending dict and does not re-flush the same entry.
        pending = self._pending_hint_flips
        self._pending_hint_flips = {}
        applied = 0
        for _block_id, (block, new_hint, source_class) in pending.items():
            self._do_real_pool_move(block, new_hint, source_class)
            applied += 1
        self.lazy_flush_total_blocks += applied
        return applied

    # --- Backward-compat fake-head/tail aliases ---------------------------
    # Legacy tests inspect ``fake_free_list_head`` / ``fake_free_list_tail``
    # directly. Default-may blocks all land in the may pool, so exposing
    # the may pool's sentinels here lets those tests continue to pass.
    @property
    def fake_free_list_head(self) -> KVCacheBlock:
        return self._pools["may"].head

    @property
    def fake_free_list_tail(self) -> KVCacheBlock:
        return self._pools["may"].tail

    # --- Aggregate state --------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return (
            self._pools["no"].size + self._pools["may"].size + self._pools["must"].size
        )

    def num_free_blocks_in_pool(self, pool: str) -> int:
        return self._pools[pool].size

    # --- Pop primitives ---------------------------------------------------
    def _sweep_ttl_must(self) -> int:
        """Lazy TTL sweep on the must pool head (docs/v2/32 §2.4 + DL-10).

        Walks the must pool from the head; any block whose deadline
        (`last_promoted_ns + ttl_at_promotion_ns`) has elapsed is
        demoted to the may pool's HEAD (LRU end — they evict before
        may's existing entries on the next popleft_n). Stops at the
        first non-expired block; since pool ordering reflects promotion
        time, all subsequent blocks are fresher.

        M16 cleanup (2026-05-13): α-shrinkage was removed entirely.
        Capacity pressure on the must pool is now handled exclusively
        via the `must_pool_evicted` backstop in `popleft_n` (no
        retroactive shortening of per-block TTLs).

        No-op when:
        - mode is pure_lru (must pool is always empty there)
        - structured_ttl_ns is 0 (TTL disabled)
        - must pool is empty

        Returns the number of blocks demoted.
        """
        # M19: apply any deferred hint flips so the must-pool head
        # walked below reflects the latest pool placement. Without
        # this, a block freshly promoted into must by a runner hint
        # (but whose move was deferred) would still appear in may and
        # the sweep would skip it; conversely, a block freshly demoted
        # out of must (deferred) would still be in must and could be
        # reaped twice.
        self._flush_pending_hint_flips()
        if self.mode == VICTIM_POLICY_PURE_LRU:
            return 0
        must = self._pools["must"]
        if must.size == 0:
            return 0
        now_ns = time.monotonic_ns()
        # Walk from head; demote blocks where
        # `now > promoted_at_ns + ttl_at_promotion_ns`. Mixed source
        # classes at head: the strict "stop at first non-expired" trick
        # doesn't strictly hold (structured may have longer TTL than
        # the unstructured behind it), but pragmatically we stop at
        # first non-expired — any expired blocks behind a non-expired
        # head get caught next cycle. Worst case: 1-cycle demotion
        # delay. Acceptable.
        demoted: list[KVCacheBlock] = []
        while must.size > 0:
            head = must.head.next_free_block
            if head is None or head is must.tail:
                break
            ttl = head.ttl_at_promotion_ns
            if ttl <= 0:
                # No TTL set (block was never properly promoted via
                # the new path); skip rather than demote. Could
                # happen during transition / for legacy blocks.
                break
            deadline_ns = head.last_promoted_ns + ttl
            if deadline_ns > now_ns:
                break
            blk = must.popleft_one()
            assert blk is not None
            blk.lifecycle_hint = "may"
            # M20: speculative block aged out without ever being hit;
            # this is the "speculation miss via TTL fallback" path.
            # (Hits are counted in BlockPool.touch() via the class
            # upgrade; reactive may/no demotes go through
            # update_block_hint and are explicitly NOT counted as
            # misses per the spec.)
            if blk.source_class == "speculative":
                self.speculation_miss_count += 1
            demoted.append(blk)
            self._pool_of[blk.block_id] = "may"
        if demoted:
            # M17: demoted blocks were popped from the must pool head
            # in promotion-time order, which correlates with (but is
            # not strictly equal to) last_access_ns order. Sort
            # explicitly so the two-pointer merge below preserves the
            # may pool's ascending invariant. Sort cost is O(k log k)
            # where k = len(demoted); merge is O(k + n_may). Combined
            # this stays well below the per-insert O(k * n_may) we'd
            # see if we looped insert_by_last_access.
            demoted.sort(key=lambda b: b.last_access_ns)
            steps = self._pools["may"].merge_sorted_by_last_access(demoted)
            # Telemetry: count this as one logical "insert" event with
            # the cumulative cursor walk distance. Treating it as one
            # event (rather than k events) keeps the bucket histogram
            # aligned with the per-call cost the caller pays.
            self._record_lru_insert_walk(steps)
            self.ttl_demoted_count += len(demoted)
            # T-45.6 (docs/v2/45 §3.3): emit one E3 row per ttl-demoted
            # block. ``pool_at_evict="must"`` because the block was
            # IN must when the demote fired; ``must_pool_evicted``
            # falls out of that comparison in the emitter so the
            # paper §3 invariant grep still works. ``ttl_at_demote_ms``
            # is the snapshot the block was promoted with.
            if _wires_telemetry_enabled():
                ts_now = time.time()
                for blk in demoted:
                    # M18: emit one E3 row per overlapping segment so
                    # each tagged segment's telemetry sees the eviction
                    # (single-tag fast path: 1 row, identical to pre-M18
                    # behavior). Untagged blocks emit one row with
                    # ``scope_key=None`` per the original contract.
                    scope_keys = blk._segment_ids if blk._segment_ids else (None,)
                    for scope_key in scope_keys:
                        _wires_emit_eviction(
                            block_id=blk.block_id,
                            scope_key=scope_key,
                            pool_at_evict="must",
                            source_class=blk.source_class,
                            hint=blk.lifecycle_hint,
                            reason=EVICT_REASON_TTL_EXPIRED,
                            ttl_at_demote_ms=(
                                blk.ttl_at_promotion_ns / 1_000_000.0
                                if blk.ttl_at_promotion_ns > 0
                                else None
                            ),
                            must_hit_count=blk._must_hit_count,
                            ref_count_at_evict=blk.ref_cnt,
                            ts_epoch=ts_now,
                        )
        return len(demoted)

    def popleft(self) -> KVCacheBlock:
        """Pop the oldest block across all pools, in priority order."""
        self._sweep_ttl_must()
        for name in self.POOL_ORDER:
            blk = self._pools[name].popleft_one()
            if blk is not None:
                self._pool_of.pop(blk.block_id, None)
                if name == "must":
                    self.must_pool_evicted_count += 1
                return blk
        raise ValueError("No free blocks available")

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Pop n blocks per the 3-pool priority algorithm (docs/v2/32 §2.2).

        Step 0: lazy TTL sweep on must pool head (§2.4).
        Step 1-3: take from no -> may -> must in order; each step
        walks only its own pool's tail end (no full-queue scans).
        """
        if n == 0:
            return []

        # Step 0: lazy TTL sweep before deciding eviction order.
        self._sweep_ttl_must()

        ret: list[KVCacheBlock] = []
        must_taken = 0
        # T-45.6 (docs/v2/45 §3.3): track per-pool pop ranges so we
        # can emit one E3 row per popped block at the end. We emit
        # AFTER all pops complete to avoid serialising the hot
        # eviction loop on telemetry; emission still happens inline
        # on the same thread (no IPC) but we keep the deque/lock
        # taps clustered.
        pool_taken: list[tuple[str, KVCacheBlock]] = []
        for name in self.POOL_ORDER:
            pool = self._pools[name]
            while pool.size > 0 and len(ret) < n:
                blk = pool.popleft_one()
                assert blk is not None
                self._pool_of.pop(blk.block_id, None)
                ret.append(blk)
                if _wires_telemetry_enabled():
                    pool_taken.append((name, blk))
                if name == "must":
                    must_taken += 1
            if len(ret) == n:
                break

        if len(ret) < n:
            raise ValueError(
                f"Cannot pop {n} blocks (got {len(ret)}); "
                "free queue is logically full but lifecycle hints prevented eviction"
            )

        if must_taken > 0:
            self.must_pool_evicted_count += must_taken
            # Critical-path WARN log per the proposal: must-pool eviction
            # is a paper §3 invariant violation. Emit a single warning
            # per popleft_n call (NOT per block) so the log doesn't
            # drown a sweep — the JSONL row carries the per-block
            # detail, the log carries the alert.
            logger.warning(
                "wires telemetry: must-pool eviction backstop fired "
                "(must_taken=%d, n=%d, must_pool_evicted_count=%d)",
                must_taken,
                n,
                self.must_pool_evicted_count,
            )

        if pool_taken:
            ts_now = time.time()
            for pool_name, blk in pool_taken:
                # M18: emit one E3 row per overlapping segment (single-tag
                # fast path: 1 row, identical to pre-M18 behavior).
                # Untagged blocks emit one row with ``scope_key=None``
                # per the original contract.
                scope_keys = blk._segment_ids if blk._segment_ids else (None,)
                for scope_key in scope_keys:
                    _wires_emit_eviction(
                        block_id=blk.block_id,
                        scope_key=scope_key,
                        pool_at_evict=pool_name,
                        source_class=blk.source_class,
                        hint=blk.lifecycle_hint,
                        reason=(
                            EVICT_REASON_MUST_PRESSURE_BACKSTOP
                            if pool_name == "must"
                            else EVICT_REASON_LRU_PRESSURE
                        ),
                        ttl_at_demote_ms=(
                            blk.ttl_at_promotion_ns / 1_000_000.0
                            if blk.ttl_at_promotion_ns > 0
                            else None
                        ),
                        must_hit_count=blk._must_hit_count,
                        ref_count_at_evict=blk.ref_cnt,
                        ts_epoch=ts_now,
                    )
        return ret

    def _stamp_must_promotion(
        self, block: KVCacheBlock, now_ns: int, source_class: str
    ) -> None:
        """Phase C3 / M20: stamp promoted_at + ttl_at_promotion +
        source_class on a block as it enters the must pool.

        M14: also resets ``_must_hit_count`` so the indicator that
        feeds p̂_h reflects only hits during this must-residency.

        M20: when ``source_class == "speculative"``, picks the short
        speculative TTL backstop (default 30s) and bumps
        ``speculation_promotion_count``. The class is temporary; on the
        first cache hit ``BlockPool.touch()`` upgrades it to
        ``"unstructured"`` (and counts the hit on
        ``speculation_hit_count``).
        """
        block.last_promoted_ns = now_ns
        block.source_class = source_class
        block._must_hit_count = 0
        if source_class == "structured":
            block.ttl_at_promotion_ns = self._structured_ttl_ns
        elif source_class == "speculative":
            block.ttl_at_promotion_ns = self._speculative_ttl_ns
            self.speculation_promotion_count += 1
        else:
            block.ttl_at_promotion_ns = self._unstructured_ttl_at_promotion_ns()

    # --- Insert / remove primitives ---------------------------------------
    def append(self, block: KVCacheBlock) -> None:
        """Append block to the pool that matches its current lifecycle_hint
        (always the may pool in ``pure_lru`` mode). Stamps must-promotion
        metadata when the destination is the must pool. Reads
        ``block.source_class`` directly (default "unstructured" per
        paper §2.3); explicit CFG callers stamp "structured" via
        ``update_block_hint`` BEFORE this re-append fires."""
        # M19: flush any deferred hint flips for OTHER blocks before
        # this append. The block being appended itself is not in any
        # pool yet, so its own (if any) pending entry is stale and
        # would be a no-op; flushing here only helps other blocks
        # whose deferred state might affect aggregate pool sizes /
        # ordering invariants observed by callers between ops.
        self._flush_pending_hint_flips()
        pool_name = self._pool_for_block(block)
        if pool_name == "must":
            self._stamp_must_promotion(block, time.monotonic_ns(), block.source_class)
        self._pools[pool_name].append(block)
        self._pool_of[block.block_id] = pool_name

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """Bulk append; groups by destination pool to do one splice per pool."""
        if not blocks:
            return
        # M19: flush deferred flips before bulk-appending so the
        # subsequent placement reflects the latest pool state.
        self._flush_pending_hint_flips()
        by_pool: dict[str, list[KVCacheBlock]] = {n: [] for n in self.POOL_ORDER}
        now_ns = time.monotonic_ns()
        for b in blocks:
            pool_name = self._pool_for_block(b)
            if pool_name == "must":
                self._stamp_must_promotion(b, now_ns, b.source_class)
            by_pool[pool_name].append(b)
            self._pool_of[b.block_id] = pool_name
        for name, group in by_pool.items():
            if group:
                self._pools[name].append_many(group)

    def remove(self, block: KVCacheBlock) -> None:
        """Remove block from whichever pool it's currently in."""
        # M19: flush deferred flips so the block is in the pool the
        # caller expects (matches the latest lifecycle_hint metadata).
        # Without this, a block whose hint was just flipped might
        # still be in the OLD pool while ``_pool_of`` reports the OLD
        # pool too, but the caller (e.g., ``BlockPool.touch()``) may
        # have read ``block.lifecycle_hint`` and assumed agreement.
        # Flushing here keeps remove() locally consistent.
        self._flush_pending_hint_flips()
        pool_name = self._pool_of.pop(block.block_id, None)
        if pool_name is None:
            raise RuntimeError(f"remove() called on an unknown block: {block}")
        self._pools[pool_name].remove(block)

    # --- Phase B / external hint update ------------------------------------
    def update_block_hint(
        self,
        block: KVCacheBlock,
        new_hint: str,
        source_class: str = "unstructured",
    ) -> None:
        """Update a block's lifecycle_hint, moving pools if necessary.

        ``wires_three_pool`` mode: if the block is in a free pool and
        the hint change crosses a pool boundary, splice it into the
        new pool's tail. In-use blocks (not in a queue) just have
        their metadata updated; pool placement resolves at next
        append().

        ``pure_lru`` mode: hint is metadata only. Updates the field
        but never moves blocks between pools (everything is in may).

        ``source_class`` defaults to ``"unstructured"`` per paper §2.3
        (access-based promotion is the dominant path; explicit CFG
        promotion is the special case). Callers on the explicit CFG
        path (``segment_actions.update_block_hints`` →
        ``update_segment_lifecycle_hint``) MUST opt in by passing
        ``source_class="structured"``. Phase D's access-based promotion
        path (``try_access_promote``) keeps the default.

        ``source_class`` is recorded on the block ONLY when the new
        hint is "must" (it's only meaningful for must-pool blocks
        whose TTL the sweep needs to pick).
        """
        if new_hint not in self._pools:
            new_hint = "may"
        if source_class not in ("structured", "unstructured", "speculative"):
            raise ValueError(
                f"source_class must be 'structured', 'unstructured', or "
                f"'speculative' (got {source_class!r})"
            )
        if block.lifecycle_hint == new_hint:
            return

        if self.mode == VICTIM_POLICY_PURE_LRU:
            # Hint is metadata only; never move pools.
            block.lifecycle_hint = new_hint
            return

        # M14: sample p̂_h on runner-driven hinted-must demotes.
        # The signal is "≥1 in-must hit during this segment's lifetime".
        # Fires when a must-pool block (lifecycle_hint == "must") with
        # source_class == "structured" (hinted, runner-managed) is
        # demoted to may|no via this method. T_h backstop demotes go
        # through `_sweep_ttl_must` which never calls update_block_hint,
        # so they are naturally excluded per paper §3.6.
        if (
            block.lifecycle_hint == "must"
            and new_hint != "must"
            and block.source_class == "structured"
        ):
            indicator = 1 if block._must_hit_count > 0 else 0
            self._feed_p_h_sample(indicator)

        # M19: eagerly update the metadata so anyone reading
        # ``block.lifecycle_hint`` between now and the next flush sees
        # the latest value. The pool list move itself is deferred.
        block.lifecycle_hint = new_hint

        current_pool = self._pool_of.get(block.block_id)
        if current_pool is None:
            # Block is in-use (not in any free pool). Pool placement
            # will be resolved at the next append() call, which reads
            # ``lifecycle_hint`` (already updated above) and stamps
            # promotion metadata if landing in must. Make sure
            # source_class is recorded so the must-stamp picks up the
            # right ttl_at_promotion when the block re-enters.
            if new_hint == "must":
                # Stamp source_class now so next append() picks up
                # the right ttl_at_promotion. (last_promoted_ns will
                # be set by append's _stamp_must_promotion call.)
                block.source_class = source_class
            # Drop any stale pending flip; block is no longer in a
            # free pool, so the deferred move would be a no-op anyway.
            self._pending_hint_flips.pop(block.block_id, None)
            return

        # M19: block IS in a free pool. Defer the move; collapse with
        # any existing pending entry on the same block (last write
        # wins). If the new hint matches the current physical pool
        # placement (e.g., previous flips already targeted it but were
        # not yet flushed), drop any pending entry: nothing to do.
        if current_pool == new_hint:
            self._pending_hint_flips.pop(block.block_id, None)
            return
        self._pending_hint_flips[block.block_id] = (
            block,
            new_hint,
            source_class,
        )

    # --- Phase D: access-based promotion ------------------------------------
    def try_access_promote(self, block: KVCacheBlock) -> bool:
        """Increment ``block._access_count`` and promote may → must if
        the threshold is reached. Called from ``BlockPool.touch()``
        on every cache hit, BEFORE the block is removed from the
        free queue (so promotion can splice it from may to must
        atomically).

        Returns True if a promotion happened, False otherwise. No-op
        when:
        - mode is pure_lru (hint is ignored)
        - access_promotion_threshold == 0 (Phase D disabled via env)
        - block is the null block
        - block.lifecycle_hint != "may" (already promoted, or
          declared no — both inappropriate for access-based promote)

        Source class is "unstructured" — access-based promotion is
        the canonical unstructured-source promotion path
        (docs/v2/32 §2.4 + DL-8). When the block is in-use (touched
        but already removed from queue), update_block_hint just sets
        the metadata; pool placement on re-free will pick must.
        """
        if self.mode == VICTIM_POLICY_PURE_LRU:
            return False
        if self.access_promotion_threshold <= 0:
            return False
        if getattr(block, "is_null", False):
            return False
        block._access_count += 1
        if block._access_count < self.access_promotion_threshold:
            return False
        if block.lifecycle_hint != "may":
            return False
        # source_class defaults to "unstructured" (the default since
        # M16 cleanup); access-based promotion is the canonical
        # unstructured path so we rely on the default.
        self.update_block_hint(block, "must")
        self.access_promoted_count += 1
        return True

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks across all pools, ordered no → may → must.

        Within each pool the natural front-to-back order is preserved
        (LRU first). For the legacy all-default case (every block has
        ``lifecycle_hint = "may"``), this reproduces exactly the
        single-queue insertion order.
        """
        # M19: caller asked for the authoritative pool snapshot; flush
        # any deferred hint flips so the returned ordering reflects
        # the latest hint state.
        self._flush_pending_hint_flips()
        ret: list[KVCacheBlock] = []
        for name in self.POOL_ORDER:
            ret.extend(self._pools[name].iter_blocks())
        return ret


def need_extra_keys(request: Request) -> bool:
    """Check whether the blocks allocated to this request need extra hash keys.

    Args:
        request (Request): The request.

    Returns:
        bool: Whether blocks allocated to this request need extra hash keys.
    """

    # Multimodal requests need to include the MM hash.
    # LoRA requests need to include the LoRA name.
    # Request with provided cache salt need to include the salt.
    return (
        bool(request.mm_features)
        or (request.lora_request is not None)
        or (request.cache_salt is not None)
    )


def _gen_mm_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[list[Any], int]:
    """Generate extra keys related to MultiModal request for block hash
    computation. For multi-modal inputs, the extra keys are
    (mm_hash, start_offset) that indicate a mm input contained in the
    block and its starting offset in the block tokens.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    extra_keys: list[Any] = []

    mm_features = request.mm_features
    if not mm_features:
        return extra_keys, start_mm_idx

    # Note that we assume mm_features are sorted by mm_position.offset.
    # We do not need to check all mm inputs if the start token index is out of
    # range. This usually happens in the late prefill phase and decoding phase.
    last_pos = mm_features[-1].mm_position
    if last_pos.offset + last_pos.length <= start_token_idx:
        return extra_keys, start_mm_idx

    # Support start_mm_idx == -1 to indicate the last mm input.
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_features)
        start_mm_idx = len(mm_features) + start_mm_idx

    curr_mm_idx = start_mm_idx
    while mm_features and curr_mm_idx < len(mm_features):
        mm_feature = mm_features[curr_mm_idx]
        assert mm_feature.identifier is not None
        offset = mm_feature.mm_position.offset
        length = mm_feature.mm_position.length
        if end_token_idx > offset:
            if start_token_idx >= offset + length:
                # This block has passed the current mm input.
                curr_mm_idx += 1
                continue

            # The block contains the current mm input. Include its offset
            # relative to the start of the block so prefix-cache keys stay
            # distinct when the same MM item appears at different positions
            # within otherwise-identical placeholder blocks.
            extra_keys.append((mm_feature.identifier, offset - start_token_idx))

            if end_token_idx >= offset + length:
                # If this block contains the end of the current mm input,
                # move to the next mm input as this block may also contain
                # the next mm input.
                curr_mm_idx += 1
            else:
                # Otherwise this block is done with mm inputs.
                break
        else:
            # This block has not reached the current mm input.
            break
    return extra_keys, curr_mm_idx


def _gen_lora_extra_hash_keys(request: Request) -> list[str]:
    """Generate extra keys related to LoRA for block hash computation.

    Args:
        request: The request object.

    Returns:
        Return LoRA name of the request if it is a LoRA request. Return empty
        list otherwise.
    """
    if not request.lora_request:
        return []
    return [request.lora_request.lora_name]


def _gen_prompt_embeds_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int
) -> list[bytes]:
    """Generate extra keys related to prompt embeds for block hash computation.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.

    Returns:
        Return a stable hash of the block prompt embeddings if prompt embeds
        are present. Return empty list otherwise.
    """
    if request.prompt_embeds is None:
        return []
    block_range = (start_token_idx, end_token_idx)
    embeds_hash = request._prompt_embeds_per_block_hashes.get(block_range)
    if embeds_hash is None:
        block_prompt_embeds = request.prompt_embeds[start_token_idx:end_token_idx]
        # Hash prompt embeds once per block and cache on request
        embeds_hash = hashlib.sha256(tensor_data(block_prompt_embeds)).digest()
        request._prompt_embeds_per_block_hashes[block_range] = embeds_hash
    return [embeds_hash]


def generate_block_hash_extra_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[tuple[Any, ...] | None, int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs, request specific metadata (e.g., LoRA names), and
    hashed data from prompt embeddings.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    mm_extra_keys: list[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx
    )
    lora_extra_keys: list[str] = _gen_lora_extra_hash_keys(request)
    cache_salt_keys: list[str] = (
        [request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []
    )
    prompt_embeds_keys = _gen_prompt_embeds_extra_hash_keys(
        request, start_token_idx, end_token_idx
    )

    extra_keys: list[Any] = (
        lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
    )

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx


def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.
    Args:
        hash_function: The hash function used to compute block hash.
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        curr_block_token_ids: A list of token ids in the current
            block. The current block is assumed to be full.
        extra_keys: Extra keys for the block.
    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.
    """
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )


def get_request_block_hasher(
    block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
) -> Callable[[Request], list[BlockHash]]:
    """
    Returns a function which computes the list of un-computed block hashes
    of a request."""

    def request_block_hasher(request: Request) -> list[BlockHash]:
        start_token_idx = len(request.block_hashes) * block_size
        num_tokens = request.num_tokens

        if start_token_idx + block_size > num_tokens:
            # Early stop when there no new full blocks created.
            return []

        curr_mm_idx = 0
        if start_token_idx > 0:
            # Set curr_mm_idx = -1 to indicate the last mm input.
            # Note that since we reach to this branch only when the block is
            # completed with generated tokens, we only need to consider the
            # last mm input.
            curr_mm_idx = -1

        prev_block_hash_value = (
            request.block_hashes[-1] if request.block_hashes else None
        )
        new_block_hashes: list[BlockHash] = []
        while True:
            end_token_idx = start_token_idx + block_size
            if end_token_idx > num_tokens:
                # We only hash full blocks
                break

            # MM and LoRA requests need extra keys for block-hash computation.
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start_token_idx, end_token_idx, curr_mm_idx
            )

            # Compute the hash of the current block
            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
            block_hash = hash_block_tokens(
                caching_hash_fn, prev_block_hash_value, block_tokens, extra_keys
            )

            new_block_hashes.append(block_hash)
            start_token_idx += block_size
            prev_block_hash_value = block_hash

        return new_block_hashes

    return request_block_hasher


def _check_enough_kv_cache_memory(
    available_memory: int,
    get_needed_memory: Callable[[], int],
    max_model_len: int,
    estimate_max_model_len: Callable[[int], int],
):
    if available_memory <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine. "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )

    needed_memory = get_needed_memory()

    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        raise ValueError(
            f"To serve at least one request with the models's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
            f"Try increasing `gpu_memory_utilization` or decreasing `max_model_len` "
            f"when initializing the engine. "
            f"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            f"for more details."
        )


def max_memory_usage_bytes(
    vllm_config: VllmConfig, kv_cache_specs: Iterable[KVCacheSpec]
) -> int:
    """
    Get the maximum memory usage in bytes for the given KV cache specs.
    """
    return sum(spec.max_memory_usage_bytes(vllm_config) for spec in kv_cache_specs)


def estimate_max_model_len(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> int:
    """
    Estimates the maximum model length that can fit in the available memory
    using binary search.

    This function temporarily modifies max_model_len during estimation but
    restores the original value before returning, ensuring no side effects.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The estimated maximum model length that can fit in the available memory.
    """
    # Save the original max_model_len to restore after estimation
    original_max_model_len = vllm_config.model_config.max_model_len

    # Define a function to check if a given model length fits in memory
    def fits_in_memory(model_len: int) -> bool:
        # Temporarily modify the max_model_len for this calculation
        vllm_config.model_config.max_model_len = model_len
        # Calculate memory needed for the given model length
        memory_needed = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())
        return memory_needed <= available_memory

    try:
        # Binary search for the maximum model length
        left, right = 1, original_max_model_len

        # If even the smallest model length doesn't fit, return 0
        if not fits_in_memory(left):
            return 0

        # Binary search for the maximum model length that fits
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits_in_memory(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        # Always restore the original max_model_len to avoid side effects
        vllm_config.model_config.max_model_len = original_max_model_len


def check_enough_kv_cache_memory(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.
    """

    # No need to check for available memory if the kv_cache_spec is empty
    if kv_cache_spec:
        _check_enough_kv_cache_memory(
            available_memory,
            lambda: max_memory_usage_bytes(vllm_config, kv_cache_spec.values()),
            vllm_config.model_config.max_model_len,
            lambda am: estimate_max_model_len(vllm_config, kv_cache_spec, am),
        )


def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec], grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]:
    """
    Create KVCacheGroupSpec object for each kv cache group layer.
    The layers in the same group should share the same
    KVCacheSpec.

    Args:
        kv_cache_spec:
            A mapping from each layer name to its corresponding KVCacheSpec.
        grouped_layer_names:
            A list of kv cache groups, where each element is a list of layer
            names that belong to the same group and should share the same
            KVCacheSpec.
    Returns:
        A list of KVCacheGroupSpec objects, one for each group.
    """
    kv_cache_groups = []
    for layer_names_one_group in grouped_layer_names:
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        merged_layer_spec = layer_specs[0].merge(layer_specs)
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
        )
    return kv_cache_groups


def is_kv_cache_spec_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same KV cache spec.
    Note that we regard FullAttentionSpec with and without sliding window as
    the same type.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    if not kv_cache_spec:
        # Encoder-only models do not have KV cache, kv_cache_type can be
        # regarded as uniform.
        return True
    try:
        kv_cache_spec_values = list(kv_cache_spec.values())
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        return False
    return True


def get_max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """
    Get the maximum concurrency for the given KV cache configuration.
    """
    num_layer_per_group = max(
        len(group.layer_names) for group in kv_cache_config.kv_cache_groups
    )
    max_memory_usage_per_request = num_layer_per_group * max_memory_usage_bytes(
        vllm_config, (group.kv_cache_spec for group in kv_cache_config.kv_cache_groups)
    )
    memory_per_block = (
        kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
        * num_layer_per_group
    )
    num_block_per_request = cdiv(max_memory_usage_per_request, memory_per_block)
    max_concurrency = kv_cache_config.num_blocks / num_block_per_request
    return max_concurrency


def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int:
    """
    Override the number of kv cache blocks if `num_gpu_blocks_override` is set.
    """
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = vllm_config.cache_config.num_gpu_blocks_override
        logger.info(
            "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
            num_blocks,
            num_gpu_blocks_override,
        )
        num_blocks = num_gpu_blocks_override

    return num_blocks


def get_num_blocks(
    vllm_config: VllmConfig, num_layers: int, available_memory: int, page_size: int
) -> int:
    """
    Get the number of kv cache blocks.

    Args:
        vllm_config: The global VllmConfig
        num_layers: The number of layers
        available_memory: Memory available for KV cache in bytes.
        page_size: The page size of the KV cache.
    """
    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    return num_blocks


def get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """
    Get the page size of the KV cache.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()


def _get_kv_cache_groups_uniform_spec(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with the same KV cache
    spec for all layers.

    Args:
        kv_cache_specs: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    return create_kv_cache_group_specs(kv_cache_specs, [list(kv_cache_specs.keys())])


def _get_kv_cache_groups_uniform_type(
    spec: UniformTypeKVCacheSpecs,
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with one type of KV cache
    but different hidden sizes. All layers are merged into one group.

    Args:
        spec: The UniformTypeKVCacheSpecs of the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    return [KVCacheGroupSpec(list(spec.kv_cache_specs.keys()), spec)]


def is_kv_cache_page_size_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same page size.
    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        True if all layers have the same page size, False otherwise.
    """

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    return len(page_sizes) == 1


def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Raise
    NotImplementedError if failed to unify the page size.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size != 0:
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting block_size."
                )
            ratio = max_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec


def is_kv_cache_type_attention_free(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    # kv_cache_spec is an empty dict for attention free models
    return not kv_cache_spec


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache groups for hybrid models with multiple
    attention types but still with a uniform page size (physical memory per
    block per layer) for all layers.

    Detailed explanation about kv cache management of hybrid models:
    The layers in the models are repeated with some patterns, e.g., a model
    with 10 full attention layers and 20 sliding window attention layers can be
    regarded as repeating the pattern (1 * full, 2 * sw) 10 times.
    The KVCacheManager allocates different block tables for each of the 3 layers
    in the pattern, and repeats each of them 10 times to generate the
    block_table for the 30 layers in the model.
    Therefore, we can group the layers in the model into 3 kv_cache_groups, each
    of which contains 10 layers in the model.
    The KVCacheManager allocates the block_table for each group based on its
    kv_cache spec, and the model runner applies the block table to each layer
    in the group.
    For example:
    1. A model only uses full attention. The pattern is
    (num_hidden_layers * full), so there is only one group and the block table
    is shared by all layers. It is already handled by
    `_get_kv_cache_config_uniform_type`.
    2. A model with 10 full attention layers and 20 sliding window
    attention layers. There are 3 layers in the pattern (1 * full, 2 * sw), so
    there are 3 kv_cache_groups, each of which represents 10 layers.

    To simplify the implementation, we make the following assumptions:
    1. Physical memory per block: Must be the same across all KV cache groups.
    Breaking this assumption is non-trivial due to memory fragmentation concerns
    when allocating blocks of different sizes.
    2. Tokens per block (block_size): Currently, we directly use
    `CacheConfig.block_size` for all layers. It can be extended to vary by KV
    cache group, but within each KV cache group, all layers must share the same
    block size.
    3. Physical memory per token per layer: This property is decided by model
    config. Currently we only support models that have the same physical memory
    per token per layer for all layers. Can be relaxed with a simple extension,
    but still need to keep physical memory per block the same for all groups.
    4. Number of layers per group: Currently assumed the same for all layers.
    Can be relaxed with a simple extension, but still need to keep physical
    memory per block the same for all groups.
    5. Attention type within groups: All layers in a group must share the same
    attention type. One exception is that, when
    `--disable-hybrid-kv-cache-manager` is true, the single group for full
    attention layers may also include attention layers using sliding window or
    LLaMA 4 local attention. See `unify_hybrid_kv_cache_specs` for more details.
    6. Support for multiple attention types: The design for most components is
    general to an arbitrary number of attention types. But
    `find_longest_cache_hit` only supports one attention type or two
    types of full-attention plus exactly one another type. The general
    implementation of this function is feasible but we don't know how to
    implement it cleanly yet.

    As we assume tokens per block, physical memory per token per layer, and
    number of layers per group are the same now, we can ensure that physical
    memory per block is the same for all groups.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
    Returns:
        The generated KVCacheGroupSpecs
    """
    # Group all layers by kv_cache_spec.
    # E.g., 2 full attention layers and 3 sliding window attention layers,
    # -> (full.0, full.1), (sw.0, sw.1, sw.2).
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    # Split each group into smaller groups, to make the number of layers in each
    # group identical. Add padding to the last group of each type if necessary.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2)
    # split to 3 groups with 2 layers each:
    # (full.0, full.1), (sw.0, sw.2), (sw.1, padding).
    # FIXME(Chen): At the moment of writing this code (2025-06-02), all
    # open-source hybrid model follows a n:1 pattern between different attention
    # types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and
    # full), so we can use the "1" in the n:1 pattern as the group size, which
    # is the minimum number of layers among all attention types. Need a better
    # strategy if we want to support more complex patterns (e.g., 20 full + 30
    # sw, where the group size should be 10).
    min_num_layers = min([len(layers) for layers in same_type_layers.values()])
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in same_type_layers.values()])
    if max_num_layers < min_num_layers * 1.5:
        # If the number of layers is not much larger than the minimum number of
        # layers, use the maximum number of layers as the group size to avoid
        # too many padding layers. A typical example is gpt-oss-20b + eagle,
        # with 12 sw + 13 full. We pad it to (13 sw, 13 full) instead of
        # (12 sw, 24 full). 1.5 is a heuristic to avoid too many padding
        # layers while accommodating speculative decoding drafters that add
        # extra layers to one attention type.
        group_size = max_num_layers
    grouped_layers = []
    for layers in same_type_layers.values():
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",  # noqa
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        # In PP case, say if we have
        # - stage 0: full.0, sw.0, sw.1
        # - stage 1: full.1, sw.2, sw.3
        # We should have 3 groups: (full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)
        # It can't be (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3) because
        # the 3 groups in stage 0 will be (full.0), (sw.0, sw.1), (empty group)
        # and it will be padded to (full.0, padding), (sw.0, sw.1),
        # (padding, padding) to ensure the number of layers in each group is
        # the same and will cause memory waste.
        # To avoid this, we assign layers[i::num_groups] to the i-th group
        # instead of layers[i * group_size: (i + 1) * group_size]
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """
    Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig
    """
    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )

    # Determine how model runners should initialize the KV cache tensors.
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # Special case: all layers have the same type of KV cache but with
        # different hidden size. Allocate different amount of memory for each
        # layer based on its hidden size.
        num_blocks = (
            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
        )
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        kv_cache_tensors = [
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
            for layer_name in kv_cache_groups[0].layer_names
        ]
    else:
        # General case:
        # We will have group_size memory pools, each is shared by one layer from
        # each group. As layers of different groups have different block table,
        # they will use different parts of the shared Tensor.
        # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
        # (sw.1, padding) will be: (group_size = 2)
        # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
        # full.1, sw.2: share another Tensor with size=available_memory//2
        group_size = max(len(group.layer_names) for group in kv_cache_groups)

        page_size = get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """
    This function tries to convert the KV cache specs to one type if the model
    is a hybrid model with multiple type of KV cache. It will convert all
    SlidingWindowSpec to FullAttentionSpec if both types are present.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model
    """

    if is_kv_cache_spec_uniform(
        kv_cache_spec
    ) or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec):
        return

    logger.warning(
        "Hybrid KV cache manager is disabled for this hybrid model, "
        "This means we do not enable any optimizations for saving KV cache "
        "memory (e.g., dropping the KV cache outside the sliding window). "
        "The compute of layers like sliding window is still saved."
    )

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in kv_cache_spec.values()
    )
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values()
    )
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec) for spec in kv_cache_spec.values()
    )
    if has_full_attention and (has_sliding_window or has_chunked_local_attention):
        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, SlidingWindowSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    sliding_window=spec.sliding_window,
                    page_size_padded=spec.page_size_padded,
                )
            elif isinstance(spec, ChunkedLocalAttentionSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    attention_chunk_size=spec.attention_chunk_size,
                    page_size_padded=spec.page_size_padded,
                )

    if not (
        is_kv_cache_spec_uniform(kv_cache_spec)
        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)
    ):
        raise ValueError(
            "Hybrid KV cache manager is disabled but failed to "
            "convert the KV cache specs to one unified type."
        )


def get_kv_cache_groups(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec]:
    """
    Split the layers in the model into groups with the same KV cache spec.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroups
    """
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    if is_kv_cache_type_attention_free(kv_cache_spec):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        return []

    if is_kv_cache_spec_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        return _get_kv_cache_groups_uniform_type(uniform_spec)

    # As KVCacheManager can only allocate memory of one size, we need to unify
    # the page size of the layers. For cases cannot be unified, this function
    # will raise an error.
    kv_cache_spec = unify_kv_cache_spec_page_size(kv_cache_spec)
    # Model contains multiple attention types, but KV cache of all layers
    # have the same physical memory per block per layer. Split the layers
    # into groups with the same number of layers, and thus same total page
    # size.
    return _get_kv_cache_groups_uniform_page_size(kv_cache_spec)


def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    """
    Generate the KV cache configuration for the scheduler.
    """
    assert all(
        [cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs]
    )
    # All workers have the same kv_cache_config except layer names, so use
    # an arbitrary one to initialize the scheduler.
    cfg = copy.deepcopy(kv_cache_configs[0])
    for group in cfg.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # so use an arbitrary one to initialize the scheduler.
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values())
            )
    return cfg


def _report_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """
    Log resolved KV cache configuration.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_config: The resolved KV cache configuration
    """
    min_block_size = min(
        [group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups]
    )

    # Log the KV cache size and maximum concurrency.
    num_tokens = (
        kv_cache_config.num_blocks
        // len(kv_cache_config.kv_cache_groups)
        * min_block_size
    )
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    if pcp_size * dcp_size > 1:
        num_tokens *= pcp_size * dcp_size
        logger.info(
            "Multiplying the GPU KV cache size by the cp_world_size %d "
            "(pcp_world_size %d * dcp_world_size %d).",
            pcp_size * dcp_size,
            pcp_size,
            dcp_size,
        )
    num_tokens_str = f"{num_tokens:,}"
    logger.info_once("GPU KV cache size: %s tokens", num_tokens_str, scope="local")
    max_model_len_str = f"{vllm_config.model_config.max_model_len:,}"
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    logger.info_once(
        "Maximum concurrency for %s tokens per request: %.2fx",
        max_model_len_str,
        max_concurrency,
        scope="local",
    )


def _max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """
    Calculate maximum memory usage in bytes from KV cache groups.

    This correctly accounts for padding in hybrid models. For example, if a
    model has 8 full attention layers and 9 sliding window layers, they will
    be padded to 9 full + 9 sliding window for uniform group sizes.
    """
    if not kv_cache_groups:
        return 0

    # UniformTypeKVCacheSpecs special case (single group, per-layer specs)
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        return sum(
            spec.max_memory_usage_bytes(vllm_config)
            for spec in per_layer_specs.values()
        )

    # General case: group_size pools, each shared by one layer per group
    # Memory = group_size * page_size * blocks_for_max_len
    group_size = max(len(group.layer_names) for group in kv_cache_groups)
    page_size = get_uniform_page_size(
        [group.kv_cache_spec for group in kv_cache_groups]
    )
    blocks_needed = sum(
        cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), page_size)
        for group in kv_cache_groups
    )

    return group_size * page_size * blocks_needed


def _estimate_max_model_len_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> int:
    """
    Binary search for the maximum model length that fits in available memory.
    Returns 0 if even 1 token doesn't fit.
    """
    original_max = vllm_config.model_config.max_model_len

    def fits(model_len: int) -> bool:
        vllm_config.model_config.max_model_len = model_len
        return (
            _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
            <= available_memory
        )

    try:
        left, right = 1, original_max
        if not fits(left):
            return 0
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        vllm_config.model_config.max_model_len = original_max


def _auto_fit_max_model_len(
    vllm_config: VllmConfig,
    projected_groups_per_worker: list[list[KVCacheGroupSpec]],
    available_memory: list[int],
) -> None:
    """
    When max_model_len is set to -1, this function estimates the largest
    context length that can be supported with the available GPU memory.
    It uses binary search to find the maximum length that fits across all
    workers.

    Args:
        vllm_config: The global VllmConfig (will be modified in-place)
        projected_groups_per_worker: KV cache groups projected to each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.
    """
    original_max = vllm_config.model_config.max_model_len

    if all(not groups for groups in projected_groups_per_worker):
        # All workers have empty specs (attention-free model)
        logger.info_once(
            "Auto-fit max_model_len: attention-free model, "
            "using derived max_model_len=%d",
            original_max,
            scope="local",
        )
        return

    # Find the max_model_len that fits across all workers.
    auto_fit_max = original_max
    limiting_worker_mem = available_memory[0]
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        worker_max = _estimate_max_model_len_from_groups(vllm_config, groups, avail_mem)
        if worker_max < auto_fit_max:
            auto_fit_max = worker_max
            limiting_worker_mem = avail_mem

    if auto_fit_max <= 0:
        raise ValueError(
            "Cannot auto-fit max_model_len: not enough GPU memory available "
            "to serve even a single token. Try increasing `gpu_memory_utilization`."
        )

    if auto_fit_max >= original_max:
        # The model's full context length fits in memory
        logger.info_once(
            "Auto-fit max_model_len: full model context length %d fits in "
            "available GPU memory",
            original_max,
            scope="local",
        )
    else:
        # Need to reduce max_model_len to fit in memory
        vllm_config.model_config.max_model_len = auto_fit_max
        logger.info_once(
            "Auto-fit max_model_len: reduced from %d to %d to fit in "
            "available GPU memory (%s GiB available for KV cache)",
            original_max,
            auto_fit_max,
            format_gib(limiting_worker_mem),
            scope="local",
        )


def _project_kv_cache_groups_to_worker(
    global_kv_cache_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Projects global KV cache groups onto a single worker's assigned layers.

    In pipeline parallelism, each worker only owns a subset of layers. This
    function filters the global groups to include only layers present on the
    given worker, adjusting UniformTypeKVCacheSpecs accordingly.

    Args:
        global_kv_cache_groups: The global KV cache groups for the whole model.
        worker_spec: The KV cache spec of each layer on this worker.

    Returns:
        The projected KV cache groups containing only this worker's layers.
    """
    projected_groups: list[KVCacheGroupSpec] = []
    for group in global_kv_cache_groups:
        worker_layer_names = [
            layer_name for layer_name in group.layer_names if layer_name in worker_spec
        ]
        group_spec = group.kv_cache_spec
        if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                },
            )
        projected_groups.append(KVCacheGroupSpec(worker_layer_names, group_spec))
    return projected_groups


def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """
    Generates the KV cache configurations for a model.
    Since we use a shared centralized controller for all workers, we need the
    `kv_cache_config` to be consistent across all workers to make sure
    the KV cache allocation can be applied to all workers. However, different
    workers may have different memory available, and different type of layers
    (when pipeline parallel is enabled). To handle the difference between
    workers, the current implementation is:
    1. Merge the KV cache specs of all workers to get the KVCacheSpecs for
       the whole model.
    2. Generate the KV cache groups based on the layer ratio of the whole model.
       This also handles spec unification for hybrid models.
    3. Handle auto-fit max_model_len and memory checks using per-worker
       projected groups to account for PP sharding.
    4. Generate the KV cache configs for each worker based on the KV cache
       grouping strategy. (This is reasonable because the layer ratio of
       different PP stages are similar.)
    5. Change the num_blocks of each worker to the smallest among all workers
       and shrink tensor sizes proportionally to avoid allocating unused memory.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    Returns:
        The generated KVCacheConfigs for each worker.
    """

    # Merge the KV cache specs of all workers. Different PP stages may have
    # different layer names, and different TP ranks of the same PP stage should
    # have the same KV cache spec.
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    # Get global KV cache groups. This also handles spec unification for
    # hybrid models when disable_hybrid_kv_cache_manager is enabled.
    # After this call, merged_kv_cache_specs may be modified in-place.
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # If original_max_model_len was -1, automatically
    # determine the maximum model length that fits in available GPU memory.
    # We use per-worker projected groups to account for PP sharding.
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
        for worker_spec in kv_cache_specs
    ]

    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(
            vllm_config, projected_groups_per_worker, available_memory
        )

    # Check if the available memory is enough per worker.
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )

    kv_cache_configs: list[KVCacheConfig] = []
    for projected_groups, kv_cache_spec_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, kv_cache_specs, available_memory
    ):
        assert sum(len(group.layer_names) for group in projected_groups) == len(
            kv_cache_spec_one_worker
        ), "Some layers are not assigned to any group."
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, projected_groups, available_memory_one_worker
            )
        )

    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        # Shrink tensor size proportionally
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks

        if len(kv_cache_config.kv_cache_groups) > 0:
            _report_kv_cache_config(vllm_config, kv_cache_config)

    return kv_cache_configs


class BlockHashListWithBlockSize:
    """
    Convert block-hash granularity from `hash_block_size` to `target_block_size`.
    Used when KV cache groups have different block sizes: `hash_block_size`
    is the size used to compute the original `block_hashes`; `target_block_size`
    is the group's actual block size.

    Currently, only scaling up by an integer factor is supported (i.e.,
    `target_block_size` is a multiple of `hash_block_size`). Conversion is
    performed lazily on access for efficiency, by concatenating consecutive
    hashes at `hash_block_size` to form each hash at `target_block_size`.

    Example (`hash_block_size` = 16, `target_block_size` = 32):
    concatenating two 16-size hashes yields one 32-size hash:

    Block hashes with block_size 16:
    | Token Range | 0-15 | 16-31 | 32-47 | 48-63 |
    |-------------|------|-------|-------|-------|
    | Hash        | A    | B     | C     | D     |

    Block hashes with block_size 32:
    | Token Range | 0-31 | 32-63 |
    |-------------|------|-------|
    | Hash        | AB   | CD    |

    Args:
        block_hashes: Block hashes to convert, computed at `hash_block_size`.
        hash_block_size: Block size at which `block_hashes` were computed.
        target_block_size: Desired block size; must be a multiple of `hash_block_size`.
    """

    def __init__(
        self,
        block_hashes: list[BlockHash],
        hash_block_size: int,
        target_block_size: int,
    ):
        self.block_hashes = block_hashes
        assert target_block_size % hash_block_size == 0
        self.scale_factor = target_block_size // hash_block_size

    def __len__(self) -> int:
        return len(self.block_hashes) // self.scale_factor

    @overload
    def __getitem__(self, idx: int) -> BlockHash: ...

    @overload
    def __getitem__(self, idx: slice) -> list[BlockHash]: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_value_at(idx)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._get_value_at(i) for i in range(start, stop, step)]

        raise TypeError(f"Invalid index type: {type(idx)!r}")

    def __iter__(self) -> Iterator[BlockHash]:
        for i in range(len(self)):
            yield self._get_value_at(i)

    def _get_value_at(self, idx: int) -> BlockHash:
        base = idx * self.scale_factor
        end = base + self.scale_factor
        merged_hash: bytes = self.block_hashes[base]
        for i in range(base + 1, end):
            merged_hash += self.block_hashes[i]
        return BlockHash(merged_hash)


BlockHashList = list[BlockHash] | BlockHashListWithBlockSize

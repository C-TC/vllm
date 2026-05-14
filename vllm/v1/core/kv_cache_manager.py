# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request
from vllm.v1.wires_engine_telemetry import (
    begin_request_alloc,
    decide_cache_source,
    end_request_alloc,
)
from vllm.v1.wires_engine_telemetry import (
    emit_segment_touch as _wires_emit_segment_touch,
)
from vllm.v1.wires_telemetry import is_enabled as _wires_telemetry_enabled

logger = init_logger(__name__)

_WORKFLOW_PREFIX_PREPARE_MAX_OUTSTANDING_ENV = "WORKFLOW_PREFIX_PREPARE_MAX_OUTSTANDING"
_DEFAULT_WORKFLOW_PREFIX_LEASE_MAX_OUTSTANDING = 128


def _workflow_prefix_lease_max_outstanding() -> int:
    raw = os.getenv(_WORKFLOW_PREFIX_PREPARE_MAX_OUTSTANDING_ENV)
    if raw is None:
        return _DEFAULT_WORKFLOW_PREFIX_LEASE_MAX_OUTSTANDING
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_WORKFLOW_PREFIX_LEASE_MAX_OUTSTANDING
    return max(1, value)


@dataclass
class _PreparedPrefixLeaseRef:
    """Engine-private retention reference for a PreparedPrefix.

    The ref owns private cached block references while it is live. Only
    redacted status events derived from this object may cross back to the API
    layer; block ids, cache hashes, and KV handles stay inside EngineCore.
    """

    lease_id: str
    prefix_id: str
    action_id: str
    blocks: tuple[KVCacheBlock, ...]
    prefix_token_count: int
    full_block_count: int
    ttl_ms: int
    state: str
    expires_at_monotonic_s: float


def _workflow_lease_result(
    lease_status: str,
    lease_reason: str,
    *,
    prefix_token_count: int,
    full_block_count: int,
    ttl_ms: int,
    prepared_prefix_ref_status: str | None = None,
    lease_event_status: str | None = None,
    lease_event_reason: str | None = None,
    lease_id_present: bool = False,
    prefix_id_present: bool = False,
) -> dict[str, int | str | bool]:
    ref_status = prepared_prefix_ref_status or lease_status
    event_status = lease_event_status or lease_status
    event_reason = lease_event_reason or lease_reason
    return {
        "lease_status": lease_status,
        "lease_reason": lease_reason,
        "lease_token_count": prefix_token_count,
        "lease_full_block_count": full_block_count,
        "lease_ttl_ms": ttl_ms,
        "prepared_prefix_ref_status": ref_status,
        "lease_event_status": event_status,
        "lease_event_reason": event_reason,
        "lease_id_present": lease_id_present,
        "prefix_id_present": prefix_id_present,
    }


def _prepared_prefix_lease_id(action_id: str) -> str:
    return f"workflow-prepared-prefix-lease:{action_id}"


def _prepared_prefix_id(action_id: str) -> str:
    return f"workflow-prepared-prefix:{action_id}"


@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but
    will be broken if we want to give different block_size to different
    kv_cache_groups in the future.

    Each single type KVCacheBlocks could be represented as:
    - list[KVCacheBlock] for more than one KVCacheBlock
    - an empty tuple for requests without KVCacheBlock
      (a precomputed KVCacheBlocks is in KVCacheManager to avoid GC overhead)
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        # Skip padding blocks.
        return [
            [
                block.block_id
                for block in group
                if block.block_hash is None and not block.is_null
            ]
            for group in self.blocks
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """
        Creates a new KVCacheBlocks instance with no blocks.
        """
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config
        self._workflow_prepared_prefix_leases: dict[str, _PreparedPrefixLeaseRef] = {}

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens

    def can_fit_full_sequence(
        self,
        request: Request,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_external_computed_tokens: int = 0,
        num_encoder_tokens: int = 0,
    ) -> bool:
        """Check if the KV cache has enough free blocks to hold the full
        sequence, accounting for prefix cache hits and sliding window.

        This is used as an admission gate to prevent over-admitting requests
        when chunked prefill would otherwise only check the first chunk.
        """
        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        full_num_tokens = min(request.num_tokens, self.max_model_len)

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=total_computed_tokens,
            num_tokens_main_model=full_num_tokens,
        )

        return num_blocks_to_allocate <= self.block_pool.get_num_free_blocks()

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens, grouped as a tuple by kv cache groups.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            num_external_computed_tokens: The number of tokens that their
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.

        Blocks layout:
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | Prefix-cached tokens from either vLLM   |
        | or connector. Can be safely removed if  |
        | they are outside sliding window.        |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    | not cached by |
                                  | vLLM, but     |
        | ref_cnt  | ref_cnt not  | cached by     |
        | increased| increased yet| connector     |
        ----------------------------------------------------------------------
        ```

        Abbrivations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, cached by the connector
        new       = num_new_tokens, including unverified draft tokens
        lookahead = num_lookahead_tokens
        ```

        NOTE: for new tokens which include both verified and unverified draft
        tokens, we only cache the verified tokens (by capping the number at
        `request.num_tokens`).

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp + ext_comp`):
            - Free unnecessary blocks (e.g. outside sliding window)
            - Allocate new blocks for `ext_comp` tokens inside
              sliding window
        - Allocate new blocks for tokens to be computed (`new + lookahead`)

        Returns:
            A list of new allocated blocks.
        """
        # When loading KV data asynchronously, we may have zero new tokens to
        # compute while still allocating slots for externally computed tokens.
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens,
            self.max_model_len,
        )

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.coordinator.remove_skipped_blocks(
            request.request_id, total_computed_tokens
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks.
            # M25 prewarm telemetry (paper §3.5 + CODE_MISMATCH_NOTES.md
            # M25): a segment_prepare dispatch that fails here is the
            # capacity-bound "decline" signal. Bump once per declined
            # dispatch (NOT once per missing block) so the counter
            # tracks dispatch outcomes, matching admitted's per-
            # dispatch semantics. Detection: hidden prewarm requests
            # carry a non-empty ``request.segment_id`` (set by
            # ``submit_workflow_segment_prewarm`` via
            # ``SamplingParams.extra_args["workflow_segment_id"]``);
            # ordinary user-facing requests have segment_id=None and
            # so do not count.
            prewarm_segment_id = getattr(request, "segment_id", None)
            if isinstance(prewarm_segment_id, str) and prewarm_segment_id:
                self.block_pool.free_block_queue.prewarm_declined_count += 1
            return None

        # T-45.4 (docs/v2/45 §3.1): mark this request as the current
        # allocate-side request so eviction telemetry on the popleft
        # path attributes ``evicts_caused`` back to it. Cleared in the
        # ``finally`` block below so an exception during allocate does
        # not leak the slot to the next request.
        begin_request_alloc(request)
        try:
            if (
                new_computed_block_list is not self.empty_kv_cache_blocks.blocks
                or num_external_computed_tokens > 0
            ):
                # Append the new computed blocks to the request blocks until now
                # to avoid the case where the new blocks cannot be allocated.
                self.coordinator.allocate_new_computed_blocks(
                    request_id=request.request_id,
                    new_computed_blocks=new_computed_block_list,
                    num_local_computed_tokens=num_local_computed_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                )

            new_blocks = self.coordinator.allocate_new_blocks(
                request.request_id,
                num_tokens_need_slot,
                num_tokens_main_model,
                num_encoder_tokens,
            )
        finally:
            end_request_alloc()

        # WIRES Phase E2 step 2: propagate the request's lifecycle hint
        # to the freshly-allocated blocks so subsequent eviction (via
        # FreeKVCacheBlockQueue.popleft_n's 3-priority traversal) honors
        # it. Default "may" reproduces legacy LRU behavior. See docs/v2/31 §E2.
        #
        # M13 (paper/CODE_MISMATCH_NOTES.md): per-segment hints from
        # ``segment_lifecycle_hints`` override the request-level fallback
        # for blocks whose ``_segment_id`` matches a known scope_key.
        # The legacy single ``lifecycle_hint`` field still applies to any
        # newly-allocated block NOT covered by a per-segment entry. This
        # lets the runner ship per-segment intent for not-yet-prepared
        # segments via the chat completion while baseline lanes (which
        # only set the legacy field) keep working unchanged.
        request_hint = getattr(request, "lifecycle_hint", "may")
        seg_hints = getattr(request, "segment_lifecycle_hints", None)
        if request_hint != "may":
            for group_blocks in new_blocks:
                for blk in group_blocks:
                    blk.lifecycle_hint = request_hint
                    # source_class Option D, Site 1 (request-level branch).
                    # Stamp "structured" at allocate-time when the request
                    # promotes a block to must via the lifecycle_hint
                    # plumbing. The block then enters _stamp_must_promotion
                    # at append_n with the right source_class so it gets
                    # the 300s structured TTL instead of the 60s
                    # unstructured TTL, which is the difference behind
                    # paper §3's "structured blocks survive much longer"
                    # narrative under the prewarm path that previously
                    # bypassed the structured-hint updater.
                    if request_hint == "must":
                        blk.source_class = "structured"
        if isinstance(seg_hints, dict) and seg_hints:
            # Per-segment override: for each newly-allocated block whose
            # ``_segment_ids`` overlap a scope_key, apply the per-
            # segment hint. Blocks without a tagged segment fall back
            # to whatever the request-level pass set above (legacy
            # hint or the default "may").
            #
            # M18: a block may carry multiple overlapping segment ids
            # (doc 32 §2.3 "min wins"). When several of those tags
            # match scope_keys, pick the MIN-priority hint
            # (no < may < must) to honor the most-conservative
            # caller's intent; this keeps the per-segment override
            # semantics consistent with the runner-driven hint flips
            # that go through ``update_block_hints`` and also fold
            # multiple overlapping segments into one effective hint.
            _PRIORITY = {"no": 0, "may": 1, "must": 2}
            for group_blocks in new_blocks:
                for blk in group_blocks:
                    blk_seg_ids = getattr(blk, "_segment_ids", ())
                    chosen: str | None = None
                    for seg in blk_seg_ids:
                        h = seg_hints.get(seg)
                        if h is None:
                            continue
                        if chosen is None or _PRIORITY[h] < _PRIORITY[chosen]:
                            chosen = h
                    if chosen is not None:
                        blk.lifecycle_hint = chosen
                        # source_class Option D, Site 1 (per-segment
                        # branch). Same allocate-time stamp as the
                        # request-level path above. After the M18
                        # min-wins recompute the per-segment hint may
                        # resolve to "must" (e.g., the block belongs
                        # to a structured scope_key); stamp
                        # source_class so the eventual append_n
                        # promotion lands the block in the must pool
                        # with the 300s structured TTL.
                        if chosen == "must":
                            blk.source_class = "structured"

        # WIRES Phase E5: when the request carries a workflow_segment_id
        # (set by submit_workflow_segment_prewarm via SamplingParams.extra_args
        # ["workflow_segment_id"]), tag every freshly-allocated block so
        # segment_telemetry can attribute cache_hit_count / evict_count_by_hint
        # back to the originating segment. Skipped for ordinary user-facing
        # requests where segment_id is None.
        request_segment_id = getattr(request, "segment_id", None)
        if isinstance(request_segment_id, str) and request_segment_id:
            try:
                from vllm.entrypoints.openai.chat_completion.segment_actions import (
                    tag_blocks_with_segment_id,
                )
            except ImportError:
                # Defensive: chat_completion module is optional in some
                # build configs; segment tagging is purely opt-in.
                tag_blocks_with_segment_id = None
            if tag_blocks_with_segment_id is not None:
                for group_blocks in new_blocks:
                    tag_blocks_with_segment_id(list(group_blocks), request_segment_id)
            # M25 prewarm telemetry (paper §3.5 + CODE_MISMATCH_NOTES.md
            # M25): mark every freshly-admitted block so subsequent
            # touch / eviction can attribute consumed vs evicted-before-
            # use. The flag's lifecycle is admission to first-touch
            # (BlockPool.touch clears it on the first non-prewarm hit)
            # or admission to eviction (popleft / popleft_n /
            # _sweep_ttl_must clear it and bump evicted_before_use_count).
            # Bumped here exactly once per admitted dispatch (NOT per
            # block) so the counter tracks dispatch outcomes; pairing
            # admitted vs declined gives the prewarm admission rate
            # without a per-block weighting bias.
            admitted_any_block = False
            for group_blocks in new_blocks:
                for blk in group_blocks:
                    blk.was_prewarmed = True
                    admitted_any_block = True
            if admitted_any_block:
                self.block_pool.free_block_queue.prewarm_admitted_count += 1

        # T-45.4 (docs/v2/45 §3.1): per-request E1 tallies. Counts the
        # freshly-allocated and prefix-cache-hit blocks across all
        # KV-cache groups for this allocate. ``new_computed_block_list``
        # is the prefix-hit set (one list per group); ``new_blocks`` is
        # the freshly-allocated set. Assignment is one int add per
        # group, no per-block work.
        try:
            new_alloc_count = sum(len(group) for group in new_blocks)
            cache_hit_count = sum(len(group) for group in new_computed_block_list)
            request.num_blocks_allocated_total += new_alloc_count
            request.num_blocks_cache_hit_total += cache_hit_count
        except AttributeError:  # pragma: no cover - defensive (legacy req types)
            pass

        # T-45.7 (docs/v2/45 §3.4): emit one E4 row per segment that
        # this allocate touched. Walks both the cache-hit set
        # (``new_computed_block_list``) and the freshly-allocated set
        # (``new_blocks``); groups blocks by their tagged
        # ``_segment_id`` (set by segment_prepare's
        # tag_blocks_with_segment_id). Non-WIRES allocates touch zero
        # tagged blocks, so this loop short-circuits to nothing.
        if _wires_telemetry_enabled():
            self._wires_emit_e4_segment_touches(
                request,
                new_computed_block_list,
                new_blocks,
            )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    def _wires_emit_e4_segment_touches(
        self,
        request: Request,
        new_computed_block_list,
        new_blocks,
    ) -> None:
        """Emit E4 ``segment_touches`` rows for this allocate
        (T-45.7, docs/v2/45 §3.4).

        Walks every block this allocate touched (cache-hit + freshly
        allocated), groups by ``_segment_id`` (the scope_key), and
        emits one row per distinct segment with the dominant
        ``cache_source`` over its blocks. Cold path relative to the
        decode loop; only fires when telemetry is enabled (caller
        already gated).
        """
        # Group all touched blocks by their segment id. Ungrouped
        # blocks (segment_id is None) are not part of any tracked
        # segment and therefore not emitted; they show up in the E1
        # ``blocks_*`` totals already.
        #
        # M18: a block may carry multiple overlapping segment ids when
        # segment boundaries don't align with block boundaries (doc 32
        # §2.3 "min wins"). Iterate ``_segment_ids`` so each
        # overlapping segment's E4 row sees the touch; single-tag
        # blocks (the common case) cost one tuple lookup over the
        # legacy single-id read.
        by_seg: dict[str, list] = {}
        for group in new_computed_block_list:
            for blk in group:
                seg_ids = getattr(blk, "_segment_ids", ())
                for seg_id in seg_ids:
                    by_seg.setdefault(seg_id, []).append(blk)
        for group in new_blocks:
            for blk in group:
                seg_ids = getattr(blk, "_segment_ids", ())
                for seg_id in seg_ids:
                    by_seg.setdefault(seg_id, []).append(blk)
        if not by_seg:
            return
        ts_now = time.time()
        req_id = getattr(request, "request_id", None)
        # ``segment_index`` is the position of this segment in a
        # request's scope-key sequence. We don't carry the runner's
        # index here; emit 0 for now and let the offline aggregator
        # join with the runner-side S3 stream which DOES carry
        # ordering. Spec §3.4 lists it as a field; populating it
        # honestly would require runner ↔ engine coupling we
        # explicitly designed out.
        for idx, (scope_key, blocks_for_seg) in enumerate(by_seg.items()):
            cache_source = decide_cache_source(blocks_for_seg, req_id, now_epoch=ts_now)
            # Pick the modal hint + source_class across the segment's
            # blocks (single block = trivial). Cheap counter scan.
            hint_counts: dict[str, int] = {}
            sclass_counts: dict[str, int] = {}
            for blk in blocks_for_seg:
                hint_counts[blk.lifecycle_hint] = (
                    hint_counts.get(blk.lifecycle_hint, 0) + 1
                )
                sclass_counts[blk.source_class] = (
                    sclass_counts.get(blk.source_class, 0) + 1
                )
            modal_hint = max(hint_counts, key=hint_counts.get)
            modal_sclass = max(sclass_counts, key=sclass_counts.get)
            _wires_emit_segment_touch(
                vllm_request_id=req_id,
                scope_key=scope_key,
                segment_index=idx,
                block_count=len(blocks_for_seg),
                cache_source=cache_source,
                lifecycle_hint=modal_hint,
                source_class=modal_sclass,
                ts_epoch=ts_now,
            )

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        self.coordinator.free(request.request_id)

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            total_computed_tokens: The total number of computed tokens, including
                local computed tokens and external computed tokens.
        """
        self.coordinator.remove_skipped_blocks(request_id, total_computed_tokens)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        self.block_pool.evict_blocks(block_ids)

    def try_lease_workflow_prepared_prefix(
        self,
        request: Request,
        *,
        action_id: str,
        prefix_token_count: int,
        ttl_ms: int,
    ) -> dict[str, int | str | bool]:
        """Observe cache availability for a PreparedPrefix without retaining it.

        Earlier experimental builds attempted to retain cached blocks here.
        That path is intentionally disabled until vLLM block ownership
        invariants are audited: this helper may discover cached full blocks for
        redacted telemetry, but it must not touch, pin, free, or otherwise own
        blocks. Normal request execution remains unchanged.
        """
        self.release_expired_workflow_prepared_prefix_leases()
        if not self.enable_caching:
            return _workflow_lease_result(
                "lease_unavailable",
                "prefix_cache_disabled",
                prefix_token_count=prefix_token_count,
                full_block_count=0,
                ttl_ms=ttl_ms,
            )
        if (
            len(self._workflow_prepared_prefix_leases)
            >= (_workflow_prefix_lease_max_outstanding())
            and action_id not in self._workflow_prepared_prefix_leases
        ):
            return _workflow_lease_result(
                "lease_failed",
                "lease_capacity_exceeded",
                prefix_token_count=prefix_token_count,
                full_block_count=0,
                ttl_ms=ttl_ms,
            )

        block_size = self.block_pool.hash_block_size
        full_block_count = prefix_token_count // block_size
        if full_block_count <= 0:
            return _workflow_lease_result(
                "lease_unavailable",
                "no_full_blocks",
                prefix_token_count=prefix_token_count,
                full_block_count=0,
                ttl_ms=ttl_ms,
            )
        if len(request.block_hashes) < full_block_count:
            return _workflow_lease_result(
                "lease_unavailable",
                "cache_blocks_missing",
                prefix_token_count=prefix_token_count,
                full_block_count=full_block_count,
                ttl_ms=ttl_ms,
            )

        kv_cache_group_ids = list(range(self.num_kv_cache_groups))
        leased_blocks_by_id: dict[int, KVCacheBlock] = {}
        for block_hash in itertools.islice(request.block_hashes, full_block_count):
            cached_blocks = self.block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            )
            if cached_blocks is None:
                return _workflow_lease_result(
                    "lease_unavailable",
                    "cache_blocks_missing",
                    prefix_token_count=prefix_token_count,
                    full_block_count=full_block_count,
                    ttl_ms=ttl_ms,
                )
            for block in cached_blocks:
                if not block.is_null:
                    leased_blocks_by_id[block.block_id] = block
        if not leased_blocks_by_id:
            return _workflow_lease_result(
                "lease_unavailable",
                "no_full_blocks",
                prefix_token_count=prefix_token_count,
                full_block_count=0,
                ttl_ms=ttl_ms,
            )

        return _workflow_lease_result(
            "lease_unavailable",
            "no_safe_internal_cache_lease_api",
            prefix_token_count=prefix_token_count,
            full_block_count=full_block_count,
            ttl_ms=ttl_ms,
            prepared_prefix_ref_status="lease_unavailable",
            lease_event_status="lease_unavailable",
            lease_event_reason="no_safe_internal_cache_lease_api",
            lease_id_present=False,
            prefix_id_present=False,
        )

    def release_workflow_prepared_prefix_lease(
        self,
        action_id: str,
        *,
        status: str = "lease_released",
    ) -> dict[str, int | str | bool] | None:
        lease = self._workflow_prepared_prefix_leases.pop(action_id, None)
        if lease is None:
            return None
        return _workflow_lease_result(
            "lease_unavailable",
            "no_safe_internal_cache_lease_api",
            prefix_token_count=lease.prefix_token_count,
            full_block_count=lease.full_block_count,
            ttl_ms=max(
                0,
                int((lease.expires_at_monotonic_s - time.monotonic()) * 1000),
            ),
            prepared_prefix_ref_status="lease_unavailable",
            lease_event_status="lease_unavailable",
            lease_event_reason="no_safe_internal_cache_lease_api",
            lease_id_present=False,
            prefix_id_present=False,
        )

    def release_expired_workflow_prepared_prefix_leases(
        self,
    ) -> tuple[dict[str, int | str | bool], ...]:
        """Release expired internal PreparedPrefix leases.

        The returned records are intentionally redacted: they contain only
        lifecycle status and token/block counts, never block ids or cache keys.
        """
        now = time.monotonic()
        expired = [
            action_id
            for action_id, lease in self._workflow_prepared_prefix_leases.items()
            if lease.expires_at_monotonic_s <= now
        ]
        released: list[dict[str, int | str]] = []
        for action_id in expired:
            result = self.release_workflow_prepared_prefix_lease(
                action_id,
                status="lease_expired",
            )
            if result is not None:
                released.append(result)
        return tuple(released)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Calculate the number of common prefix blocks for each kv cache group.

        The function selects a running request and iterates through its blocks.
        A block is considered a common prefix block if ALL requests with
        allocated KV cache share it (i.e., ref_cnt equals the number of entries
        in req_to_blocks).

        NOTE(woosuk): The number of requests with allocated KV cache is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because having allocated KV cache only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must have allocated KV cache, the inverse
        is not necessarily true. There may be requests with allocated KV cache
        that are not scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled requests that do not share the
        common prefix. Currently, this case cannot be easily detected, so the
        function returns 0 in such cases.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        return self.block_pool.take_events()

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.

        Args:
            request: The request to cache the blocks.
            num_computed_tokens: The number of computed tokens, including tokens
                that are already cached and tokens to be cached.
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    def take_new_block_ids(self) -> list[int]:
        """Drain and return new attention block IDs for zeroing."""
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_block_ids())
        return ids

    def new_step_starts(self) -> None:
        """Called when a new step is started."""
        self.coordinator.new_step_starts()

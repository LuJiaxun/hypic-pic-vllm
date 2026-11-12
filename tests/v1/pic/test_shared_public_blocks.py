# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared canonical block sets are counted like cached KV blocks.

Several requests import the same stored segment into one block set.  The set keeps one
holder for the cache plus one per request, and returns to the pool only when the segment
is evicted and no request holds it.

``Block.ref_cnt`` stays owned by ``BlockPool`` (``get_new_blocks`` raises it once and
``free_blocks`` lowers it once), so the sharing count lives in the scheduler.
"""

from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler

SEG_HASH = b"segment-hash-0123456789"
SEGMENT_KEY = (SEG_HASH, 3)


class _Block:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        # get_new_blocks hands out blocks with the pool's single reference.
        self.ref_cnt = 1


class _Pool:
    def __init__(self) -> None:
        self.freed: list[tuple[int, ...]] = []

    def free_blocks(self, blocks) -> None:
        self.freed.append(tuple(int(block.block_id) for block in blocks))


def _scheduler() -> tuple[Scheduler, _Pool]:
    scheduler = object.__new__(Scheduler)
    scheduler._pic_public_shared_blocks = {}
    scheduler._pic_public_shared_holders = {}
    scheduler._pic_public_shared_held = {}
    scheduler.vllm_config = SimpleNamespace(
        pic_config=SimpleNamespace(debug=False)
    )
    pool = _Pool()
    scheduler.kv_cache_manager = SimpleNamespace(block_pool=pool)
    return scheduler, pool


def _register(
    scheduler: Scheduler, request_id: str, blocks: tuple[_Block, ...]
) -> None:
    """Grant a fresh shared set: one holder for the request, one for the cache."""
    scheduler._pic_public_shared_blocks[SEGMENT_KEY] = blocks
    scheduler._pic_public_shared_holders[SEGMENT_KEY] = 2
    scheduler._pic_public_shared_held.setdefault(request_id, []).append(SEGMENT_KEY)


def _join(scheduler: Scheduler, request_id: str) -> None:
    """A second request reuses the existing set."""
    scheduler._pic_public_shared_holders[SEGMENT_KEY] += 1
    scheduler._pic_public_shared_held.setdefault(request_id, []).append(SEGMENT_KEY)


def test_shared_set_survives_until_cache_eviction() -> None:
    scheduler, pool = _scheduler()
    blocks = (_Block(10), _Block(11))
    _register(scheduler, "req-a", blocks)
    _join(scheduler, "req-b")

    scheduler._release_pic_shared_blocks("req-a")
    assert pool.freed == [], "a set still held by req-b must not be freed"
    scheduler._release_pic_shared_blocks("req-b")
    assert pool.freed == [], "the cache still holds one reference"

    scheduler._release_pic_shared_cache_hold(SEG_HASH)
    assert pool.freed == [(10, 11)]
    assert scheduler._pic_public_shared_blocks == {}
    assert scheduler._pic_public_shared_holders == {}


def test_release_is_idempotent_per_request() -> None:
    scheduler, pool = _scheduler()
    blocks = (_Block(20),)
    _register(scheduler, "req-a", blocks)

    scheduler._release_pic_shared_blocks("req-a")
    scheduler._release_pic_shared_blocks("req-a")

    assert scheduler._pic_public_shared_holders[SEGMENT_KEY] == 1
    assert pool.freed == []


def test_eviction_waits_for_the_last_requester() -> None:
    scheduler, pool = _scheduler()
    blocks = (_Block(30), _Block(31))
    _register(scheduler, "req-a", blocks)

    scheduler._release_pic_shared_cache_hold(SEG_HASH)
    assert pool.freed == [], "a request still holds the set"

    scheduler._release_pic_shared_blocks("req-a")
    assert pool.freed == [(30, 31)]


def test_pool_reference_count_is_left_alone() -> None:
    scheduler, _ = _scheduler()
    blocks = (_Block(40),)
    _register(scheduler, "req-a", blocks)

    scheduler._release_pic_shared_blocks("req-a")
    scheduler._release_pic_shared_cache_hold(SEG_HASH)

    # Exactly one reference existed, so it is released exactly once by free_blocks.
    assert blocks[0].ref_cnt == 1

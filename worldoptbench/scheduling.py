"""Request grouping and caching for batched serving (catalog L2 and L4).

Batched inference needs requests of identical shape: the same number of context
frames, the same number of steps, the same frame size. Real traffic is mixed, so
`bucket_requests` groups it; `ResultCache` skips recomputation for repeated
requests. Both are pure Python/numpy — no GPU, no model — and are the pieces the
`worldserve` layer will sit on.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def request_shape(request: dict[str, Any]) -> tuple:
    """The shape key a batch must agree on: (context frames, steps, frame shape)."""
    frames = request["init_video"]
    return (len(frames), len(request["actions"]), tuple(np.asarray(frames[0]).shape))


def bucket_requests(
    requests: Sequence[dict[str, Any]],
    max_batch: int = 8,
    key: Callable[[dict[str, Any]], Any] = request_shape,
) -> list[list[int]]:
    """Groups request indices into batches of identical shape, at most `max_batch`
    each, preserving request order within a bucket and ordering buckets by their
    first appearance (so earlier traffic isn't starved by later, larger buckets).

    Returns lists of indices into `requests`, not copies of the requests, so the
    caller can scatter results back to the right callers.
    """
    if max_batch < 1:
        raise ValueError("max_batch must be >= 1")
    buckets: OrderedDict[Any, list[int]] = OrderedDict()
    for index, request in enumerate(requests):
        buckets.setdefault(key(request), []).append(index)
    batches: list[list[int]] = []
    for indices in buckets.values():
        for start in range(0, len(indices), max_batch):
            batches.append(indices[start : start + max_batch])
    return batches


def request_key(request: dict[str, Any], seed: int = 0) -> str:
    """A stable hash of everything that determines a result: the context frames,
    both action sequences, the horizon and the seed. Two requests that hash equal
    produce equal output on the same model and settings.
    """
    digest = hashlib.sha1()
    digest.update(f"seed={int(seed)};horizon={request['horizon']!r};".encode())
    for frame in request["init_video"]:
        arr = np.ascontiguousarray(frame)
        digest.update(str(arr.shape).encode() + str(arr.dtype).encode() + arr.tobytes())
    for name in ("context_actions", "actions"):
        actions = request.get(name)
        if actions is None:
            digest.update(f"{name}=none;".encode())
        else:
            arr = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
            digest.update(name.encode() + str(arr.shape).encode() + arr.tobytes())
    return digest.hexdigest()


class ResultCache:
    """A small least-recently-used cache. Not thread-safe; wrap with a lock if
    the server shares one across threads.

    The key must include every setting that changes output (the model, the active
    optimization stack, ...), not just the request; `namespace` is prepended for that,
    so the same request under a different stack can't return a stale result.
    """

    def __init__(self, capacity: int = 128, namespace: str = ""):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.namespace = namespace
        self._items: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._items)

    def _full(self, key: str) -> str:
        return f"{self.namespace}|{key}"

    def get(self, key: str) -> Any | None:
        full = self._full(key)
        if full in self._items:
            self._items.move_to_end(full)
            self.hits += 1
            return self._items[full]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        full = self._full(key)
        self._items[full] = value
        self._items.move_to_end(full)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

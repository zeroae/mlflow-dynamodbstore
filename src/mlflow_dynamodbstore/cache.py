"""LRU resolution cache for name/ID lookups."""

from __future__ import annotations

import threading
from collections import OrderedDict

# Module-level shared cache for trace_id → experiment_id.
# Solves DynamoDB GSI eventual consistency: start_trace() populates this on one
# store instance, _resolve_trace_experiment() on a different instance finds it
# immediately instead of waiting for GSI propagation.
_shared_trace_exp_cache: OrderedDict[str, str] = OrderedDict()
_shared_trace_exp_lock = threading.Lock()
_SHARED_TRACE_CACHE_MAX = 10000


def shared_trace_exp_put(trace_id: str, experiment_id: str) -> None:
    """Record a trace_id → experiment_id mapping visible to all store instances."""
    with _shared_trace_exp_lock:
        if trace_id in _shared_trace_exp_cache:
            _shared_trace_exp_cache.move_to_end(trace_id)
        else:
            if len(_shared_trace_exp_cache) >= _SHARED_TRACE_CACHE_MAX:
                _shared_trace_exp_cache.popitem(last=False)
        _shared_trace_exp_cache[trace_id] = experiment_id


def shared_trace_exp_get(trace_id: str) -> str | None:
    """Look up a trace_id → experiment_id mapping from the shared cache."""
    with _shared_trace_exp_lock:
        if trace_id in _shared_trace_exp_cache:
            _shared_trace_exp_cache.move_to_end(trace_id)
            return _shared_trace_exp_cache[trace_id]
    return None


class ResolutionCache:
    """LRU cache for resolving names to IDs and vice versa.

    Used for:
    - run_id → experiment_id
    - model_name → model_ulid
    - experiment_name → experiment_id
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, str] = OrderedDict()

    def _key(self, namespace: str, key: str) -> str:
        return f"{namespace}:{key}"

    def get(self, namespace: str, key: str) -> str | None:
        """Get a cached value, or None if not found."""
        cache_key = self._key(namespace, key)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        return None

    def put(self, namespace: str, key: str, value: str) -> None:
        """Cache a value, evicting LRU entry if at capacity."""
        cache_key = self._key(namespace, key)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            self._cache[cache_key] = value
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[cache_key] = value

    def invalidate(self, namespace: str, key: str) -> None:
        """Remove a cached entry."""
        cache_key = self._key(namespace, key)
        self._cache.pop(cache_key, None)

"""LRU resolution cache for name/ID lookups."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable


class ResolutionCache:
    """LRU cache for resolving names to IDs and vice versa.

    Used for:
    - run_id → experiment_id
    - model_name → model_ulid
    - experiment_name → experiment_id
    """

    def __init__(
        self,
        max_size: int = 1000,
        workspace: Callable[[], str | None] | None = None,
    ) -> None:
        self._max_size = max_size
        self._workspace = workspace
        self._cache: OrderedDict[tuple[str, ...], str] = OrderedDict()

    def _key(self, namespace: str, key: str) -> tuple[str, ...]:
        ws = self._workspace() if self._workspace is not None else None
        if ws is not None:
            return (namespace, ws, key)
        return (namespace, key)

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

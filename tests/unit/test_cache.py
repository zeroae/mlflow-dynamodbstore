from mlflow_dynamodbstore.cache import (
    ResolutionCache,
    _shared_trace_exp_cache,
    shared_trace_exp_get,
    shared_trace_exp_put,
)


class TestSharedTraceExpCache:
    """Tests for the module-level shared trace_id → experiment_id cache."""

    def setup_method(self):
        _shared_trace_exp_cache.clear()

    def test_put_and_get(self):
        shared_trace_exp_put("tr-1", "exp-1")
        assert shared_trace_exp_get("tr-1") == "exp-1"

    def test_get_miss(self):
        assert shared_trace_exp_get("nonexistent") is None

    def test_lru_eviction(self):
        from mlflow_dynamodbstore import cache

        orig = cache._SHARED_TRACE_CACHE_MAX
        cache._SHARED_TRACE_CACHE_MAX = 2
        try:
            shared_trace_exp_put("a", "1")
            shared_trace_exp_put("b", "2")
            shared_trace_exp_put("c", "3")  # evicts "a"
            assert shared_trace_exp_get("a") is None
            assert shared_trace_exp_get("b") == "2"
            assert shared_trace_exp_get("c") == "3"
        finally:
            cache._SHARED_TRACE_CACHE_MAX = orig

    def test_overwrite_updates_value(self):
        shared_trace_exp_put("tr-1", "exp-old")
        shared_trace_exp_put("tr-1", "exp-new")
        assert shared_trace_exp_get("tr-1") == "exp-new"


class TestResolutionCache:
    def test_get_miss_returns_none(self):
        cache = ResolutionCache(max_size=100)
        assert cache.get("run", "abc123") is None

    def test_put_and_get(self):
        cache = ResolutionCache(max_size=100)
        cache.put("run", "abc123", "exp456")
        assert cache.get("run", "abc123") == "exp456"

    def test_invalidate(self):
        cache = ResolutionCache(max_size=100)
        cache.put("model_name", "my-model", "ulid123")
        cache.invalidate("model_name", "my-model")
        assert cache.get("model_name", "my-model") is None

    def test_lru_eviction(self):
        cache = ResolutionCache(max_size=2)
        cache.put("run", "a", "1")
        cache.put("run", "b", "2")
        cache.put("run", "c", "3")  # evicts "a"
        assert cache.get("run", "a") is None
        assert cache.get("run", "b") == "2"
        assert cache.get("run", "c") == "3"

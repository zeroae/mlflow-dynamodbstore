from mlflow_dynamodbstore.cache import ResolutionCache


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

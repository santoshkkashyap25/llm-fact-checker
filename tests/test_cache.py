import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from core.cache import QueryCache
from config import CACHE_PATH

class TestQueryCache:
    """Test suite for QueryCache"""

    @pytest.fixture
    def cache(self, tmp_path, monkeypatch):
        """Use temporary cache file"""
        fake_cache_path = tmp_path / "cache.json"
        monkeypatch.setattr("core.cache.CACHE_PATH", fake_cache_path)
        c = QueryCache()
        c.cache.clear()
        return c

    def test_set_and_get(self, cache):
        """Basic set/get flow"""
        result = {"verdict": "True"}
        cache.set("India is in Asia", result)
        retrieved = cache.get("India is in Asia")
        assert retrieved == result

    def test_expired_item_removed(self, cache, monkeypatch):
        """Expired entries are deleted"""
        key = cache.get_cache_key("old_claim")
        cache.cache[key] = {
            "result": {"verdict": "False"},
            "timestamp": (datetime.now() - timedelta(seconds=cache.ttl_seconds + 1)).isoformat(),
            "claim": "old_claim"
        }
        assert cache.get("old_claim") is None
        assert key not in cache.cache  # expired entry deleted

    def test_lru_eviction(self, cache):
        """Oldest entry should be evicted"""
        cache.max_size = 2
        cache.set("claim1", {"r": 1})
        cache.set("claim2", {"r": 2})
        first_key = list(cache.cache.keys())[0]
        cache.set("claim3", {"r": 3})
        # After eviction, first key should be gone
        assert first_key not in cache.cache
        assert len(cache.cache) == 2

    def test_clear(self, cache):
        cache.set("a", {"x": 1})
        cache.clear()
        assert len(cache.cache) == 0

    def test_save_and_load_disk(self, cache):
        """Save cache and reload"""
        cache.set("test", {"data": 1})
        cache.save_to_disk()

        new_cache = QueryCache()
        new_cache.load_from_disk()
        assert any("test" in entry["claim"] for entry in new_cache.cache.values())

    def test_save_to_disk_error(self, cache, monkeypatch):
        """Simulate file write failure"""
        def fake_open(*args, **kwargs):
            raise IOError("disk error")
        monkeypatch.setattr("builtins.open", fake_open)
        cache.set("x", {"y": 1})
        cache.save_to_disk()  # should log error, not crash

    def test_load_from_disk_error(self, cache, monkeypatch):
        """Simulate bad JSON load"""
        bad_path = cache.index_path if hasattr(cache, "index_path") else CACHE_PATH
        CACHE_PATH.write_text("{ invalid json }")
        with pytest.MonkeyPatch.context() as m:
            m.setattr("core.cache.CACHE_PATH", CACHE_PATH)
            broken_cache = QueryCache()
            assert isinstance(broken_cache.cache, dict)

    def test_get_stats(self, cache):
        cache.set("a", {"b": 1})
        stats = cache.get_stats()
        assert "size" in stats
        assert stats["max_size"] == cache.max_size
        assert stats["ttl_seconds"] == cache.ttl_seconds

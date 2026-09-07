from veritas.llm.cache import DiskCache, cache_key


def test_disk_cache_get_set(tmp_path):
    dc = DiskCache(tmp_path)
    key = cache_key("model", "system", "user")
    assert dc.get(key) is None
    dc.set(key, "hello world")
    assert dc.get(key) == "hello world"


def test_cache_key_deterministic():
    k1 = cache_key("m", "s", "u")
    k2 = cache_key("m", "s", "u")
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex


def test_cache_key_different_inputs():
    k1 = cache_key("m", "s1", "u")
    k2 = cache_key("m", "s2", "u")
    assert k1 != k2

import pytest

from lru import LRUCache


def _recorder():
    evicted = []

    def on_evict(key, value):
        evicted.append((key, value))

    return evicted, on_evict


@pytest.mark.parametrize("bad", [0, -1, -100, 1.5, 2.0, "3", None])
def test_capacity_must_be_a_positive_int(bad):
    with pytest.raises(ValueError):
        LRUCache(bad)


def test_new_cache_is_empty():
    cache = LRUCache(3)
    assert len(cache) == 0
    assert cache.keys() == []
    assert "a" not in cache
    assert (cache.hits, cache.misses) == (0, 0)


def test_put_then_get():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert len(cache) == 2


def test_missing_key_returns_default():
    cache = LRUCache(2)
    cache.put("a", 1)
    assert cache.get("zzz") is None
    assert cache.get("zzz", "fallback") == "fallback"
    assert cache.peek("zzz") is None
    assert cache.peek("zzz", 0) == 0


def test_none_is_a_storable_value():
    cache = LRUCache(2)
    cache.put("a", None)
    assert "a" in cache
    assert len(cache) == 1
    assert cache.get("a", "fallback") is None
    assert cache.peek("a", "fallback") is None
    assert (cache.hits, cache.misses) == (1, 0)


def test_falsy_values_are_stored_like_any_other():
    cache = LRUCache(4)
    for key, value in [("zero", 0), ("empty", ""), ("false", False), ("list", [])]:
        cache.put(key, value)
    assert cache.get("zero", "fallback") == 0
    assert cache.get("empty", "fallback") == ""
    assert cache.get("false", "fallback") is False
    assert cache.get("list", "fallback") == []
    assert cache.hits == 4


def test_inserting_into_a_full_cache_evicts_the_oldest_untouched_entry():
    cache = LRUCache(3)
    for key in "abcd":
        cache.put(key, key.upper())
    assert cache.keys() == ["b", "c", "d"]
    assert "a" not in cache
    assert cache.get("a") is None
    assert len(cache) == 3


def test_get_makes_an_entry_most_recently_used():
    cache = LRUCache(3)
    for key in "abc":
        cache.put(key, 0)
    cache.get("a")
    assert cache.keys() == ["b", "c", "a"]
    cache.put("d", 0)
    assert cache.keys() == ["c", "a", "d"]
    assert "b" not in cache


def test_updating_a_key_refreshes_it_and_never_evicts():
    evicted, on_evict = _recorder()
    cache = LRUCache(3, on_evict=on_evict)
    for key in "abc":
        cache.put(key, 0)
    cache.put("a", 10)
    assert len(cache) == 3
    assert cache.keys() == ["b", "c", "a"]
    assert cache.peek("a") == 10
    assert evicted == []
    cache.put("d", 0)
    assert cache.keys() == ["c", "a", "d"]
    assert evicted == [("b", 0)]


def test_peek_membership_len_keys_and_misses_leave_recency_alone():
    cache = LRUCache(3)
    for key in "abc":
        cache.put(key, key)
    assert cache.peek("a") == "a"
    assert "a" in cache
    assert len(cache) == 3
    assert cache.keys() == ["a", "b", "c"]
    assert cache.get("nope") is None
    cache.put("d", "d")
    assert "a" not in cache
    assert cache.keys() == ["b", "c", "d"]


def test_keys_returns_a_new_list_each_time():
    cache = LRUCache(3)
    cache.put("a", 1)
    cache.put("b", 2)
    snapshot = cache.keys()
    snapshot.append("x")
    snapshot.reverse()
    assert cache.keys() == ["a", "b"]
    assert cache.keys() is not cache.keys()


def test_on_evict_is_called_once_per_eviction_in_order():
    evicted, on_evict = _recorder()
    cache = LRUCache(2, on_evict=on_evict)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    cache.put("d", 4)
    assert evicted == [("b", 2), ("a", 1)]
    assert cache.keys() == ["c", "d"]


def test_on_evict_is_not_called_for_pop_or_overwrite():
    evicted, on_evict = _recorder()
    cache = LRUCache(2, on_evict=on_evict)
    cache.put("a", 1)
    cache.put("a", 2)
    cache.put("b", 3)
    assert cache.pop("a") == 2
    assert evicted == []


def test_pop_removes_and_returns_the_value():
    cache = LRUCache(3)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.pop("a") == 1
    assert "a" not in cache
    assert len(cache) == 1
    assert cache.keys() == ["b"]
    with pytest.raises(KeyError):
        cache.pop("a")
    with pytest.raises(KeyError):
        cache.pop("never-there")


def test_pop_frees_capacity():
    evicted, on_evict = _recorder()
    cache = LRUCache(2, on_evict=on_evict)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.pop("a")
    cache.put("c", 3)
    assert cache.keys() == ["b", "c"]
    assert evicted == []


def test_hits_and_misses_count_get_calls_only():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.get("a")
    cache.get("a")
    cache.get("b")
    cache.peek("a")
    cache.peek("b")
    assert "a" in cache
    assert "b" not in cache
    assert (cache.hits, cache.misses) == (2, 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert (cache.hits, cache.misses) == (2, 2)


def test_capacity_one():
    evicted, on_evict = _recorder()
    cache = LRUCache(1, on_evict=on_evict)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.keys() == ["b"]
    cache.put("b", 3)
    assert cache.keys() == ["b"]
    assert cache.get("b") == 3
    assert evicted == [("a", 1)]


def test_shrinking_evicts_least_recently_used_first():
    evicted, on_evict = _recorder()
    cache = LRUCache(4, on_evict=on_evict)
    for index, key in enumerate("abcd"):
        cache.put(key, index)
    cache.get("a")
    cache.resize(2)
    assert evicted == [("b", 1), ("c", 2)]
    assert cache.keys() == ["d", "a"]
    assert len(cache) == 2
    cache.put("e", 4)
    assert evicted == [("b", 1), ("c", 2), ("d", 3)]
    assert cache.keys() == ["a", "e"]


def test_growing_keeps_every_entry():
    evicted, on_evict = _recorder()
    cache = LRUCache(2, on_evict=on_evict)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.resize(3)
    cache.put("c", 3)
    assert cache.keys() == ["a", "b", "c"]
    assert evicted == []
    cache.put("d", 4)
    assert evicted == [("a", 1)]


@pytest.mark.parametrize("bad", [0, -1, 1.5, None])
def test_resize_rejects_invalid_capacity_and_changes_nothing(bad):
    evicted, on_evict = _recorder()
    cache = LRUCache(2, on_evict=on_evict)
    cache.put("a", 1)
    cache.put("b", 2)
    with pytest.raises(ValueError):
        cache.resize(bad)
    assert cache.keys() == ["a", "b"]
    assert evicted == []
    cache.put("c", 3)
    assert cache.keys() == ["b", "c"]


def test_keys_can_be_any_hashable():
    cache = LRUCache(3)
    cache.put((1, 2), "tuple")
    cache.put(3, "int")
    cache.put(frozenset({1}), "frozenset")
    assert cache.get((1, 2)) == "tuple"
    assert cache.get(3) == "int"
    assert cache.get(frozenset({1})) == "frozenset"


def test_instances_do_not_share_state():
    first = LRUCache(2)
    second = LRUCache(2)
    first.put("a", 1)
    first.get("a")
    assert "a" not in second
    assert len(second) == 0
    assert (second.hits, second.misses) == (0, 0)


def test_a_long_mixed_session():
    evicted, on_evict = _recorder()
    cache = LRUCache(3, on_evict=on_evict)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") == 1
    cache.put("d", 4)
    assert evicted == [("b", 2)]
    assert cache.get("b") is None
    assert cache.peek("c") == 3
    cache.put("c", 30)
    assert cache.keys() == ["a", "d", "c"]
    cache.put("e", 5)
    assert evicted == [("b", 2), ("a", 1)]
    assert cache.get("d") == 4
    assert "c" in cache
    cache.put("f", 6)
    assert evicted == [("b", 2), ("a", 1), ("c", 30)]
    assert cache.keys() == ["e", "d", "f"]
    assert cache.pop("e") == 5
    cache.put("g", 7)
    assert cache.get("f") == 6
    assert cache.get("zz") is None
    assert cache.keys() == ["d", "g", "f"]
    cache.resize(2)
    cache.put("h", 8)
    assert evicted == [("b", 2), ("a", 1), ("c", 30), ("d", 4), ("g", 7)]
    assert cache.keys() == ["f", "h"]
    assert (cache.hits, cache.misses) == (3, 2)
    assert len(cache) == 2

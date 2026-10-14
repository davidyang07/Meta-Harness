import pytest

import telemetry

API = ["increment", "counter", "set_gauge", "gauge", "snapshot", "reset"]
EMPTY = {"counters": {}, "gauges": {}}


@pytest.fixture(autouse=True)
def _isolated():
    original = getattr(telemetry, "default_registry", None)
    telemetry.reset()
    yield
    if original is not None:
        telemetry.default_registry = original
    telemetry.reset()


# ── the module-level API keeps working ───────────────────────────────


def test_counters():
    assert telemetry.counter("requests") == 0
    assert telemetry.increment("requests") == 1
    assert telemetry.increment("requests", 4) == 5
    assert telemetry.increment("requests", 0) == 5
    assert telemetry.counter("requests") == 5
    assert telemetry.counter("other") == 0


def test_counters_only_go_up():
    telemetry.increment("x", 2)
    with pytest.raises(ValueError):
        telemetry.increment("x", -1)
    assert telemetry.counter("x") == 2


def test_gauges():
    assert telemetry.gauge("temp") is None
    telemetry.set_gauge("temp", 21.5)
    telemetry.set_gauge("temp", 19.0)
    assert telemetry.gauge("temp") == 19.0


def test_snapshot_is_a_copy_and_reset_clears_everything():
    telemetry.increment("a", 2)
    telemetry.set_gauge("g", 1.5)
    snap = telemetry.snapshot()
    assert snap == {"counters": {"a": 2}, "gauges": {"g": 1.5}}
    snap["counters"]["a"] = 100
    snap["gauges"].clear()
    assert telemetry.counter("a") == 2
    assert telemetry.gauge("g") == 1.5
    telemetry.reset()
    assert telemetry.snapshot() == EMPTY


def test_track_calls_counts_calls_and_preserves_the_function():
    @telemetry.track_calls("calls.double")
    def double(x):
        """Double x."""
        return 2 * x

    assert double(4) == 8
    assert double(x=5) == 10
    assert telemetry.counter("calls.double") == 2
    assert double.__name__ == "double"
    assert double.__doc__ == "Double x."


# ── the Registry ──────────────────────────────────────────────────────


@pytest.mark.parametrize("name", API)
def test_registry_offers_the_same_operations(name):
    assert callable(getattr(telemetry.Registry, name))


def test_a_registry_behaves_like_the_module_did():
    registry = telemetry.Registry()
    assert registry.counter("x") == 0
    assert registry.increment("x") == 1
    assert registry.increment("x", 3) == 4
    with pytest.raises(ValueError):
        registry.increment("x", -2)
    assert registry.counter("x") == 4
    assert registry.gauge("g") is None
    registry.set_gauge("g", 2.5)
    assert registry.gauge("g") == 2.5
    snap = registry.snapshot()
    assert snap == {"counters": {"x": 4}, "gauges": {"g": 2.5}}
    snap["counters"]["x"] = 0
    assert registry.counter("x") == 4
    registry.reset()
    assert registry.snapshot() == EMPTY


def test_registries_are_independent():
    first = telemetry.Registry()
    second = telemetry.Registry()
    first.increment("x", 2)
    second.increment("x")
    first.set_gauge("g", 1.0)
    assert first.counter("x") == 2
    assert second.counter("x") == 1
    assert second.gauge("g") is None
    assert telemetry.counter("x") == 0
    assert telemetry.gauge("g") is None
    first.reset()
    assert second.counter("x") == 1


def test_default_registry_is_a_registry():
    assert isinstance(telemetry.default_registry, telemetry.Registry)


def test_module_functions_act_on_the_default_registry():
    telemetry.increment("hits", 3)
    assert telemetry.default_registry.counter("hits") == 3
    telemetry.default_registry.set_gauge("load", 0.75)
    assert telemetry.gauge("load") == 0.75
    assert telemetry.snapshot() == telemetry.default_registry.snapshot()


def test_module_functions_follow_a_replaced_default_registry():
    old = telemetry.default_registry
    old.increment("before")
    new = telemetry.Registry()
    telemetry.default_registry = new
    telemetry.increment("after", 2)
    telemetry.set_gauge("g", 3.0)
    assert new.counter("after") == 2
    assert new.gauge("g") == 3.0
    assert old.counter("after") == 0
    assert old.gauge("g") is None
    assert telemetry.counter("before") == 0
    assert telemetry.snapshot() == {"counters": {"after": 2}, "gauges": {"g": 3.0}}
    telemetry.reset()
    assert new.snapshot() == EMPTY
    assert old.counter("before") == 1


def test_tracked_functions_count_into_the_registry_current_at_call_time():
    @telemetry.track_calls("calls.f")
    def f():
        return "ok"

    assert f() == "ok"
    first = telemetry.default_registry
    second = telemetry.Registry()
    telemetry.default_registry = second
    f()
    f()
    assert first.counter("calls.f") == 1
    assert second.counter("calls.f") == 2


def test_no_module_level_containers_remain():
    leftovers = [
        name
        for name, value in vars(telemetry).items()
        if not name.startswith("__") and isinstance(value, (dict, list, set))
    ]
    assert leftovers == []

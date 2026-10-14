import inspect

import calculator
from calculator import add, sub


def test_add_basic():
    assert add(2, 3) == 5


def test_add_zero():
    assert add(0, 0) == 0


def test_add_negative():
    assert add(-1, 1) == 0


def test_sub_basic():
    assert sub(5, 3) == 2


def test_sub_self():
    assert sub(7, 7) == 0


def test_add_returns_int_when_given_ints():
    """add() of two ints returns an int (and not a bool)."""
    result = add(2, 3)
    assert isinstance(result, int) and not isinstance(result, bool)


def test_sub_function_unchanged():
    """sub() is left exactly as it was."""
    src = inspect.getsource(sub)
    body_lines = [
        line.strip()
        for line in src.split("\n")
        if line.strip() and not line.strip().startswith(("def ", '"""', "#", "'''"))
    ]
    assert body_lines == ["return a - b"], (
        f"sub() body should be exactly 'return a - b'; got {body_lines}"
    )


def test_module_only_defines_add_and_sub():
    """The module's public functions are exactly add and sub."""
    public_funcs = {
        name
        for name, obj in inspect.getmembers(calculator, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == "calculator"
    }
    assert public_funcs == {"add", "sub"}, (
        f"calculator module should expose only {{add, sub}}; got {public_funcs}"
    )

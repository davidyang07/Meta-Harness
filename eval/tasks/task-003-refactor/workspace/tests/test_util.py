import inspect

import util
from util import process_admins, process_engineers, process_managers


SAMPLE = [
    {"id": 1, "name": " alice ", "email": "ALICE@X.com", "role": "admin", "active": True},
    {"id": 2, "name": "bob",     "email": "bob@x.com",   "role": "manager", "active": True},
    {"id": 3, "name": " carol ", "email": "Carol@x.com", "role": "engineer", "active": True},
    {"id": 4, "name": "dan",     "email": "dan@x.com",   "role": "admin", "active": False},
    {"id": 5, "name": "eve",     "email": "eve@x.com",   "role": "manager", "active": True},
    {"id": 6, "name": " frank ", "email": "Frank@x.com", "role": "engineer", "active": False},
]


def test_process_admins():
    out = process_admins(SAMPLE)
    assert len(out) == 1
    assert out[0]["id"] == 1
    assert out[0]["name"] == "Alice"
    assert out[0]["email"] == "alice@x.com"


def test_process_managers():
    out = process_managers(SAMPLE)
    assert {u["id"] for u in out} == {2, 5}
    assert all(u["email"] == u["email"].lower() for u in out)


def test_process_engineers():
    out = process_engineers(SAMPLE)
    assert len(out) == 1
    assert out[0]["id"] == 3
    assert out[0]["name"] == "Carol"


def test_process_admins_empty_input():
    assert process_admins([]) == []


def test_process_managers_skips_inactive():
    inactive_only = [{"id": 9, "name": "x", "email": "x@x.com", "role": "manager", "active": False}]
    assert process_managers(inactive_only) == []


def test_each_public_function_is_thin_wrapper_after_refactor():
    """Each public function body has at most 4 non-empty lines."""
    for fn_name in ("process_admins", "process_managers", "process_engineers"):
        fn = getattr(util, fn_name)
        src = inspect.getsource(fn)
        body_lines = [
            line
            for line in src.split("\n")
            if line.strip() and not line.lstrip().startswith(("def ", '"""', "#"))
        ]
        assert len(body_lines) <= 4, (
            f"{fn_name} body has {len(body_lines)} non-empty lines; "
            "at most 4 are allowed."
        )


def test_role_string_not_in_public_function_bodies():
    """No public function compares a record's role itself."""
    role_for_fn = {
        "process_admins": "admin",
        "process_managers": "manager",
        "process_engineers": "engineer",
    }
    for fn_name, role in role_for_fn.items():
        fn = getattr(util, fn_name)
        src = inspect.getsource(fn)
        body = "\n".join(
            line for line in src.split("\n")
            if not line.lstrip().startswith(("def ", '"""', "'''", "#"))
        )
        for pattern in (f'!= "{role}"', f'== "{role}"', f"!= '{role}'", f"== '{role}'"):
            assert pattern not in body, (
                f"{fn_name} still has a direct role check ({pattern!r})."
            )


def test_helper_function_exists():
    """util defines at least one private (underscore-named) function."""
    helpers = [
        name
        for name, obj in inspect.getmembers(util, inspect.isfunction)
        if name.startswith("_") and obj.__module__ == "util"
    ]
    assert helpers, (
        "no private function (name starting with '_') found in util.py"
    )

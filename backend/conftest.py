"""Backend pytest config: repository-local temp root, and .env at session start.

Two session-wide concerns live here.

**`.env`** — lets ``test_inner.py``'s live LLM test pick up
``ANTHROPIC_API_KEY`` without the user having to export it in their shell.

**A repository-local ``--basetemp``** — pytest otherwise derives ``tmp_path``
from ``tempfile.gettempdir()``/``pytest-of-<user>``. That directory is
per-machine state outside the repository: if its ACL is broken (a real
condition on at least one Windows development box here, where every
``tmp_path`` test errored with ``PermissionError: [WinError 5]``) the suite
cannot run at all, and the failure looks like 159 broken tests rather than
one broken directory. Anchoring the temp root inside ``backend/`` makes the
suite depend only on the checkout it is running from, on every platform.

``--basetemp`` given explicitly on the command line still wins, so
``pytest --basetemp=...`` behaves normally.

Putting the temp root inside the checkout does mean ``git`` commands run
from a ``tmp_path`` now find *this* repository, and inherit its
``.gitattributes``. ``_pin_basetemp_git_attributes`` writes ``* text=auto``
at the temp root, which is what git applies when it has no repository and
no configuration — so a fixture behaves the same whether its temp
directory happens to sit inside this checkout or outside it, and a later
edit to the repository's own ``.gitattributes`` cannot change what the
suite measures.
"""

from pathlib import Path

import pytest
from dotenv import load_dotenv

#: Wiped and recreated by pytest at session start, so it must be a
#: directory nothing else owns. Gitignored.
BASETEMP_DIRNAME = ".pytest-tmp"


def pytest_configure(config):
    backend_root = Path(__file__).resolve().parent
    repo_root = backend_root.parent

    env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    if not config.option.basetemp:
        config.option.basetemp = str(backend_root / BASETEMP_DIRNAME)


@pytest.fixture(scope="session", autouse=True)
def _pin_basetemp_git_attributes(tmp_path_factory: pytest.TempPathFactory):
    """Give fixtures git's no-repository line-ending behaviour.

    The temp root is inside the working tree, so ``git`` invocations under
    ``tmp_path`` resolve this repository's attributes. ``text=auto`` is
    what git uses with no repository and no configuration, and an
    attributes file in a subdirectory overrides its parents — so pinning
    it here decouples the suite from the repository's own settings.

    ``-text`` would *not* do: it disables the CRLF reconciliation that
    ``git apply`` performs on an LF patch against a CRLF file, which is
    the behaviour ``tests/test_tools.py`` exists to protect.
    """
    basetemp = tmp_path_factory.getbasetemp()
    (basetemp / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")

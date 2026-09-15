"""Bootstrap sys.path so the vendored SyntaxSHAP subset in `thirdparty/` can be
imported using its own flat, absolute import style (`import maskers`,
`import models`, `import utils`, `from _serializable import ...`).

The upstream syntax-shap-main project (https://github.com/ ... , vendored here
as a trimmed subset — see plans/xai-syntaxshap-promptguard.md) is not an
installable package: it has no setup.py/pyproject.toml, and its own modules
use top-level absolute imports that only resolve when its directory sits
directly on sys.path (exactly what happens when you `python syntaxshap/main.py`
and Python puts that directory in sys.path[0]). We replicate that by inserting
`thirdparty/` itself.

Call `ensure_thirdparty_on_path()` before importing anything from `maskers`,
`models`, `utils`, or `_serializable` inside this package. It is idempotent —
safe to call from every module that needs it.
"""
import sys
from pathlib import Path

_THIRDPARTY_DIR = str(Path(__file__).parent / "thirdparty")


def ensure_thirdparty_on_path() -> None:
    """Insert the vendored thirdparty/ directory at the front of sys.path, once."""
    if _THIRDPARTY_DIR not in sys.path:
        sys.path.insert(0, _THIRDPARTY_DIR)

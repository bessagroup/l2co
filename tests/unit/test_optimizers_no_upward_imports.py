"""Guard the optimizers subpackage against upward imports.

The ``l2co._src.optimizers`` subpackage must depend only *downward* (on
the shared substrate + ``l2co_tasks.Task``); it must never import the
selection models, the strategy wrapper, or the meta-optimizer module.
That back-edge was the ``l2co_update -> strategy_wrapper -> sub_optimizer``
import cycle broken in ``docs/adr/0007``; ``UpdateClass.l2co_update`` was
moved out to ``l2co._src.meta_optimizer`` precisely so this direction
stays one-way. This test fails CI if the cycle is reintroduced.
"""

from __future__ import annotations

import ast
from pathlib import Path

OPTIMIZERS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "l2co"
    / "_src"
    / "optimizers"
)
SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src"

# Modules the optimizers subpackage must not reach up into.
FORBIDDEN_PREFIXES = (
    "l2co._src.models",
    "l2co._src.strategy_wrapper",
    "l2co._src.meta_optimizer",
)


def _python_files() -> list[Path]:
    return [
        p for p in OPTIMIZERS_DIR.rglob("*.py") if "__pycache__" not in p.parts
    ]


def _package_of(path: Path) -> str:
    """Dotted package of the module's parent (``l2co._src.optimizers``)."""
    rel = path.resolve().relative_to(SRC_ROOT)
    return ".".join(rel.parent.parts)


def _resolve(level: int, module: str | None, package: str) -> str:
    """Resolve a possibly-relative import target to an absolute module path."""
    if level == 0:
        return module or ""
    base = package.split(".")
    if level - 1:
        base = base[: -(level - 1)]
    return ".".join(base + ([module] if module else []))


def test_optimizers_do_not_import_models_or_strategy() -> None:
    offenders: list[tuple[Path, int, str]] = []
    for path in _python_files():
        package = _package_of(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.ImportFrom):
                targets.append(_resolve(node.level, node.module, package))
            elif isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            for target in targets:
                if target.startswith(FORBIDDEN_PREFIXES):
                    offenders.append((path, node.lineno, target))
    assert not offenders, (
        "l2co._src.optimizers must not import models / strategy_wrapper / "
        "meta_optimizer (that reintroduces the import cycle — see "
        "docs/adr/0007). Offenders:\n"
        + "\n".join(
            f"  {p.relative_to(SRC_ROOT.parent)}:{ln} -> {mod}"
            for p, ln, mod in offenders
        )
    )

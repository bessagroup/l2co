"""Guard against experiments reaching into ``l2co._src.*``.

Phase 3 of the restructure promoted everything ``experiments/`` needed into
the public facade modules. This test catches accidental regressions.
"""

from __future__ import annotations

import ast
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments"


def _python_files() -> list[Path]:
    return [
        p
        for p in EXPERIMENTS_DIR.rglob("*.py")
        if "__pycache__" not in p.parts
    ]


def test_no_private_l2co_imports_in_experiments() -> None:
    offenders: list[tuple[Path, int, str]] = []
    for path in _python_files():
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("l2co._src"):
                    offenders.append((path, node.lineno, node.module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("l2co._src"):
                        offenders.append((path, node.lineno, alias.name))
    assert not offenders, (
        "experiments/ must import only from l2co's public surface. "
        "Offenders:\n"
        + "\n".join(
            f"  {p.relative_to(EXPERIMENTS_DIR.parent)}:{ln} → {mod}"
            for p, ln, mod in offenders
        )
    )

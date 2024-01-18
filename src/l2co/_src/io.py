"""
Module containing common IO operations

Functions
---------

_validate_path
    Validate the path to a file or directory.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from pathlib import Path

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


def _validate_path(path: str | float, project_dir: Path) -> bool:
    """
    Validate the path to a file or directory.

    Parameters
    ----------
    path : str or float
        Path to a file or directory.
    project_dir : Path
        Path to the project directory.

    Returns
    -------
    bool
        True if the path exists, False otherwise.
    """
    if isinstance(path, str):
        return Path(project_dir / path).exists()

    return False

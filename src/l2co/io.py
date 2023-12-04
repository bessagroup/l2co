"""
Module containing common IO operations
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from pathlib import Path

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Under development'
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

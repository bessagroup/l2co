"""Abstract base classes shared by all L2CO model implementations."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, TypeVar

import equinox as eqx

# =============================================================================

T = TypeVar("T", bound="AbstractModel")


class AbstractModel(eqx.Module):
    """
    Abstract base class for policy models.
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize model to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing model configuration.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def reconstruct(cls: type[T], **kwargs) -> T:
        """
        Reconstruct model from dictionary.

        Parameters
        ----------
        **kwargs
            Model configuration key-value pairs.

        Returns
        -------
        T
            Reconstructed model instance.
        """
        raise NotImplementedError

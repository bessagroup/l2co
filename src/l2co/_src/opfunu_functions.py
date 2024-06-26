"""
Port of the opfunu functions to the PyBenchFunction class.

Functions
---------
OpfunuFunc
    Base class for the opfunu functions.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from typing import List, Optional, Tuple

# Third-party
import autograd.numpy as np
import opfunu
from f3dasm._src.datageneration.functions.adapters.pybenchfunction import \
    PyBenchFunction
from opfunu.cec_based import CecBenchmark

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================

opfunu.np = np

# =============================================================================


class OpfunuFunc(PyBenchFunction):
    opfunu_class: type[CecBenchmark] = None
    name: str = None

    def __init__(
            self,
            dimensionality: int,
            scale_bounds: Optional[np.ndarray] = None,
            noise: Optional[float] = None,
            offset: bool = True,
            seed: Optional[int] = None):
        """
        _description_

        Parameters
        ----------
        dimensionality
            number of dimensions
        scale_bounds, optional
            array containing the lower and upper bound of the scaling
             factor of the input data, by default None
        noise, optional
            inflict Gaussian noise on the input, by default None
        offset, optional
            set this True to randomly off-set the pybenchfunction,
             by default True
        seed, optional
            seed for the random number generator, by default None
        """
        self.opfunu_function = self.opfunu_class(ndim=dimensionality)
        super().__init__(dimensionality=dimensionality,
                         scale_bounds=scale_bounds,
                         noise=noise,
                         offset=offset,
                         seed=seed)

    @classmethod
    def from_partial(
            cls, opfunu_class: type[CecBenchmark]) -> type[OpfunuFunc]:
        class OpfunuFuncPartial(OpfunuFunc):
            ...

        OpfunuFuncPartial.opfunu_class = opfunu_class
        OpfunuFuncPartial.name = opfunu_class.__name__
        OpfunuFuncPartial.differentiable = opfunu_class.differentiable
        OpfunuFuncPartial.convex = opfunu_class.convex
        OpfunuFuncPartial.parametric = opfunu_class.parametric
        OpfunuFuncPartial.randomized_term = opfunu_class.randomized_term
        OpfunuFuncPartial.separable = opfunu_class.separable
        OpfunuFuncPartial.multimodal = not opfunu_class.unimodal

        return OpfunuFuncPartial

    @classmethod
    def is_dim_compatible(cls, d: int) -> bool:
        _o = cls.opfunu_class(ndim=None)
        if hasattr(_o, 'dim_supported'):
            return cls._check_dim_supported(d, _o.dim_supported, _o.dim_max)

        else:
            return _o.is_ndim_compatible(d)

    @staticmethod
    def _check_dim_supported(
            d: int,
            dim_supported: List[int] | None, dim_max: int) -> bool:
        if dim_supported is None:
            return d < dim_max

        return d in dim_supported

    #                                                                Properties
    # =========================================================================

    # @classmethod
    # @property
    # def differentiable(cls) -> bool:
    #     return cls.opfunu_class.differentiable

    @property
    def convex(self) -> bool:
        return self.opfunu_class.convex

    @property
    def parametric(self) -> bool:
        return self.opfunu_class.parametric

    @property
    def randomized_term(self) -> bool:
        return self.opfunu_class.randomized_term

    @property
    def separable(self) -> bool:
        return self.opfunu_class.separable

    @property
    def multimodal(self) -> bool:
        return not self.opfunu_class.unimodal

    def get_global_minimum(
            self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(self.opfunu_function.x_global)
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X: np.ndarray) -> float:
        return self.opfunu_function.evaluate(X)

    def _set_parameters(self):
        self.input_domain = self.opfunu_function.bounds

# =============================================================================


CEC_FUNCTIONS = {
    f.__name__.lower(): OpfunuFunc.from_partial(f)
    for f in opfunu.get_all_cec_functions()
    if not f.__name__.endswith('2010') and not f.__name__.endswith('2019')}

# =============================================================================

NAMED_FUNCTIONS = {
    f.__name__.lower(): OpfunuFunc.from_partial(f)
    for f in opfunu.get_all_named_functions()}

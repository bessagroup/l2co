from __future__ import annotations

# Standard
from typing import List, Optional, Tuple

# Third-party
import jax.numpy as jnp
import numpy as np
import opfunu
import pandas as pd
from f3dasm import Block, ExperimentData
from f3dasm._src.datageneration.functions.adapters.pybenchfunction import \
    PyBenchFunction
from opfunu import create_benchmark_loss_function
from opfunu.cec_based import CecBenchmark

# Local
from ..task import Task

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
        Opfunu function base class.

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
        super().__init__(scale_bounds=scale_bounds,
                         noise=noise,
                         offset=offset,
                         seed=seed)

    @classmethod
    def from_partial(
            cls, opfunu_class: type[CecBenchmark]) -> type[OpfunuFunc]:
        """
        Create a partial class from the opfunu class.

        Parameters
        ----------
        opfunu_class
            opfunu class to create a partial class from

        Returns
        -------
        type[OpfunuFunc]
            partial class
        """
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
        """
        Check if the dimensionality is compatible with the Opfunu function

        Parameters
        ----------
        d
            dimensionality

        Returns
        -------
        bool
            True if the dimensionality is compatible, False otherwise
        """
        _o = cls.opfunu_class(ndim=None)
        if hasattr(_o, 'dim_supported'):
            return cls._check_dim_supported(d, _o.dim_supported, _o.dim_max)

        else:
            return _o.is_ndim_compatible(d)

    @staticmethod
    def _check_dim_supported(
            d: int,
            dim_supported: List[int] | None, dim_max: int) -> bool:
        """
        Static method to check if the dimensionality is supported.

        Parameters
        ----------
        d
            dimensionality
        dim_supported
            list of supported dimensionality, by default None
        dim_max
            maximum dimensionality, by default None

        Returns
        -------
        bool
            True if the dimensionality is supported, False otherwise
        """
        if dim_supported is None:
            return d < dim_max

        return d in dim_supported

    #                                                                Properties
    # =========================================================================

    @property
    def convex(self) -> bool:
        """
        Convex property

        Returns
        -------
        bool
            True if the function is convex, False otherwise
        """
        return self.opfunu_class.convex

    @property
    def parametric(self) -> bool:
        """
        Parametric property

        Returns
        -------
        bool
            True if the function is parametric, False otherwise
        """
        return self.opfunu_class.parametric

    @property
    def randomized_term(self) -> bool:
        """
        Randomized term property

        Returns
        -------
        bool
            True if the function has a randomized term, False otherwise
        """
        return self.opfunu_class.randomized_term

    @property
    def separable(self) -> bool:
        """
        Separable property

        Returns
        -------
        bool
            True if the function is separable, False otherwise
        """
        return self.opfunu_class.separable

    @property
    def multimodal(self) -> bool:
        """
        Multimodal property

        Returns
        -------
        bool
            True if the function is multimodal, False otherwise
        """
        return not self.opfunu_class.unimodal

    def get_global_minimum(
            self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the global minimum of the function.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            input parameters and objective function value of global minimum
        """
        X = np.atleast_2d(self.opfunu_function.x_global)
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X: np.ndarray) -> float:
        """
        Evaluate the objective function value at a given point.

        Parameters
        ----------
        X
            input parameters

        Returns
        -------
        float
            objective function value
        """
        return self.opfunu_function.evaluate(X)

    def _set_parameters(self):
        """
        Set the parameters of the function.

        Note
        ----
        This method is called by the base class to set the parameters of the
        function.
        """
        self.input_domain = self.opfunu_function.bounds

#                                                     CEC competition functions
# =============================================================================


CEC_FUNCTIONS = {
    f.__name__.lower(): OpfunuFunc.from_partial(f)
    for f in opfunu.get_all_cec_functions()
    if not f.__name__.endswith('2010') and not f.__name__.endswith('2019')}

#                                                    Named analytical functions
# =============================================================================

NAMED_FUNCTIONS = {
    f.__name__.lower(): OpfunuFunc.from_partial(f)
    for f in opfunu.get_all_named_functions()}

# =============================================================================

MY_FUNCTION_MAPPING = {**CEC_FUNCTIONS, **NAMED_FUNCTIONS}


class BenchmarkFuncSampler(Block):
    def __init__(self, noise_distribution: List[float]):
        self.noise_distribution = noise_distribution

    def call(self, data: ExperimentData, n_samples: int,
             seed: int) -> ExperimentData:
        dims = data.domain.input_space['dimensionality'].categories
        fns = data.domain.input_space['fn_name'].categories
        noises = data.domain.input_space['noise'].categories

        dim_filters = {d: list(filter(
            lambda f: MY_FUNCTION_MAPPING[f.lower()].is_dim_compatible(d)
            and MY_FUNCTION_MAPPING[f.lower()].differentiable, list(fns)))
            for d in dims}

        rng = np.random.default_rng(seed=seed)

        dims_list = rng.choice(
            dims,
            size=n_samples)
        function_names_list = [rng.choice(dim_filters[d]) for d in dims_list]
        noise_list = rng.choice(noises,
                                size=n_samples,
                                p=self.noise_distribution)
        seed_list = rng.integers(0, 2**30, size=n_samples)

        df = pd.DataFrame({
            "dimensionality": dims_list,
            "fn_name": function_names_list,
            "noise": noise_list,
            "seed": seed_list
        })

        return ExperimentData(domain=data.domain, input_data=df)


class CEC2019Sampler(Block):
    def __init__(self):
        self.dim_mapping = {
            'F12019': 9,
            'F22019': 16,
            'F32019': 18,
            'F42019': 10,
            'F52019': 10,
            'F62019': 10,
            'F72019': 10,
            'F82019': 10,
            'F92019': 10,
            'F102019': 10,
        }

    def call(self, data: ExperimentData) -> ExperimentData:
        fns = data.domain.input_space['fn_name'].categories

        df = pd.DataFrame({
            "dimensionality": [self.dim_mapping[f] for f in fns],
            "fn_name": fns,
        })

        return ExperimentData(domain=data.domain, input_data=df)


def create_benchmark_task(
    fn_name: str, seed: int, noise: float,
        dimensionality: int, offset: bool = True,
        scale_output: bool = True, grad_noise: float = 0.0) -> Task:

    pass_rng = True if noise > 0.0 else False

    model, loss_fn, tag = create_benchmark_loss_function(
        fn_name=str(fn_name),
        seed=int(seed),
        scale_bounds=jnp.tile(
            jnp.array([0., 1.]), (dimensionality, 1)),
        dimensionality=int(dimensionality),
        noise=float(noise),
        offset=offset,
        scale_output=scale_output,
        grad_noise=float(noise),
    )

    return Task(model=model, pass_rng=pass_rng,
                loss_fn=loss_fn,
                tag=tag)

"""
EvoSax optimizer name registry.

The closure factories (``evosax_distribution_based_fn`` /
``evosax_population_based_fn``) and the ``UpdateClass`` factory
classmethods both live in ``l2co._src.optimizers.update_class``. This
module only keeps the optimizer-name registry and binds each EvoSax
algorithm directly to the appropriate ``UpdateClass`` classmethod via
:class:`jax.tree_util.Partial`. Population size defaults to
:func:`l2co._src.optimizers.utils.variable_popsize` on each classmethod,
except for the algorithms in :data:`_EVEN_POPSIZE_REQUIRED`, which are
bound to :func:`l2co._src.optimizers.utils.variable_popsize_even` because
they require an even population size. The caller can override either
default by supplying ``popsize`` in the ``OptimizationStep``
hyperparameters.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
from evosax.algorithms import RandomSearch, algorithms
from evosax.algorithms.distribution_based import distribution_based_algorithms
from evosax.algorithms.population_based import population_based_algorithms
from jax.tree_util import Partial

# Local
from l2co._src.optimizers.update_class import UpdateClass
from l2co._src.optimizers.utils import normalize_key, variable_popsize_even

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

normalized_evosax = {normalize_key(k): v for k, v in algorithms.items()}

# remove svcmaes and svopenes from the mapping as they require special
# treatment
_ = normalized_evosax.pop("svcmaes", None)
_ = normalized_evosax.pop("svopenes", None)
_ = normalized_evosax.pop("evotfes", None)
_ = normalized_evosax.pop("les", None)
_ = normalized_evosax.pop("lga", None)


# EvoSax algorithms (normalized names) that require an even population
# size. The antithetic / mirrored-sampling strategies assert
# ``population_size % 2 == 0`` at construction, and ESMC silently returns
# ``popsize - 1`` candidates for an odd population size, which then
# mismatches the ``popsize``-shaped buffers downstream. These default to
# :func:`variable_popsize_even` instead of :func:`variable_popsize`.
_EVEN_POPSIZE_REQUIRED = frozenset(
    {
        "ars",
        "asebo",
        "crfmnes",
        "esmc",
        "guidedes",
        "noisereusees",
        "openes",
        "persistentes",
        "pgpe",
    }
)


def _factory_for(evosax_class: type):
    """Pick the ``UpdateClass`` classmethod matching an EvoSax algorithm."""
    if evosax_class in distribution_based_algorithms.values():
        if evosax_class is RandomSearch:
            return UpdateClass.evosax_random_search_update
        return UpdateClass.evosax_distribution_update
    if evosax_class in population_based_algorithms.values():
        return UpdateClass.evosax_population_update
    raise ValueError(f"EvoSax class '{evosax_class.__name__}' not recognized.")


def _evosax_partial(name: str, cls: type) -> Partial:
    """Bind ``cls`` to its ``UpdateClass`` factory.

    Algorithms in :data:`_EVEN_POPSIZE_REQUIRED` get
    :func:`variable_popsize_even` as their default population size; the
    rest fall back to the classmethod's :func:`variable_popsize` default.
    """
    factory = _factory_for(cls)
    if name in _EVEN_POPSIZE_REQUIRED:
        return Partial(factory, optimizer=cls, popsize=variable_popsize_even)
    return Partial(factory, optimizer=cls)


evosax_mapping = {
    name: _evosax_partial(name, cls) for name, cls in normalized_evosax.items()
}

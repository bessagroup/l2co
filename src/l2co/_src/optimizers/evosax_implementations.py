"""
EvoSax optimizer name registry.

The closure factories (``evosax_distribution_based_fn`` /
``evosax_population_based_fn``) and the ``UpdateClass`` factory
classmethods both live in ``l2co._src.optimizers.update_class``. This
module only keeps the optimizer-name registry and binds each EvoSax
algorithm directly to the appropriate ``UpdateClass`` classmethod via
:class:`jax.tree_util.Partial`. Population size defaults to
:func:`l2co._src.optimizers.utils.variable_popsize` on each classmethod;
the caller can override it by supplying ``popsize`` in the
``OptimizationStep`` hyperparameters.
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
from l2co._src.optimizers.utils import normalize_key

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


def _factory_for(evosax_class: type):
    """Pick the ``UpdateClass`` classmethod matching an EvoSax algorithm."""
    if evosax_class in distribution_based_algorithms.values():
        if evosax_class is RandomSearch:
            return UpdateClass.evosax_random_search_update
        return UpdateClass.evosax_distribution_update
    if evosax_class in population_based_algorithms.values():
        return UpdateClass.evosax_population_update
    raise ValueError(f"EvoSax class '{evosax_class.__name__}' not recognized.")


evosax_mapping = {
    name: Partial(_factory_for(cls), optimizer=cls)
    for name, cls in normalized_evosax.items()
}

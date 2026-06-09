"""
Optax optimizer name registry.

The closure factories (``optax_fn`` / ``optax_extra_kwargs_fn``) and the
``UpdateClass`` factory classmethods both live in
``l2co._src.optimizers.update_class``. This module only keeps the
optimizer-name lists and binds each name to the appropriate
``UpdateClass`` classmethod via :class:`jax.tree_util.Partial`. The
caller is responsible for supplying every hyperparameter the underlying
optax constructor needs (e.g. ``learning_rate`` for ``adam``); no
defaults are injected here.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import optax
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

opt_names_normal = [
    "adabelief",
    "adadelta",
    "adan",
    "adafactor",
    "adagrad",
    "adam",
    "adamw",
    "adamax",
    "adamaxw",
    "amsgrad",
    "fromage",
    "lamb",
    "lars",
    "lion",
    "nadam",
    "nadamw",
    "noisy_sgd",
    "novograd",
    "optimistic_gradient_descent",
    "optimistic_adam",
    # "polyak_sgd", # function call signature is different, ignored!
    "radam",
    "rmsprop",
    "rprop",
    "sgd",
    "sign_sgd",
    # "signum", # not available in this version of optax!
    "sm3",
    "yogi",
]

opt_names_from_state = [
    "lbfgs",
]

normalized_optax_normal = {
    normalize_key(name): getattr(optax, name) for name in opt_names_normal
}

normalized_optax_from_state = {
    normalize_key(name): getattr(optax, name) for name in opt_names_from_state
}

optax_mapping = {
    **{
        name: Partial(UpdateClass.optax_update, optimizer=cls)
        for name, cls in normalized_optax_normal.items()
    },
    **{
        name: Partial(UpdateClass.optax_update_extra_kwargs, optimizer=cls)
        for name, cls in normalized_optax_from_state.items()
    },
}

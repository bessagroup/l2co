"""Public API for optimizers and for *authoring* new ones.

This module is the single stable surface an optimizer author builds
against (``docs/adr/0007``). Everything needed to write a new optimizer
-- or a meta-optimizer that self-registers via
:func:`register_optimizer` -- is re-exported here, so downstream packages
(rl2co, agentic-l2co) never have to reach into ``l2co._src``:

* **Registry & construction** -- ``optimizer_mapping`` / ``optimizers``
  (name -> factory), the ``register_optimizer`` extension point, the
  ``UpdateClass`` container every factory returns, and the
  ``create_schedules_experimentdata`` helper.
* **Contract types** -- the ``InitFunction`` / ``StepFunction`` /
  ``StopFunction`` signatures, ``InputParameters`` / ``OptState`` /
  ``SamplerFunction`` aliases, and the ``PopSize`` spec.
* **Loss helpers** -- the ``vmapped_loss*`` helpers that evaluate a
  task's loss (and gradients) over a population, plus
  ``menu_loss_and_grad``, the menu-dispatched evaluator that spends
  only the *active* optimizer's evaluation width instead of the
  menu-wide generation width (``docs/adr/0011``).
* **Sub-optimizer primitives** -- ``SubOpt`` and the
  ``grad_sub_optimizer`` / ``pop_sub_optimizer`` / ``optstep_to_subopt``
  factories, ``resolve_popsize``, ``l2co_native_optax``,
  ``l2co_native_evosax``, and ``CONSTRUCTOR_HYPERPARAMETERS`` -- the
  shared adapter layer both strategy wrappers dispatch across.
* **Handshake policy** -- ``HandshakePolicy``, the ``HANDSHAKE_POLICY``
  table, ``DEFAULT_HANDSHAKE_POLICY`` and ``handshake_policy_for``: which
  handshake variants an optimizer needs when control passes to it. The
  single source of truth shared by the rl2co environment and both
  deployment wrappers (``docs/adr/0009``).
* **State transfer** -- ``TransferBundle``, ``build_transfer_fns``,
  ``transfer_spec_for``, the ``FAMILY_*`` / ``CONF_*`` constants and
  ``TRANSFER_OVERRIDES``: what one optimizer hands the next across a
  switch, and how each family reads and writes it. The handshake policy
  above decides *which population* and *whether state survives*; this
  decides *what information crosses* (``docs/adr/0013``).
* **Base optimizer factories** -- the ``normalized_optax_normal`` /
  ``normalized_optax_from_state`` optax registries and
  ``random_search_update``.
* **Built-in algorithms** -- the ``SHADE`` algorithm (``shade_update``
  factory, ``shade_popsize`` default) and the L-BFGS entry
  (``lbfgs_update`` factory with per-eval-key noise semantics on
  stochastic tasks, plus the ``lbfgs_per_eval_key`` optimizer and
  ``scale_by_zoom_linesearch_per_eval_key`` linesearch building blocks).
"""

# Registry, construction, built-in algorithms
from ._src.opt_history import OptHistory
from ._src.optimizers import (
    SHADE,
    UpdateClass,
    create_schedules_experimentdata,
    lbfgs_per_eval_key,
    lbfgs_update,
    optimizer_mapping,
    optimizers,
    register_optimizer,
    scale_by_zoom_linesearch_per_eval_key,
    shade_popsize,
    shade_update,
)
from ._src.optimizers.handshake_policy import (
    DEFAULT_HANDSHAKE_POLICY,
    HANDSHAKE_POLICY,
    OPT_STATE_VARIANTS,
    POPULATION_HANDSHAKES,
    POPULATION_VARIANTS,
    HandshakePolicy,
    handshake_policy_for,
    population_best,
    population_best_one,
    population_handshake_for,
)
from ._src.optimizers.menu_eval import (
    GRAD_EVALUATION_WIDTH,
    menu_loss_and_grad,
)

# Base optimizer factories / registries
from ._src.optimizers.optax_implementations import (
    normalized_optax_from_state,
    normalized_optax_normal,
)
from ._src.optimizers.random_search import random_search_update
from ._src.optimizers.state_transfer import (
    CONF_ABSENT,
    CONF_ESTIMATED,
    CONF_EXACT,
    FAMILIES,
    FAMILY_DISTRIBUTION,
    FAMILY_GRADIENT,
    FAMILY_POPULATION,
    TRANSFER_OVERRIDES,
    TransferBundle,
    TransferSpec,
    build_transfer_fns,
    transfer_spec_for,
    write_none,
)

# Sub-optimizer primitives shared by the strategy wrappers
from ._src.optimizers.sub_optimizer import (
    CONSTRUCTOR_HYPERPARAMETERS,
    LINESEARCH_FEVAL_BOUND,
    SubOpt,
    grad_sub_optimizer,
    l2co_native_evosax,
    l2co_native_optax,
    optstep_to_subopt,
    pop_sub_optimizer,
    resolve_popsize,
)

# Loss helpers
from ._src.tree_utils import (
    vmapped_loss,
    vmapped_loss_and_grad,
    vmapped_loss_and_grad_with_rng,
    vmapped_loss_with_rng,
)

# Contract types
from ._src.typing import (
    InitFunction,
    InputParameters,
    OptState,
    PopSize,
    SamplerFunction,
    StepFunction,
    StopFunction,
)

__all__ = [
    "CONF_ABSENT",
    "CONF_ESTIMATED",
    "CONF_EXACT",
    "CONSTRUCTOR_HYPERPARAMETERS",
    "DEFAULT_HANDSHAKE_POLICY",
    "FAMILIES",
    "FAMILY_DISTRIBUTION",
    "FAMILY_GRADIENT",
    "FAMILY_POPULATION",
    "GRAD_EVALUATION_WIDTH",
    "HANDSHAKE_POLICY",
    "HandshakePolicy",
    "InitFunction",
    "InputParameters",
    "LINESEARCH_FEVAL_BOUND",
    "OPT_STATE_VARIANTS",
    "OptHistory",
    "OptState",
    "POPULATION_HANDSHAKES",
    "POPULATION_VARIANTS",
    "PopSize",
    "SHADE",
    "SamplerFunction",
    "StepFunction",
    "StopFunction",
    "SubOpt",
    "TRANSFER_OVERRIDES",
    "TransferBundle",
    "TransferSpec",
    "UpdateClass",
    "build_transfer_fns",
    "create_schedules_experimentdata",
    "grad_sub_optimizer",
    "handshake_policy_for",
    "l2co_native_evosax",
    "l2co_native_optax",
    "lbfgs_per_eval_key",
    "lbfgs_update",
    "menu_loss_and_grad",
    "normalized_optax_from_state",
    "normalized_optax_normal",
    "optimizer_mapping",
    "optimizers",
    "optstep_to_subopt",
    "pop_sub_optimizer",
    "population_best",
    "population_best_one",
    "population_handshake_for",
    "random_search_update",
    "register_optimizer",
    "resolve_popsize",
    "scale_by_zoom_linesearch_per_eval_key",
    "shade_popsize",
    "shade_update",
    "transfer_spec_for",
    "vmapped_loss",
    "vmapped_loss_and_grad",
    "vmapped_loss_and_grad_with_rng",
    "vmapped_loss_with_rng",
    "write_none",
]

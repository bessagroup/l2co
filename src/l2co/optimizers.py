"""Public API for optimizers and for *authoring* new ones.

Since l2co ADR 0016 the bare optimizers live in the sibling
``l2co-optimizers`` package; this module re-exports its surface --
the same objects, not copies -- so ``l2co.optimizers.X`` keeps working
and remains the stable import path for downstream packages (rl2co,
agentic-l2co). New code may import from ``l2co_optimizers`` directly.
Everything needed to write a new optimizer -- or a meta-optimizer that
self-registers via :func:`register_optimizer` -- is reachable here
(``docs/adr/0007``):

* **Registry & construction** -- ``optimizer_mapping`` / ``optimizers``
  (name -> factory), the ``register_optimizer`` extension point and its
  naming twin ``register_schedule_namer`` (how a meta-optimizer's steps
  are named, ``docs/adr/0016``), the ``UpdateClass`` container every
  factory returns, and the ``create_schedules_experimentdata`` helper.
* **Contract types** -- the ``InitFunction`` / ``StepFunction`` /
  ``StopFunction`` signatures, the handshake-side ``AskFunction`` /
  ``TransferReadFunction`` / ``TransferWriteFunction`` signatures,
  ``InputParameters`` / ``OptState`` / ``SamplerFunction`` aliases, and
  the ``PopSize`` spec.
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
* **Base optimizer factories** -- the ``optax_update`` /
  ``optax_update_extra_kwargs`` / ``evosax_distribution_update`` /
  ``evosax_population_update`` ``UpdateClass`` factories, the
  ``normalized_optax_normal`` / ``normalized_optax_from_state`` optax
  registries and ``random_search_update``.
* **Built-in algorithms** -- the ``SHADE`` algorithm (``shade_update``
  factory, ``shade_popsize`` default) and the L-BFGS entry
  (``lbfgs_update`` factory with per-eval-key noise semantics on
  stochastic tasks, plus the ``lbfgs_per_eval_key`` optimizer and
  ``scale_by_zoom_linesearch_per_eval_key`` linesearch building blocks).
"""

# Registry, construction, built-in algorithms
# Base optimizer factories / registries
# Sub-optimizer primitives shared by the strategy wrappers
# Loss helpers
# Contract types
from l2co_optimizers import (
    CONF_ABSENT,
    CONF_ESTIMATED,
    CONF_EXACT,
    CONSTRUCTOR_HYPERPARAMETERS,
    DEFAULT_HANDSHAKE_POLICY,
    FAMILIES,
    FAMILY_DISTRIBUTION,
    FAMILY_GRADIENT,
    FAMILY_POPULATION,
    GRAD_EVALUATION_WIDTH,
    HANDSHAKE_POLICY,
    LINESEARCH_FEVAL_BOUND,
    OPT_STATE_VARIANTS,
    POPULATION_HANDSHAKES,
    POPULATION_VARIANTS,
    SHADE,
    TRANSFER_OVERRIDES,
    AskFunction,
    HandshakePolicy,
    InitFunction,
    InputParameters,
    OptHistory,
    OptState,
    PopSize,
    SamplerFunction,
    StepFunction,
    StopFunction,
    SubOpt,
    TransferBundle,
    TransferReadFunction,
    TransferSpec,
    TransferWriteFunction,
    UpdateClass,
    build_transfer_fns,
    create_schedules_experimentdata,
    evosax_distribution_update,
    evosax_population_update,
    grad_sub_optimizer,
    handshake_policy_for,
    l2co_native_evosax,
    l2co_native_optax,
    lbfgs_per_eval_key,
    lbfgs_update,
    menu_loss_and_grad,
    normalized_optax_from_state,
    normalized_optax_normal,
    optax_update,
    optax_update_extra_kwargs,
    optimizer_mapping,
    optimizers,
    optstep_to_subopt,
    pop_sub_optimizer,
    population_best,
    population_best_one,
    population_handshake_for,
    random_search_update,
    register_optimizer,
    register_schedule_namer,
    resolve_popsize,
    scale_by_zoom_linesearch_per_eval_key,
    shade_popsize,
    shade_update,
    transfer_spec_for,
    vmapped_loss,
    vmapped_loss_and_grad,
    vmapped_loss_and_grad_with_rng,
    vmapped_loss_with_rng,
    write_none,
)

__all__ = [
    "AskFunction",
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
    "TransferReadFunction",
    "TransferSpec",
    "TransferWriteFunction",
    "UpdateClass",
    "build_transfer_fns",
    "create_schedules_experimentdata",
    "evosax_distribution_update",
    "evosax_population_update",
    "grad_sub_optimizer",
    "handshake_policy_for",
    "l2co_native_evosax",
    "l2co_native_optax",
    "lbfgs_per_eval_key",
    "lbfgs_update",
    "menu_loss_and_grad",
    "normalized_optax_from_state",
    "normalized_optax_normal",
    "optax_update",
    "optax_update_extra_kwargs",
    "optimizer_mapping",
    "optimizers",
    "optstep_to_subopt",
    "pop_sub_optimizer",
    "population_best",
    "population_best_one",
    "population_handshake_for",
    "random_search_update",
    "register_optimizer",
    "register_schedule_namer",
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

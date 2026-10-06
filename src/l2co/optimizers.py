"""Public API for optimizers and for *authoring* new ones.

Since l2co ADR 0016 the bare optimizers live in the sibling
``l2co-optimizers`` package; this module re-exports its surface --
the same objects, not copies -- so ``l2co.optimizers.X`` keeps working
and remains the stable import path for downstream packages (rl2co,
agentic-l2co). New code may import from ``l2co_optimizers`` directly.
Everything needed to write a new optimizer -- or a meta-optimizer that
self-registers via :func:`register_optimizer` -- is reachable here
(``docs/adr/0007``).

The registry is the exception (``docs/adr/0018``): ``optimizer_mapping``
and ``register_optimizer`` here are l2co's *Task-level* ones, not the
bare registry's. ``optimizer_mapping(name)`` returns a factory called
with ``task=``, for meta-optimizers and bare optimizers alike;
``register_optimizer`` registers a meta-optimizer into
``meta_optimizers``. The bare registry (``optimizers``, and the
factories that take ``model`` / ``loss_fn`` / ``pass_rng``) is imported
from ``l2co_optimizers`` directly.

* **Registry & construction** -- the Task-level ``optimizer_mapping``,
  the ``register_optimizer`` meta-optimizer extension point and
  ``meta_optimizers`` it fills, the naming twin
  ``register_schedule_namer`` (how a meta-optimizer's steps are named,
  ``docs/adr/0016``), the ``UpdateClass`` container every factory
  returns, and the ``create_schedules_experimentdata`` helper.
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
  factories, ``resolve_popsize`` and ``CONSTRUCTOR_HYPERPARAMETERS`` --
  the shared adapter layer both strategy wrappers dispatch across.
* **Handshake policy** -- ``HandshakePolicy``, the ``HANDSHAKE_POLICY``
  table, ``DEFAULT_HANDSHAKE_POLICY`` and ``handshake_policy_for``: which
  handshake variants an optimizer needs when control passes to it. The
  single source of truth shared by the rl2co environment and both
  deployment wrappers (``docs/adr/0009``).
* **State transfer** -- ``TransferBundle``, ``build_transfer_fns`` and
  the ``FAMILY_*`` / ``CONF_*`` constants: what one optimizer hands the
  next across a switch, and how each family reads and writes it. The
  handshake policy above decides *which population* and *whether state
  survives*; this decides *what information crosses* (``docs/adr/0013``).
* **Built-ins** -- the ``normalized_optax_normal`` registry and
  ``shade_update`` / ``shade_popsize``. The other per-library factories,
  ``SHADE`` and the L-BFGS building blocks are private to
  ``l2co-optimizers``; built-in optimizers are reached by registry name
  (the per-library ``*_mapping`` registries are imported from
  ``l2co_optimizers`` directly).
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
    HANDSHAKE_POLICY,
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
    TransferWriteFunction,
    UpdateClass,
    build_transfer_fns,
    create_schedules_experimentdata,
    grad_sub_optimizer,
    handshake_policy_for,
    menu_loss_and_grad,
    normalized_optax_normal,
    optstep_to_subopt,
    pop_sub_optimizer,
    population_best,
    population_best_one,
    register_schedule_namer,
    resolve_popsize,
    shade_popsize,
    shade_update,
    vmapped_loss,
    vmapped_loss_and_grad,
    vmapped_loss_with_rng,
)

from l2co._src.task_bridge import (
    meta_optimizers,
    optimizer_mapping,
    register_optimizer,
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
    "HANDSHAKE_POLICY",
    "HandshakePolicy",
    "InitFunction",
    "InputParameters",
    "OptHistory",
    "OptState",
    "PopSize",
    "SamplerFunction",
    "StepFunction",
    "StopFunction",
    "SubOpt",
    "TransferBundle",
    "TransferReadFunction",
    "TransferWriteFunction",
    "UpdateClass",
    "build_transfer_fns",
    "create_schedules_experimentdata",
    "grad_sub_optimizer",
    "handshake_policy_for",
    "menu_loss_and_grad",
    "meta_optimizers",
    "normalized_optax_normal",
    "optimizer_mapping",
    "optstep_to_subopt",
    "pop_sub_optimizer",
    "population_best",
    "population_best_one",
    "register_optimizer",
    "register_schedule_namer",
    "resolve_popsize",
    "shade_popsize",
    "shade_update",
    "vmapped_loss",
    "vmapped_loss_and_grad",
    "vmapped_loss_with_rng",
]

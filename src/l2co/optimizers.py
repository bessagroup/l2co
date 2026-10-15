"""Public API for optimizers, for *authoring* new ones, and for switching.

Two kinds of name live here:

* **Re-exported from ``l2co-optimizers``** (ADR 0016) -- the same
  objects, not copies -- so ``l2co.optimizers.X`` keeps working and
  remains the stable import path for downstream packages (rl2co,
  agentic-l2co). New code may import these from ``l2co_optimizers``
  directly. Everything needed to write a new optimizer -- or a
  meta-optimizer that self-registers via :func:`register_optimizer` --
  is reachable here (ADR 0007).
* **Owned by l2co** -- the Task-level registry (ADR 0018) and the
  switching layer (ADR 0019): what runs when control passes between
  optimizers. These exist only here.

The registry is l2co's (ADR 0018): ``optimizer_mapping`` and
``register_optimizer`` here are the *Task-level* ones, not the bare
registry's. ``optimizer_mapping(name)`` returns a factory called with
``task=``, for meta-optimizers and bare optimizers alike;
``register_optimizer`` registers a meta-optimizer into
``meta_optimizers``. The bare registry (``optimizers``, and the
factories that take ``model`` / ``loss_fn`` / ``pass_rng``) is imported
from ``l2co_optimizers`` directly.

Owned by l2co:

* **Registry** -- the Task-level ``optimizer_mapping``, the
  ``register_optimizer`` meta-optimizer extension point and the
  ``meta_optimizers`` it fills.
* **Sub-optimizer adapter** -- ``SubOpt`` and the
  ``grad_sub_optimizer`` / ``pop_sub_optimizer`` / ``optstep_to_subopt``
  factories: the adapter both strategy wrappers dispatch across.
  ``optstep_to_subopt`` is an optimizer's *parts* (re-exported below)
  plus its handshake.
* **Handshake policy** -- ``HandshakePolicy``, the ``HANDSHAKE_POLICY``
  table, ``DEFAULT_HANDSHAKE_POLICY``, ``handshake_policy_for`` and the
  canonical population layouts ``population_best`` /
  ``population_best_one``: which handshake an optimizer gets when
  control passes to it. The single source of truth shared by the rl2co
  environment and both deployment wrappers (ADR 0009).
* **Menu evaluation** -- ``menu_loss_and_grad``, the menu-dispatched
  evaluator that spends only the *active* optimizer's evaluation width
  instead of the menu-wide generation width (ADR 0011).
* **The switch** -- ``has_news`` and ``handshake_incoming``: whether a
  switch tells the incoming optimizer anything new, and the one
  reset-write-population sequence the rl2co environment and both
  deployment wrappers run on every switch (ADR 0022).

Re-exported from ``l2co-optimizers``:

* **Construction** -- the naming twin ``register_schedule_namer`` (how a
  meta-optimizer's steps are named, ADR 0016), the ``UpdateClass``
  container every factory returns, and the
  ``create_schedules_experimentdata`` helper.
* **Contract types** -- the ``InitFunction`` / ``StepFunction`` /
  ``StopFunction`` signatures, the handshake-side ``AskFunction`` /
  ``TransferReadFunction`` / ``TransferWriteFunction`` signatures,
  ``InputParameters`` / ``OptState`` / ``SamplerFunction`` aliases, and
  the ``PopSize`` spec.
* **Loss helpers** -- ``OptHistory`` and the ``vmapped_loss*`` helpers
  that evaluate a loss (and gradients) over a population.
* **Optimizer parts** -- ``optimizer_parts`` and the ``GradientParts`` /
  ``PopulationParts`` it returns: an optimizer unpacked into the optax
  transform or ``(init, ask, tell)`` triple a switching loop drives,
  with its popsize, family and transfer ports but no handshake.
* **State transfer** -- ``TransferBundle``, ``build_transfer_fns`` and
  the ``FAMILY_*`` / ``CONF_*`` constants: what one optimizer hands the
  next across a switch, and how each family reads and writes it. These
  are properties of the optimizer, so they stay in ``l2co-optimizers``;
  the handshake policy above decides *which population* and *whether
  state survives*, this decides *what information crosses* (ADR 0013).
* **Built-ins** -- the ``normalized_optax_normal`` registry and
  ``shade_update`` / ``shade_popsize``. The other per-library factories,
  ``SHADE``, the L-BFGS building blocks and ``resolve_popsize`` are
  private to ``l2co-optimizers``; built-in optimizers are reached by
  registry name (the per-library ``*_mapping`` registries are imported
  from ``l2co_optimizers`` directly).
"""

# Third-party
from l2co_optimizers import (
    CONF_ABSENT,
    CONF_ESTIMATED,
    CONF_EXACT,
    FAMILIES,
    FAMILY_DISTRIBUTION,
    FAMILY_GRADIENT,
    FAMILY_POPULATION,
    AskFunction,
    GradientParts,
    InitFunction,
    InputParameters,
    OptHistory,
    OptState,
    PopSize,
    PopulationParts,
    SamplerFunction,
    StepFunction,
    StopFunction,
    TransferBundle,
    TransferReadFunction,
    TransferWriteFunction,
    UpdateClass,
    build_transfer_fns,
    create_schedules_experimentdata,
    normalized_optax_normal,
    optimizer_parts,
    register_schedule_namer,
    shade_popsize,
    shade_update,
    vmapped_loss,
    vmapped_loss_and_grad,
    vmapped_loss_with_rng,
)

# Local
from l2co._src.switching.handshake_policy import (
    DEFAULT_HANDSHAKE_POLICY,
    HANDSHAKE_POLICY,
    HandshakePolicy,
    handshake_policy_for,
    population_best,
    population_best_one,
)
from l2co._src.switching.menu_eval import menu_loss_and_grad
from l2co._src.switching.sub_optimizer import (
    SubOpt,
    grad_sub_optimizer,
    optstep_to_subopt,
    pop_sub_optimizer,
)
from l2co._src.switching.switch import handshake_incoming, has_news
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
    "DEFAULT_HANDSHAKE_POLICY",
    "FAMILIES",
    "FAMILY_DISTRIBUTION",
    "FAMILY_GRADIENT",
    "FAMILY_POPULATION",
    "GradientParts",
    "HANDSHAKE_POLICY",
    "HandshakePolicy",
    "InitFunction",
    "InputParameters",
    "OptHistory",
    "OptState",
    "PopSize",
    "PopulationParts",
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
    "handshake_incoming",
    "handshake_policy_for",
    "has_news",
    "menu_loss_and_grad",
    "meta_optimizers",
    "normalized_optax_normal",
    "optimizer_mapping",
    "optimizer_parts",
    "optstep_to_subopt",
    "pop_sub_optimizer",
    "population_best",
    "population_best_one",
    "register_optimizer",
    "register_schedule_namer",
    "shade_popsize",
    "shade_update",
    "vmapped_loss",
    "vmapped_loss_and_grad",
    "vmapped_loss_with_rng",
]

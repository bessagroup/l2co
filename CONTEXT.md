# L2CO

Learning to Choose Optimizers: a meta-learner that selects an
optimizer from problem features before any evaluations, then
reassesses that choice once from the observed optimization
trajectory.

## Language

**L2CO strategy**:
The two-stage optimizer-selection scheme — a stage-0 fire picks the
initial optimizer from task features alone, then a single stage-1
fire reassesses it from the observed trajectory. Embodied by
`L2COModel`.
_Avoid_: "the model" (ambiguous with the task's model), "strategy"
unqualified (collides with the evosax sense below)

**Evosax strategy**:
An ask/eval/tell population-based algorithm object exposing
`init`/`ask`/`tell` plus `default_params`, possibly duck-typed.

**Strategy wrapper**:
An object exposing a trained selection model (an rl2co policy, or
the L2CO strategy) as a duck-typed evosax strategy, so a generic
ask/eval/tell loop can drive it.
_Avoid_: "deployment wrapper", "policy wrapper"

**Fire**:
One consultation of a selection model yielding an optimizer choice.
The L2CO strategy fires exactly twice: stage-0 at construction (task
features only) and stage-1 at the switch generation
(trajectory-aware). The rl2co policy fires every time an iteration
budget expires.
_Avoid_: "predict" (reserved for the model-method names), "query"

**Switch generation**:
The generation index at which the stage-1 fire happens:
`round(t × budget)`, where t is the stage-0 fraction. Static — known
before the run starts.

**Stage-0 fraction (t)**:
The fraction of the total iteration budget allocated to the stage-0
optimizer before the stage-1 fire.

**Generation**:
One ask/eval/tell cycle: exactly one iteration of the active
optimizer, one history entry, one budget tick.
_Avoid_: "step" (overloaded with optimizer-internal steps), "epoch"

**Handshake**:
The state transfer applied when the active optimizer changes: which
population the incoming optimizer sees, and whether its internal
state is reset or continued. Named variants: `best`, `best_one`,
`reset_opt_state`, `continue_opt_state`. Which variants a given
optimizer needs is tabulated once, in
`l2co.optimizers.HANDSHAKE_POLICY`, and shared by every consumer.
What crosses the switch besides the population — the scale carried by
each optimizer's **transfer ports** — is the optimizer's own business
and lives in `l2co-optimizers` (ADR 0019).
It is due exactly when `last_optimizer_idx >= 0` and the optimizer
about to run differs from it — optimizer identity is by *menu index*,
never by `OptimizationStep.hash`.

**Stale**:
An optimizer's own state is stale at a switch's destination when it no
longer fits there: the destination is too far from where it was, in its
own widths, or it cannot report a scale, or its shape rests on too
little evidence. A stale optimizer restarts (ADR 0022).
_Avoid_: invalid, expired, outdated

**Scale gate**:
The check that a shape's measurement scale is within a factor of the
receiving optimizer's own scale. A shape measured at a coarser scale
always passes (ADR 0022).

**Distance gate**:
The check that a switch's destination lies within a few of the shape's
own widths of the point where the shape was measured: near along
directions the shape says are wide, nearer along those it says are
narrow (ADR 0022).

**Sub-optimizer** (`SubOpt`):
One menu entry as a strategy wrapper drives it: an optimizer's
**parts** (`l2co_optimizers.optimizer_parts` — the optax transform or
`(init, ask, tell)` triple, popsize, family, transfer ports) plus the
handshake it gets on a switch. Built by `optstep_to_subopt`; the
static `is_pop` flag picks the `lax.switch` branch implementation. The
switching layer (`_src/switching/`) owns it (ADR 0019).
_Avoid_: "optimizer parts" for the adapter (the parts carry no
handshake), "sub-step"

**Menu**:
The ordered tuple of candidate optimizers (`OptimizationStep`s) a
selection model chooses from; logit index i corresponds to menu
entry i.
_Avoid_: "portfolio" (reserved for the benchmark algorithm
portfolio run during dataset creation), "optimizer list"

**Generation width**:
The fixed leading axis of every population a strategy wrapper emits:
`max(popsize)` over the menu. A property of the *menu*, constant for
a whole run — one `ask` must return one shape whichever optimizer is
active.

**Evaluation width**:
How many rows of an emitted population a generation actually
evaluates: the active optimizer's `popsize`, or 1 for a
gradient-based one (only the iterate is consumed). A property of the
*active optimizer*, and never larger than the generation width.
_Avoid_: conflating with generation width — they coincide only when
the active optimizer is the widest in the menu.

**Padding row**:
A row of an emitted population beyond the current evaluation width.
Carried as `NaN` and never evaluated; masked out of history
aggregation and best-tracking.
_Avoid_: "dummy candidate", "slack slot"

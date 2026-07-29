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
It is due exactly when `last_optimizer_idx >= 0` and the optimizer
about to run differs from it — optimizer identity is by *menu index*,
never by `OptimizationStep.hash`.

**Menu**:
The ordered tuple of candidate optimizers (`OptimizationStep`s) a
selection model chooses from; logit index i corresponds to menu
entry i.
_Avoid_: "portfolio" (reserved for the benchmark algorithm
portfolio run during dataset creation), "optimizer list"

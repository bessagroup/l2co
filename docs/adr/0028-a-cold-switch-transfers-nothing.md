---
status: accepted
---

# A cold switch transfers nothing, and the switching layer decides it

Every switch today carries the incumbent to the incoming optimizer. The
per-optimizer table (`HANDSHAKE_POLICY`, ADR 0009) only chooses *how*:
the population is `best` or `best_one`, the internal state is `reset`
or `continue`, and the transfer ports (ADR 0013) write σ and shape on
top. There was no way to say "start this optimizer as if the run had
just begun". Two questions need exactly that (issue #117): whether a
random schedule wins by time-sharing alone or by what crosses each
switch (RL2CO_paper#14), and the `h = 0` "no handshake, explore" arm of
the explore/exploit head (rl2co#68).

## Decision

**A switch has a transfer mode.** `TRANSFER_MODES = ("full", "none")`
in `l2co._src.switching.handshake_policy`, next to
`POPULATION_VARIANTS` and `OPT_STATE_VARIANTS`. `validate_transfer_mode`
rejects any other name. `DEFAULT_TRANSFER_MODE` is `"full"`, and every
caller that accepts a mode defaults to it. The mode is a tuple of string
constants, like the table's variants, so a third mode (σ only, no shape,
left to the rl2co#68 grilling) is an addition, not a breaking change.

**The mode is orthogonal to the table.** The table is a property of the
*optimizer*: can its state survive an iterate jump. The mode is a
property of the *switch*: does anything cross it at all. `full` means
"apply the table and the transfer ports exactly as before"; `none` means
"apply neither".

### What "no transfer" means

A `none` switch initialises the incoming optimizer **exactly as at the
start of a run**:

- its own `init`, with its default hyperparameters and scale;
- a starting point drawn from the run's sampler with a fresh key, by the
  same path the run uses at reset, so the state after a cold switch is
  literally `init` with that key;
- no `transfer_write_fn` (no σ or shape write), no best-point injection
  (neither `best` nor `best_one`), no `continue`. A cold switch back to
  an optimizer that ran earlier also discards that optimizer's frozen
  state: cold means cold.

Two things are *not* reset, because they belong to the run, not to the
incoming optimizer:

- the best-so-far bookkeeping, which is what the run is scored on;
- the evaluation budget. The fresh initial sample's evaluations are
  billed like any other (ADR 0026).

The mode applies only on an actual switch, the same predicate that fires
the handshake today. Re-selecting the running optimizer is not a switch
and stays a continuation whatever the mode. A schedule of only cold
switches is therefore a restart portfolio with a shared best-so-far.

### Why it lives in the switching layer

ADR 0019 puts *when* control passes and *what the incoming optimizer
sees* in l2co, and what an optimizer can describe or absorb in
`l2co-optimizers`. "Transfer nothing" is a decision about what the
incoming optimizer sees, made per switch, so it belongs with the table
and the population handshakes, not with any optimizer: no optimizer
needs a new port to be started cold, it only needs its existing `init`.
One home also gives the rl2co environment and every wrapper (l2co's and
rl2co's) the same names and the same validation.

### Where it is implemented

The names live here, exported from `l2co.optimizers`. The behaviour is
planned for the rl2co environment, in a separate change that stacks on
this one, because multi-switch schedules and the rl2co#68 action act
there (today: `HandshakeClass.handshake` and `env/state.py`'s run loop).
The intended shape is a per-switch mode argument, traced so an action
can drive it, with a per-schedule default from the configuration, and
dispatch with `lax.cond` so `full` traces as before and its
trajectories stay byte-identical. A traced mode is an integer code,
`transfer_mode_index(name)`: the mode's position in `TRANSFER_MODES`,
whose order is append-only so codes never change when a mode is added.
If the rl2co change settles on a different shape, this section is
updated with it.

Wrappers that cannot start an optimizer cold refuse `none` rather than
run a full handshake. The l2co strategy wrapper fires stage 1 once; it
accepts `transfer_mode` (default `"full"`) and raises
`NotImplementedError` on `"none"` until a consumer needs a cold stage-1
fire. The rl2co deployment wrappers (evosax, optax) live in rl2co and
take the same guard in the rl2co change.

## Consequences

- `full` is the default everywhere, so no existing trajectory changes.
- A third mode is a new constant plus its implementation in the env;
  callers that only pass `full` or `none` are unaffected.
- The strategy wrapper refuses `none` at construction rather than
  silently running a full handshake.

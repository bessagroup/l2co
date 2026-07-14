# The strategy wrapper counts handshake evaluations; nothing hidden is added

The old eager L2CO path spent function evaluations that never
appeared in the history: `reset_from_stage0` evaluated the full
handshake population purely to seed `RunState` bookkeeping (the
stock step rhythm then re-evaluated the same samples as the first
counted stage-1 iteration — a genuine double evaluation). The
ask/eval/tell wrapper cannot hide evaluations, and we decided not to
try: the handshake population is a **counted generation** (one
history entry, popsize fevals), which removes the redundant hidden
eval. Convergence curves therefore shift slightly versus data
produced by the old path — deliberate, not a regression.

On the init side the bridge adds no hidden cost of its own: its
`init_fn` performs **no evaluation** (it forwards an all-NaN fitness
to `strategy.init`, so sub-optimizers initialise around
`population[0]` — the same `samples[0]` the old path initialised
from). The only uncounted evaluation on the path is `reset`'s
bookkeeping pass over the initial samples, which is stock machinery
applied uniformly to *every* optimizer, so comparisons stay fair. An
earlier draft of this ADR planned to NaN-pad the initial population
down to the stage-0 optimizer's popsize to preserve the old path's
initial-sampling cost; that provision proved moot — no
initial-sampling history entry exists anywhere on the bridge path,
so there is nothing to pad (the wrapper's fixed `population_size` =
max menu popsize only changes the width of `reset`'s hidden,
uncounted pass).

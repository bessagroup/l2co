---
status: proposed
---

# A switch hands over shape, not scale, and only where the shape still holds

When a meta-optimizer switches optimizers, the incoming optimizer gets
the best point found so far. Since ADR 0013 it also gets the outgoing
optimizer's search **scale**: one number, the per-coordinate width that
optimizer was searching at. Nothing else the outgoing optimizer learned
crosses. In particular its **shape** never does. A shape is how much
more curved the landscape is along some directions than along others.

What happens to the incoming optimizer's *own* state is settled by two
blanket rules, and they contradict each other:

- L-BFGS and the solvers that evaluate the objective themselves are
  always reset. The reason given is that "their curvature describes the
  neighbourhood of the iterate they built it at" (ADR 0009's table).
- Every other optimizer keeps its state across any jump, because "scale
  goes stale and shape does not" (ADR 0013).

This ADR decides four things:

1. No scale crosses a switch.
2. An optimizer whose own state no longer fits where it is being sent
   is **stale**, and restarts.
3. A shape crosses only where it can be trusted, and this ADR gives the
   rule for trusting one.
4. One function in l2co does all of it, for every consumer.

It amends ADR 0013. Two other ADRs cover the rest:

- l2co-optimizers ADR 0012: the unit and layout a shape is expressed
  in, how each optimizer reads and writes one, and the properties every
  switchable optimizer must declare.
- rl2co ADR 0010: how rl2co's environment and deployment wrappers call
  the function defined here.

Nothing here is implemented. Before any of it is built, the measurements
in "Order of work" must pass.

## Why the scale stops crossing

A paired A/B (l2co_experiments#57, using the switch from rl2co#58) ran
the uniform-random meta-policy over `headroom4` on
`bbob_headroom_train`. It used 25 realizations per task, once with the
scale write and once without. Both arms made identical switches.
Without the scale write:

- the final gap to the optimum was 1.64 decades better per task on
  average (95% CI 1.22–2.08);
- 68 tasks were better, 9 worse and 23 tied;
- 1256 of 2500 runs reached 1e-8, against 931 with the write.

The losses concentrate on multimodal and high-dimensional tasks.

The readers explain why. The number each one reports means something
different:

- CR-FM-NES's σ tracks the distance still to go.
- Rprop's is its own learning rate warming up, 1000× below the distance
  at the start.
- L-BFGS's is the step it just took: 1/√d after one step, and about
  1e-32 once it has converged.

A converged provider is the common case. A policy switches away from an
optimizer that has stalled, often to escape, and the transfer then
hands the incoming optimizer the stalled optimizer's scale. CR-FM-NES
started with a scale 1e-5 to 1e-8 of the distance to the optimum turns
NaN within 50 generations.

A scale describes what an optimizer *is looking at*. It says nothing
about what the next optimizer *should* look at.

**The scale write accounts for the whole drop.** The databank's
random-policy row for the same task set (schedule `1567478`, written
2026-08-22, before the 2026-08-24/25 switching changes) appeared to
reach 1e-8 in 1472 of 2500 runs, against 1256 with the scale off. That
difference is a measurement artifact, which step 1 settled:

- the databank stores losses in float32, while the A/B harness stores
  float64;
- near a BBOB optimum (|f| ~ 10–100), one float32 step is about 1e-6, so
  a stored run within half a step of the optimum rounds onto it and
  counts as reaching 1e-8;
- on all 100 tasks the optimum rounds down in float32, so this always
  favours the stored row.

Rounding the harness runs to float32 before scoring
(l2co_experiments#57, `ab.py databank`) gives:

| | runs reaching 1e-8 |
|---|---|
| databank row | 1472 |
| scale on | 1180 |
| scale off | 1552 |

Scale off against the stored row is +0.8 runs per task (Wilcoxon
p = 0.096): within noise, since the two use different keys. Scale on
against it is −2.9 runs per task (p = 2.8e-6). The other changes in that
batch (own-ask, the repaired GA baseline, refusing non-finite bundles)
cost nothing detectable.

## What goes stale, and how

1. **The receiver's own state goes stale too.** CR-FM-NES, Rprop and
   mr15ga get the default handshake, `continue`, so they keep their
   state across any jump. Take a CR-FM-NES that collapsed to σ = 1e-8
   in one basin, was switched away, and is later sent to a better point
   in another basin. It arrives with the same tiny σ the A/B blamed on
   the scale write. Only this time the σ is its own.
2. **A shape is measured at one scale.** Curvature depends on the scale
   it is measured at. On Rastrigin the curvature of one ripple is about
   200× that of the bowl underneath. L-BFGS measures along its own
   steps and sees the ripple. CR-FM-NES samples a cloud of width σ and
   sees the bowl. A shape only holds for a search at roughly the scale
   it was measured at.
3. **A shape comes in different units.** Covariances, standard
   deviations and preconditioners each relate to curvature by a
   different power. That is l2co-optimizers ADR 0012's subject.
4. **A shape is measured somewhere.** The best point may lie far from
   where the outgoing optimizer was working, and far from where the
   incoming one last ran. On a single bowl, curvature holds everywhere.
   It does not hold along a bending valley (Rosenbrock) or between
   basins.

## Decision

### No scale crosses

The incoming optimizer keeps its own scale. The outgoing optimizer's
scale stays in the bundle only as a label: the **measurement scale** of
the shape it hands over.

### A switch with news checks whether the receiver is stale

rl2co ADR 0009 calls what a switch carries **news** when the handed-over
point is better than anything the incoming optimizer has, or when the
incoming optimizer has never run. A switch with no news writes nothing
at all: the receiver resumes exactly where it was.

On a switch with news, the receiver's own state is **stale** at the
destination if any of these is true:

- **It is too far away.** The destination lies more than c of the
  receiver's own widths from where it last was (the distance is defined
  below).
- **It cannot report a scale.** Its scale reader returns "absent"
  (RMSProp, Adagrad, Adafactor, SM3, SGD, random search; l2co-optimizers
  ADR 0012). It then has no way to show its state still fits.
- **Its shape has too little evidence.** This applies only to an
  optimizer that holds a shape.

A stale optimizer **restarts** at the destination. What a restart keeps
is declared per optimizer in l2co-optimizers (ADR 0012). By default it
is the optimizer's own `init` at the destination. TuRBO and the RBF
trust region keep their archive of evaluated points, because a measured
function value never goes stale; their own restart already keeps it.

This changes what the handshake table means, not what it says:

- a `reset` row restarts on every switch with news, as today;
- a `continue` row continues while its state is not stale, and restarts
  when it is.

An optimizer that has never run has no evidence, so it always restarts
at the destination.

### What the bundle carries

The bundle carries a **shape**, in the unit and layout l2co-optimizers
ADR 0012 defines: a diagonal plus a widest and a narrowest direction,
with the overall size removed. It travels with three tags:

- **measurement scale:** the scale it was measured at;
- **measurement point:** where it was measured;
- **evidence:** how much adaptation is behind it, per part.

### One distance, measured in the shape's own widths

Every distance in this ADR is measured the same way: the offset from
the measurement point to the destination, scaled along each direction
by that shape's own stretch times its measurement scale. This is the
Mahalanobis distance in the shape's own metric. For a neutral shape it
reduces to the RMS distance divided by the scale.

The reason: an ES adapted to a long valley can move 100σ *along* the
valley and still be within 3 of its own widths, while 100σ *across* the
valley is thousands of widths. A plain RMS distance cannot tell these
apart.

The same number is used by the distance gate, by the choice between two
shapes, and by rl2co's observation feature (rl2co ADR 0010).

### Three checks on a shape

All three are hard pass/fail checks.

- **Scale gate.** The shape's measurement scale is within a factor k of
  the receiver's scale *after* any restart. A shape measured at a
  *coarser* scale always passes, because a finer search corrects it
  within a few steps.
- **Distance gate.** The destination is within c of the shape's own
  widths of its measurement point.
- **Evidence.** Each part of the shape (the diagonal, each direction)
  needs at least **one adaptation time** of evidence, and the parts are
  judged separately. An adaptation time is the number of updates times
  the rate at which that part is replaced (l2co-optimizers ADR 0012).
  For example, CR-FM-NES at d = 1024 replaces its diagonal at about 0.02
  per generation but its direction at about 1/(3d). Its diagonal
  therefore passes after about 50 generations and its direction only
  after about 3000.

### Choosing between the incoming shape and the receiver's own

The receiver's own shape takes the same checks. Its measurement scale
and point are its *live* values: its current σ and where it is. Then:

| Incoming | Own | Use |
|---|---|---|
| passes | passes | the one closer to the destination, in its own widths; a tie goes to own |
| passes | fails | incoming |
| fails | passes | own |
| fails | fails | the **neutral shape** (no preferred direction) |

The tie rule means a switch that brings nothing better writes nothing.

The choice is made **part by part**: the diagonal, the widest direction
and the narrowest direction. Parts pass or fail separately, so a
CR-FM-NES whose diagonal has adapted but whose direction has not hands
over only its diagonal, and the receiver's own direction can stay.

### Read, restart, write

On a switch with news, the order is:

1. read the receiver's own shape, so it can compete as "own";
2. restart the receiver if it is stale (or tabulated `reset`);
3. write the winning shape into the receiver, wherever it has a slot
   for one (l2co-optimizers ADR 0012).

### Bookkeeping after a write

The switching layer keeps one evidence counter per menu member. It
does not use the optimizer's `generation_counter`, which never restarts
after a write.

- When a shape is written, the counter is set to **that shape's
  evidence**: zero for a neutral shape, the provider's for an incoming
  one, and unchanged when the member's own shape wins.
- From then on it grows by the member's own adaptation.
- A member's measurement point and scale are always its live values,
  not values stored at write time.

### One implementation, called by every consumer

The checks, the choice, the restart decision, the read–restart–write
order and the evidence counters are one function in the switching
layer. It lives here for the same reason the handshake policy and the
population builders do (ADR 0009 and its amendment). Its callers are:

- rl2co's environment and both deployment wrappers (rl2co ADR 0010),
  from step 2;
- **l2co's own strategy wrapper, from step 3.** Step 2 changes nothing
  there: L2CO carries no scale, and its one switch goes into an
  optimizer that has never run, so it always has news. It joins when the
  restart lands, which is what makes its option redundant. L2CO
  switches at most once, at the stage-1 fire, and a different pick has
  never run, so under the restart it always restarts. Its
  `opt_state_handshake` option then has nothing left to choose
  between: `"per_optimizer"` and the default `"legacy_reset"` both end
  in the same restart. The option is removed. `transfer_shape`
  is off for L2CO until a measurement on L2CO's own switch shows it
  helps. Before this lands, the default must be checked bit for bit
  against `legacy_reset` on the existing L2CO tests. If it differs (a
  different random key, say), the databank's `l2co` rows join the
  regeneration below.

### Off switches

Three independent keyword-only arguments of the shared function, passed
through by every consumer. They are never `OptimizationStep`
hyperparameters, so schedule names and hashes do not change; anything
stored in the databank runs with the defaults.

| Keyword | Default | Deleted when |
|---|---|---|
| `transfer_scale` | off from step 2 | step 3 lands, together with the scale-write code |
| `gate_own_state` | off: step 3 measured, no arm passed | step 5 is decided |
| `restart_broken` | on in rl2co since 2026-10-16 (off in the bare function, which needs `own=`) | not deleted: a correctness fix |
| `transfer_shape` | on from step 5, if step 5 lands | the final A/B is done |

### Thresholds

One global k and one global c. Both are chosen from the grid
{1, 2, 3, 5, 10} on the calibration half of both menus together. A
family (gradient, distribution, population) gets its own value only if
the gate census shows its routes significantly worse at the global
value, and the confirmation half confirms the split. Every value and
exception is recorded in this ADR with the numbers behind it.

### Deferred

- **The quadratic-consistency check.** A shape that is demonstrably
  quadratic would skip the scale gate and widen the distance gate.
  L-BFGS can test this cheaply: on a quadratic, its pairs satisfy
  sᵢᵀyⱼ = sⱼᵀyᵢ. It is brought forward into step 4 if the gate census
  shows incoming shapes rarely pass, mostly because of the scale gate.
- **Soft weights** instead of hard gates. Revisit after the oracle
  experiment.
- **Feeding evaluations across a switch.** TuRBO's archive could take
  the points other optimizers evaluated. That is a new kind of transfer.

## Order of work

Each step is measured before the next one starts. Steps 3 and 5 stay on
branches until their A/Bs decide.

1. **Explain the gap. Done (2026-10-15).** The gap was a float32
   measurement artifact, not a cost of the other August changes (see "Why
   the scale stops crossing"). Neither planned job was needed: rerunning
   the pipeline, or turning own-ask off. A side effect for every
   databank-based analysis: on BBOB, a 1e-8 target on stored losses
   means "float32-equal to the optimum", about 1e-6. Float64 runs must be
   rounded the same way before they are compared with stored rows.
2. **Close the deployment mismatch.** The shared function lands
   carrying only the news gate and "no scale", in the environment and
   both deployment wrappers. `optax_wrapper` always has news, because
   its sub-optimizers share one iterate (rl2co ADR 0010). This is
   certain and large, so it lands on `develop` and is followed by the
   first regeneration.
3. **Staleness, in two arms.** Both are compared against step 2:
   - **3a:** the own-shape gate only. A stale shape becomes neutral, but
     the member keeps its scale and does not restart.
   - **3b:** the gate plus the restart.

   Both arms log a **gate census**: for every switch, the provider's
   scale and point, the receiver's resulting scale, the distance, the
   evidence, and whether an incoming shape would pass. Counts are broken
   down by route and by smooth versus rugged function. Step 3 also gives
   the first value of c.

   | 3a | 3b | Next |
   |---|---|---|
   | passes | passes | 3b is step 4's baseline |
   | passes | fails | 3a is the baseline; the restart is dropped |
   | fails | fails | step 4's baseline is step 2, and step 5 is reconsidered: without a working own-shape gate, incoming shapes would go only into members that restart |

   From the census: if incoming shapes rarely pass and the scale gate is
   the reason, the quadratic-consistency check moves into step 4. If
   they rarely pass for every reason, step 5 is dropped for that menu
   without running the oracle.

   **Measured (2026-10-16): both arms fail.** On the calibration half
   (bessagroup/l2co_experiments#58), 3a is neutral at every c (−0.03 to
   −0.08 decades, every interval spanning zero). 3b is worse at c = 1
   (+0.48 [+0.06, +1.00]) and c = 2 (+0.35), all of it on the embedded
   suite, and the harm shrinks as c grows, to −0.09 [−0.29, +0.16] at
   c = 10. So `gate_own_state` stays off, no c is chosen, and the last
   row of the table applies. The census was not logged: the rl2co
   bridge has no hook for it yet.

   The same run found that the step-2 switch keeps a *broken* optimizer.
   Once CR-FM-NES's state goes NaN, every later visit to it is NaN from
   start to end: 80% of its NaN generations. A narrower rule was added
   and measured once on the confirmation half: `restart_broken` restarts
   an optimizer whose point, or a width it reports, is NaN or infinite
   (`OwnState.broken`), on every switch, news or not. It did no harm
   (all 80 tasks tie) and removed every visit that starts broken
   (478 → 0 on `headroom4`). It missed its pre-set bar (a 90% cut of
   "carried" NaN generations; 86.6%) only because that count also took
   in restarted visits that broke again on their own. It was switched
   on anyway, by decision after the data: resuming a NaN state never
   helps. It is on by default in rl2co's environment and deployment
   bridge. It changes dynamics, so it joins step 2 in the first
   regeneration if both land before it.
4. **Slope test, then the oracle experiment.** The slope test
   (l2co-optimizers ADR 0012) checks every reader's unit against a
   known answer. The oracle experiment then measures what an incoming
   shape could be worth.
5. **Build the shape ports and the curvature record**, then run the
   final A/B against step 3's baseline.

### The oracle experiment

*Revised 2026-10-16, after step 3.* The first version measured shapes at
the switches a uniform-random policy happens to make. Step 3 and the
reach count (bessagroup/l2co_experiments#58) showed that random
switching gives a shape almost nothing to act on: at most 1.3% of a
`headroom4` run and 5.7% of a `families10` run. That says little about a
trained policy, which could switch *in order to* use a shape: let a
gradient optimizer measure the curvature, then hand it to a fresh
evolution strategy. So the oracle now asks the question that bounds
every policy, on scripted switches: **at the best moment, does a correct
shape help the receiver enough for any policy to profit?**

**Schedule.** One scripted switch per run. L-BFGS runs from the
harness's initial sample for k iterations, k ∈ {10, 100}; both are
reported and neither is tuned. Then a fresh receiver starts at the best
point, as a first visit does today, and runs the rest of the budget
(the harness rule: 1000 · d iterations, at most 25,000).

**Receivers.** CMA-ES (full matrix), sep-CMA-ES (diagonal) and
CR-FM-NES (diagonal plus one widening direction; `headroom4`'s only
evolution strategy).

**Arms.** All keep the receiver's own default width; only its shape
differs. They share the L-BFGS phase and the receiver's keys, so every
realization is paired across arms.

| Arm | The receiver starts with |
|---|---|
| neutral | its default shape: today's behaviour |
| true | the true curvature at the switch point, at the receiver's width, written as fully as the receiver can hold it |
| true, layout | `true` projected onto the bundle's layout first (diagonal plus widest and narrowest direction); differs from `true` only for CMA-ES |
| true, L-BFGS's scale | the true curvature at the switch point, averaged at L-BFGS's last step length instead of the receiver's width: the scale problem on its own |
| L-BFGS estimate | the inverse curvature L-BFGS's memory implies (its last ≤ 10 pairs, two-loop recursion applied to the identity), projected onto the layout: what a real reader would hand over |
| true, elsewhere | the true curvature at the run's start point instead of the switch point: how much "where measured" matters |

"True curvature" keeps the definition above (Q17): the Hessian averaged
over an isotropic Gaussian of the stated width at the stated point, 16
samples on one fixed key, eigenvalues floored at 1e-6 × the largest,
inverted and normalised to determinant 1. A point Hessian is not enough:
BBOB's oscillation transform makes even the ellipsoid's Hessian
indefinite at some points.

**Tasks.** Built directly with `create_bbob_task` at d ∈ {10, 40},
instances 101 and 102. Seed 0, the policy test set's instance, is never
used. Badly conditioned, where a shape should matter: ellipsoid,
discus, bent cigar, sharp ridge, different powers, rotated Rosenbrock
and attractive sector. Controls: sphere, where there is no shape to
learn, so every arm must match `neutral` (a check on the writers); and
Rastrigin, where the local curvature is the bumps' and not the bowl's
(the scale problem). The sphere runs with k = 0, starting the receiver
at the initial sample: L-BFGS solves it outright. That is 204 cells,
each running the arms × 10 realizations.

**Measure.** For each paired realization, the log10 gap of the
receiver's *own* best (from the switch on) at 0.1, 0.3, 1, 3, 10, 30
and 100% of its budget, and the generation it first reaches each target
from 1e1 to 1e-8. The primary measure is *anytime*: the mean of those
log gaps. It is averaged per task, then over the badly conditioned
tasks, with a bootstrap interval.

**Decision, fixed before the run.** "Helps" means beating `neutral` by
at least 0.3 decades in the anytime gap, with the 95% interval clear of
zero, averaged over the badly conditioned tasks, for a given receiver.

| Result | Then |
|---|---|
| `true` does not help, for any receiver | Step 5 is closed: not even a perfect shape at the best moment pays, so no policy can profit from one. |
| `true` helps, `true, layout` does not | The layout loses the value: redesign it or close step 5. |
| `true, layout` helps, `L-BFGS estimate` does not | The readers are the bottleneck. `true, L-BFGS's scale` says whether that is the scale (a gating problem) or the estimate itself. |
| `L-BFGS estimate` helps for a receiver a menu uses | Build step 5 for that receiver, give the policy a "restart with this shape" action and the matching observation, then train with and without it. |

`true, elsewhere` against `true` gives the first value of the distance
gate c if step 5 goes ahead. The sphere control must pass, with every
arm within 0.01 decades of `neutral` in every realization, or the run is
void.

*Revised after the smoke run, before the real run (2026-10-16).* The
measure was the run's final gap. It could not see the effect: on the
10D ellipsoid every arm reached 1e-8 long before the budget ended,
although the true shape got CMA-ES there 25–30% sooner, and on the 40D
ellipsoid a fresh CR-FM-NES never got back below L-BFGS's point, so the
run's best tied every arm there. Hence the receiver's own best and the
anytime gap. The sphere's k = 0 and its 0.01-decade tolerance came from
the same smoke run. Implementation: bessagroup/l2co_experiments
`experiments/shape_oracle/`.

### How every step is judged

- **Tasks.** `bbob_headroom_train` is split **by function** into a
  calibration half (thresholds are chosen here) and a confirmation half
  (the decision run, run once). `bbob_headroom_test` is not used: it is
  rl2co's validation set (`base_rl2co_train.yaml`) and the set the ERTD
  and selection reports run on, and tuning switch design on it would
  leak into every policy number reported later.
- **Two menus.**
  - `headroom4` (lbfgs, rprop, crfmnes, mr15ga) must **pass**.
  - `families10` (cmaes, bfgs, crfmnes, sepcmaes, rprop, lbfgs, adam,
    turbo, shade, mr15ga, one per kind of shape slot or restart) must
    **do no harm**. It runs only on tasks with d ≤ 64, because
    full-covariance ES time out at 256D and 1024D. If `headroom4` passes
    and `families10` fails, the failing routes are investigated before
    anything lands.
- **Pass mark**, all at once:
  - the mean Δfinal per task (in decades) is in favour, with a 95% CI
    that excludes 0;
  - no suite (bbob, embedded, noisy) is worse with a CI that excludes 0;
  - CR-FM-NES's share of NaN generations does not rise.

  The confirmation half has half the tasks, so it may need more
  realizations to detect an effect near 0.3 decades.

### Regeneration

Post-switch dynamics change, so trained checkpoints and the databank
rows of switching schedules (the uniform-random baseline included) go
stale. Static optimizer rows are unaffected. They are regenerated at
most twice:

1. after step 2 lands;
2. once more, after whatever passes of steps 3 and 5 lands.

## Consequences

- **Supersedes part of ADR 0013:** "what is carried: scale only" and
  the scale writes. The rest of ADR 0013 stands: one reader and one
  writer per optimizer, no pairwise table, writes that rescale rather
  than overwrite. Its three confidence levels stay as labels.
- **Amends ADR 0009:** `continue` now means "continue unless stale".
  The table's rows themselves do not change.
- **Breaking for every consumer that switches.** See "Regeneration".
- **Glossary change on landing.** The *Handshake* entries in this
  repo's and rl2co's `CONTEXT.md` change when this lands. Their three
  steps become: keep or restart; what shape the incoming optimizer
  takes; which population. rl2co's *News* entry drops the carried
  scale: a handshake with no news applies nothing. *Stale*, the *scale
  gate* and the *distance gate* are added now.

## Considered options

- *Keep carrying the scale, and fix what each reader reports.* Rejected.
  Even a perfect reader reports the provider's scale, and the provider's
  scale is often what the switch is trying to leave.
- *Keep the receiver's stale scale, but floor it at a fraction of the
  jump length.* Rejected. The jump length would then be a scale crossing
  the switch.
- *Approximate a restart by writing a starting scale and a neutral shape
  into the existing state.* Rejected. It needs a new writer per optimizer
  that only approximates `init`. For CR-FM-NES, Rprop and mr15ga nothing
  learned would survive anyway.
- *Measure distance as RMS over coordinates.* Rejected. It cannot tell a
  move along a valley from a move across it.
- *Soft weights instead of hard gates.* Deferred. Two numbers are easier
  to calibrate and measure than a weighting curve.
- *Incoming always wins, or blend the two when both pass.* Rejected.
  Closer-wins needs nothing beyond what the bundle carries. Blending
  would need a way to compare evidence across different kinds of
  optimizer.
- *A full covariance in the bundle.* Still rejected, for ADR 0013's
  reasons: 8 MB at d = 1024. Also, at d = 1024 CMA-ES's covariance
  adapts at roughly 2/d² per generation, so it barely moves within the
  25000-generation budget.
- *Thresholds per family or per route.* Rejected by default. With two
  menus and about 100 routes they would overfit. Allowed only by the
  exception above.
- *One off switch with ordered levels (none / own / shape).* Rejected.
  It cannot express an incoming shape without own gating.
- *Calibrate on `bbob_headroom_test`, or on all 66 switchable optimizers
  at once.* Rejected. The first leaks into reported policy numbers. The
  second gives about 4300 routes, each too thin to read, and its
  full-covariance members stall high-dimensional runs.
- *Each consumer implements the checks itself.* Rejected. rl2co's
  deployment wrappers already lack the environment's "moves" gate
  (rl2co ADR 0010). Another copy would drift the same way.

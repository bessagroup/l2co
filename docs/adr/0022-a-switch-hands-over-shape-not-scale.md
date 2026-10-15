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

**Part of the drop is still unexplained.** The databank's random-policy
row for the same task set (schedule `1567478`, written 2026-08-22,
before the 2026-08-24/25 switching changes) reaches 1e-8 in 1472 of
2500 runs. Turning the scale off wins back 325 of the 541 runs lost; 216
are still missing. The same batch also added:

- own-ask: CR-FM-NES and the mutation GAs draw their own post-switch
  population;
- the repaired GA baseline;
- refusal of non-finite bundles;
- the "no news" gate, in the training environment only.

Step 1 below explains this gap before anything else is measured.

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

- rl2co's environment and both deployment wrappers (rl2co ADR 0010);
- **l2co's own strategy wrapper.** L2CO switches at most once, at the
  stage-1 fire, and a different pick has never run, so it always
  restarts. Its `opt_state_handshake` option then has nothing left to
  choose between: `"per_optimizer"` and the default `"legacy_reset"`
  both end in the same restart. The option is removed. `transfer_shape`
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
| `gate_own_state` | on from step 3 | step 5 has been measured |
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

1. **Explain the gap.** Two jobs on the existing A/B harness:
   - rerun the databank's own random-policy pipeline on today's code,
     with the scale on. About 931 means the gap is a code change; about
     1472 means the harness or the key is responsible.
   - run the harness with the scale off and own-ask off for crfmnes and
     the GAs. `TransferSpec.own_ask` is a static per-optimizer field, so
     the harness overrides it without touching the library.

   If own-ask turns out to cost the gap, it joins the things this plan
   changes.
2. **Close the deployment mismatch.** The shared function lands
   carrying only the news gate and "no scale", in the environment, both
   wrappers and l2co's strategy wrapper. This is certain and large, so
   it lands on `develop` and is followed by the first regeneration.
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
4. **Slope test, then the oracle experiment.** The slope test
   (l2co-optimizers ADR 0012) checks every reader's unit against a
   known answer. The oracle experiment then measures what an incoming
   shape could be worth.
5. **Build the shape ports and the curvature record**, then run the
   final A/B against step 3's baseline.

### The oracle experiment

BBOB tasks are JAX functions, so the true curvature at any point and
scale can be computed. Four arms separate the two gates:

| Arm | Curvature measured at | At scale |
|---|---|---|
| ideal | the destination | the receiver's resulting scale |
| wrong scale | the destination | the provider's |
| wrong place | the measurement point | the receiver's resulting scale |
| real reader | the measurement point | the provider's |

"True curvature at a scale" is defined as follows:

- **Cloud:** the Hessian averaged over an isotropic Gaussian centred
  where the arm says, with the width the arm says. 16 samples, with one
  fixed key shared by all arms.
- **Negative curvature:** eigenvalues are floored at 1e-6 × the largest
  before inverting. This is also the rule for any reader that meets
  negative curvature.
- **Representation:** the result is projected onto the bundle's layout
  before writing. The oracle therefore measures the best *this design*
  can do. No `headroom4` receiver can hold a full matrix, so there is
  no unprojected arm.

The cost on the calibration half (about 41k switches, about 9k of them
at d = 1024) is an estimated 75–225 CPU-hours, as an array job.

Run it on functions whose curvature varies from place to place
(Rosenbrock, attractive sector, Rastrigin) and on one where it does not
(ellipsoid). **The ideal arm must beat step 3's baseline by at least 0.3
decades per task**, or step 5 is dropped even if the effect is
statistically real. Otherwise the arms calibrate k and check c.

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
- The step-1 gap check runs on `headroom4` only.

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

# Runtime assessment triggers on an anti-luck windowed quality indicator, not best-so-far

`calculate_ert` (and the SBS/VBS selection in `_performance`) no
longer first-passages the cumulative minimum of observed losses.
The quality indicator is the worst 1% of quality values inside a
trailing **anti-luck window** of `⌈ln²(t+3)/2⌉` grid steps (Hansen
et al. 2022, IEEE TEVC 26(6), §III-A fn. 9–11): a target counts as
reached only when recent quality is *sustained* below it. Cummin
first-passage is an extreme-value statistic — on stochastic tasks
every target above the noise gate is eventually hit by pure luck
(probability → 1 with budget; measured: severe-uniform BBOB noise
fakes a 1/N convergence rate for a stalled optimizer). The window
makes chance hits superpolynomially improbable for *any* per-step
luck probability, which is why it is applied **unconditionally** —
we generally cannot know whether a task's loss is stochastic, so
assessment is safe-by-default. Measured cost on deterministic
functions (CMA-ES/DE/PSO/adam on 5-D BBOB): identical solved-target
sets, median runtime inflation 1.02–1.23× (the window lag k), so
rankings of settling optimizers are preserved.

Deliberate consequences a future reader should not "fix":

- **Randomsearch solves almost nothing** under the indicator (its
  proposals never settle). This is intended semantics, not a bug.
- **Target creation stays exempt**: `compute_task_targets` keeps
  the *raw* randomsearch quality-value quantile. There, RS is a
  landscape probe, not an assessed algorithm — its extreme-value
  reach is the information being harvested, and the raw quantile is
  stationary in budget where a cummin quantile is not (13–49×
  harder and budget-dependent, measured).
- **Window axis is the grid step**, not Hansen's evaluation count.
  On the nearest-interpolated `BUDGET_STEPS` grid this makes the
  window a roughly constant *fraction* of budget — strictly more
  conservative than the paper, uniform within a databank, slightly
  different strictness across databanks with different budgets. A
  feval-faithful window would collapse below one grid step at late
  times and lose the anti-luck guarantee where it matters.
- **Cauchy-noise functions degenerate** (10/30 of bbob_noisy): the
  one-sided +α·10³ noise offset means no schedule can demonstrate
  sustained quality below the offset from single readings, so their
  ERTDs show (near-)zero success for everyone — symmetric, hence
  zero-information rather than wrong labels. The fix is the Hansen
  layer-1 channel (a noise-free `fn_true` value per proposal
  through Task → rollout → databank), deferred to its own wave.
- **`success`/`cost` are baked at databank write time.** Existing
  stores hold cummin-based values; an in-place refresh tool
  recomputes the two variables from stored loss histories and
  stamps an `indicator` attribute. Stored per-task targets are
  untouched (see the exemption above). `indicator="cummin"` remains
  available as a kwarg for reproducing pre-switch numbers; it is
  deliberately not a Hydra-level experiment knob.

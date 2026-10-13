---
status: accepted
---

# Performance profiles are per-task runtime ratios

The ERTD shows how many targets each optimizer reaches as the budget
grows, averaged over tasks. It cannot show how an optimizer compares
with the best one *on each task*. Two optimizers with the same ERTD can
differ: one is fastest on half the tasks and hopeless on the rest,
the other is never fastest but always within a factor of three. A
Dolan–Moré performance profile shows that difference. For each task,
it divides every optimizer's cost by the cheapest optimizer's cost.
Then, for each ratio α, it plots the fraction of tasks on which the
optimizer is within a factor α of the best.

l2co has no working version. `_src/performance.py` has a
`performance_profile`, but it divides final losses, not costs, and
nothing calls it.

**Decision.** A performance profile compares optimizers' expected
runtimes on each task. Each profile is drawn at one position on every
task's own target ladder. It covers only tasks that every compared
optimizer has data for. l2co computes and draws it. l2co_experiments'
`ertd_report` only lays out the panels (see its
`docs/plans/performance-profile-report.md`).

**A problem is one task.** An optimizer's cost on a task, for a given
target, is its expected runtime there. That is the evaluations it spent
over all its runs on the task, divided by the number of runs that
reached the target. Failed runs add to the total but not to the
divisor. If no run reached the target, the cost is infinite, meaning
unsolved.
- This is the closed form of what the ERTD's restart resampling
  estimates, so it needs no resampling.
- It comes straight from the stored `cost` and `success` arrays.
- It keeps one quirk of the ERTD. A failed run's stored cost is its
  evaluations up to its last improvement plus one step, not its full
  budget (`calculate_ert`).

Runs cannot be the problem because they are not paired across
optimizers. Every optimizer's run *i* gets the same random key
(`_src/rollout/rollout_wrapper.py`, `jr.split(key, n_realizations)`).
But the starting population is drawn as `sampler(key, …, popsize)`
(l2co-optimizers, `_src/core/run_state.py`). So run *i* starts from
the same point only for optimizers with the same population size.

**Precision is a position on the task's own target ladder.** The
databank stores 51 targets per task (`compute_task_targets`). They
measure f − f\* and are log-spaced from 1e-8 up to an easiest target
anchored on what random search reaches.
- Position 0 is the easiest stored target. Position 1 is the hardest,
  1e-8.
- Position *p* uses the stored target nearest to index
  (n − 1)(1 − p). Exactly halfway, it takes the harder one, so a
  profile never asks for less than it was told to. With 51 targets,
  multiples of 0.02 are exact; 0.25 and 0.75 fall halfway and use
  indices 37 and 12.
- One profile is drawn per position.

A fixed absolute precision (f − f\* ≤ τ) would ask for a different
amount of progress on each task. On bbob the easiest target ranges
from about 5 to about 2×10⁹. A performance profile compares optimizers
within a task, so the precision should mean the same difficulty
everywhere. Position 1 is still exactly 1e-8 on every task.

**Only real optimizers compete.** The reference on each task is the
cheapest of the compared optimizers. The oracles (`vbs`, `vbs_task`,
`sbs`) are not offered.
- They pick the optimizer that reaches the most targets at the full
  budget, not the cheapest one at a given position.
- The per-run `vbs` combines different optimizers on different runs.
  It can be cheaper than all of them, which would give ratios
  below 1.

**Tasks nobody solves stay in.** Such a task has no cheapest optimizer.
It counts as unsolved for every optimizer rather than being dropped.
The right end of each curve is then the optimizer's solve rate over
the whole task set, comparable between positions and between task
groups. On bbob (576 tasks, 71 optimizers, 2026-10-08), the number of
tasks nobody solves is 30, 80, 101 and 112 at positions 0.25, 0.5,
0.75 and 1. The count is reported with every profile.

**One common task set.** If any compared optimizer has no data on a
task, the task is dropped for every optimizer. The number of tasks
each optimizer caused to be dropped is reported.
- Counting missing data as unsolved would penalize an optimizer for
  runs that never happened.
- Leaving an optimizer out of only the tasks it lacks (the ERTD's
  choice) would give each curve its own denominator. It would also pick
  the cheapest optimizer on those tasks from an incomplete field.
- The cost: on bbob_embedded with cmaes selected, the profile shrinks
  to the 64D tasks, because cmaes has no 256D or 1024D data (as of
  July 2026). The drop counts are there to make that visible.

**The ratio axis is fixed: log10, from 1 to 10⁴, then an ∞ point.** On
a 96-task bbob sample, the ratios have a median of 22–79, a 99th
percentile of 2–5×10⁴ and a maximum of 8.6×10⁵, depending on the
position. A per-panel range would let one outlier squeeze the
informative region against the left edge, and panels would stop being
comparable.
- A ratio above the cap still counts as solved.
- Each curve ends at an ∞ point at its solve rate, so the gap between
  the curve at 10⁴ and its ∞ point shows the very slow solves.
- The cap is a parameter, defaulting to 10⁴.

**Two summary numbers, no area.**
- `win_rate` is the curve's height at ratio 1. As in Dolan–Moré, an
  exact tie counts as a win for every tied optimizer.
- `solve_rate` is the height of the ∞ point.
- There is no area under the curve. It would blend speed and
  robustness into one number, which is the same averaging that hid
  budget-dependent winners in the ERTD's area. `pp_auc` also already
  names a different quantity (below).

**Interface.**
- `DataLoader.expected_runtime(positions, cost_metric="fevals")`
  returns a task × schedule × position array. NaN means no data; ∞
  means never reached.
- `performance_profile(ert, max_ratio=1e4, n_points=201)` is exported from
  `l2co.data`. It is a pure function over any subset of tasks of that
  array, so a report reads the store once and calls it once per panel.
  It returns the curve on a log10 ratio grid plus the ∞ point,
  `win_rate`, `solve_rate`, the task count, the count of tasks nobody
  solved, and the drop count per optimizer. It replaces the uncalled
  final-loss function of the same name.
- `l2co.plotting.plot_performance_profile(...)` sits next to
  `plot_ertd` and takes the same styling options (`style_by`,
  `right_labels`, `show_markers`), so profile and ERTD panels of one
  run match.

**Naming.** `pp_auc` (`_src/data/report_helpers.py`) and
l2co_experiments' `auc_pp` JSON key are final-loss ratio scores, not
performance profiles in this sense. They keep their names, because
renaming would change a JSON key that other tools read. Their
docstrings call them "final-loss ratio profiles".

**Considered and rejected:**
- *One run as the problem.* Runs are not paired across optimizers
  (above).
- *Moré–Wild's convergence test* (a fraction of the gap between the
  start value and the best value found). It needs a pass over the loss
  histories, and the best value found changes whenever the optimizer
  set does.
- *A fixed absolute precision.* Scale-dependent (above).
- *Pooling all 51 targets as separate problems*, as the ERTD does. Each
  task would count 51 times and easy targets would dominate. It would
  also break "a problem is one task".
- *Dropping tasks nobody solves.* The denominator would change between
  panels and positions.
- *Drawing oracles but keeping them out of the reference.* That would
  put curves picked by a different criterion into a cost plot.
- *An area-under-curve score.* Above.
- *Computing in l2co_experiments.* The curve styling lives in
  `l2co.plotting`, so drawing elsewhere would duplicate it.

# Policy artifacts carry training-time provenance manifests; manifest-less checkpoints are inadmissible

Learned optimizers (rl2co policies, l2co stage models) enter the databank as
schedules, comparable against static optimizers in ERTD. But a schedule's
existing identity notion — "identity + settings" — is insufficient for learned
optimizers: the same training config trained twice yields different weights,
and one training run yields different policies at different checkpoints.

**Decision.** The unit of identity is the **policy artifact**: one checkpoint
of one training run, identified by `(run_id, training_step)` where `run_id` is
minted at training start (`<experiment>-<UTC timestamp>-<short hex>`; resumed
runs keep it). A **provenance manifest** is written next to every checkpoint
*at training time* — run_id, step, the full resolved config inline (not just a
hash), per-package git SHAs, weights hash, feature-processor pointer, and the
**explicit list of training task hashes**. Checkpoint + manifest together form
the artifact. A checkpoint without a manifest is not admissible to the
databank: no backfill helper for pre-manifest checkpoints, no
`--allow-unproven` escape hatch. Enforced twice — fail-fast when the
evaluation pipeline loads the model, and as the invariant backstop in
`dump_databank`.

In the store, the schedule name stays a short human alias; identity travels in
a `provenance` variable (dims `(schedule,)`, sibling to `optimizationstep`,
shaped `{"training": <manifest(s)>, "evaluation": {git, evaluated_at}}`, with
a `components` map for composite artifacts like l2co's stage-0 + stage-1
pair). The writer enforces alias→`weights_hash` consistency on append, and the
identity hash used for raw-sidecar filenames derives from weights hash +
evaluation-relevant settings, never from file paths. An `in_training` variable
(dims `(taskID, schedule)`) is materialized from the manifest's task-hash list
at dump time, and the physical artifact (checkpoints + manifests + feature
processor) is archived into `<dataset>/policy_artifacts/<schedule>/`, making
the dataset directory self-sufficient (training dirs live on purged scratch).

The numerical record of a policy rollout is format-identical to a static
optimizer's — same variables, dims, and budget interpolation; nothing about
the consolidated store's array schema distinguishes learned from static
schedules. The switching sequence (which sub-optimizer was active when) needs
no new variable: the pre-existing `update_step` variable already stamps the
active optimizer's `OptimizationStep.hash` at every iteration (static
optimizers are simply constant at their own hash), in both the raw sidecar
and the consolidated store — per-iteration granularity, already consumed by
`plot_update_step`. An originally-planned `action` sidecar variable was
dropped as redundant with it.

**Rejected alternatives.**

- *Dump-time provenance assembly* (scrape `.hydra/` + `report.json` from the
  checkpoint's run dir): breaks whenever a checkpoint is moved or the run dir
  is cleaned; provenance silently degrades to "unknown".
- *Central policy registry*: a second store to sync and keep consistent; a
  dataset alone would no longer be self-sufficient.
- *Identity = config hash (+ seed)*: collides on retraining the same config,
  silently mixing rollouts from different weights.
- *Symbolic training-data reference only* ("re-resolve `tasks=bbob` at seed
  42"): `Task.hash` derives from cloudpickle and is not stable across library
  versions, so only the hash list recorded at training time supports exact
  in-training/held-out intersection later.
- *Config hash instead of inline config*: reintroduces run-dir archaeology for
  content questions the manifest exists to answer.

"""Audit a databank store for anomalies (NaN/inf, partial writes).

Verification of a consolidated store, pair by pair. Scattered non-finite
values (``NaN`` / ``inf``) inside the ``loss_history`` variables of a
populated ``(taskID, schedule)`` cell are *expected* -- the optimizer
simply produced them while traversing the problem -- and surface as
warnings. What signals a broken dataset, and surfaces as an error, is a
``loss_history`` variable that is entirely non-finite for a written
pair, any non-finite value in the performance variables (``success`` /
``cost``), a half-written cell (performance without loss data, or vice
versa), or a store an aborted in-place append left mid-flight (see
:mod:`l2co._src.databank.writer`). Fully-unwritten pairs are by design
``NaN`` and are only reported as completeness info.

The noise-free series ``output_min_true`` (ADR 0024) is conditional: a
store has it only when it holds a noisy task, and within such a store
only the noisy tasks' cells are written. So its absence is never a
finding, and an all-NaN cell of it is fine on its own; what is an error
is a task that has it on some populated pairs but not on others, whose
schedules would be scored on different series.

Verification reads the store with ``allow_incomplete=True`` (unlike
every other reader entry point, which refuses a mid-append store
outright -- see :func:`l2co._src.databank.reader._open_store`): auditing
a broken store is exactly this module's job, so a leftover append marker
surfaces as an ``error``-level finding instead of an exception. The same
holds for the protocol stamp (ADR 0025): a store whose stamp is missing
or names another protocol is audited as usual, and the stamp is reported
as an ``error``-level ``store.protocol`` finding.

All per-pair checks are evaluated as one batch of lazy dask reductions
over the store and computed in a single pass, so verifying a large
dataset never loads the full arrays into memory. The entry point is
:func:`verify_databank`, which returns a structured
:class:`DatabankVerificationReport`; the caller decides whether to log
it or raise on ``report.ok`` being ``False``.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Third-party
import numpy as np
import xarray as xr

from l2co._src.databank import layout
from l2co._src.databank.reader import _open_store, _resolve_task_ids
from l2co._src.databank.stamp import DATABANK_PROTOCOL, protocol_problem
from l2co._src.databank.tasksets import load_taskset

# Local
from l2co.naming import (
    BUDGET,
    COST,
    FEATURE,
    FEVALS,
    IN_TRAINING,
    ITERATIONS,
    METRIC,
    OPTIMIZATIONSTEP,
    OUTPUT_DIM,
    OUTPUT_MEAN,
    OUTPUT_MIN,
    OUTPUT_MIN_TRUE,
    OUTPUT_STD,
    PROVENANCE,
    REALIZATION,
    SCHEDULE,
    SUCCESS,
    TARGET,
    TARGET_VALUE,
    TASK_ID,
    TASK_LABELS,
    UPDATE_STEP,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

# Severity levels of a :class:`VerificationFinding`.
ERROR = "error"
WARNING = "warning"
INFO = "info"

_PAIR_DIMS = (TASK_ID, SCHEDULE)
_LOSS_VARS = (
    OUTPUT_MIN,
    OUTPUT_MEAN,
    OUTPUT_STD,
    FEVALS,
    ITERATIONS,
    UPDATE_STEP,
)
_COUNTER_VARS = (FEVALS, ITERATIONS, UPDATE_STEP)
_PERF_VARS = (SUCCESS, COST)

# Expected dim *sets* per stored variable (storage order is canonical
# ``(taskID, schedule, ...)`` but verification is order-agnostic).
_LOSS_DIMS = frozenset((TASK_ID, SCHEDULE, REALIZATION, OUTPUT_DIM, BUDGET))
_EXPECTED_DIMS: dict[str, frozenset[str]] = {
    **{v: _LOSS_DIMS for v in _LOSS_VARS},
    OUTPUT_MIN_TRUE: _LOSS_DIMS,
    SUCCESS: frozenset((TASK_ID, SCHEDULE, REALIZATION, OUTPUT_DIM, TARGET)),
    COST: frozenset(
        (TASK_ID, SCHEDULE, REALIZATION, OUTPUT_DIM, TARGET, METRIC)
    ),
    TASK_LABELS: frozenset((TASK_ID, FEATURE)),
    TARGET_VALUE: frozenset((TASK_ID, TARGET)),
    OPTIMIZATIONSTEP: frozenset((SCHEDULE,)),
    PROVENANCE: frozenset((SCHEDULE,)),
    IN_TRAINING: frozenset((TASK_ID, SCHEDULE)),
}

# Variables added by the policy-provenance schema: absent from stores
# written before it, so a missing one is an INFO (legacy store), not an
# ERROR. Their dims are still checked when present.
_OPTIONAL_VARS = frozenset((PROVENANCE, IN_TRAINING))

# Variables a store has only when its data calls for them: the
# noise-free series, present iff the store holds a noisy task (ADR
# 0024). Absent is silent; present, its dims are checked.
_CONDITIONAL_VARS = frozenset((OUTPUT_MIN_TRUE,))

# Display caps so a badly broken store still yields a readable summary.
_MAX_FINDINGS_SHOWN = 50
_MAX_MISSING_SHOWN = 20
_MAX_SAMPLE_IDS = 5


@dataclass(frozen=True)
class VerificationFinding:
    """One anomaly (or notice) found while verifying a store.

    Attributes
    ----------
    severity : str
        One of :data:`ERROR`, :data:`WARNING`, :data:`INFO`.
    check : str
        Stable identifier of the check that fired (e.g.
        ``"loss.scattered_nonfinite"``); filter on this
        programmatically.
    message : str
        Self-contained human-readable description.
    task_id : str | None
        Task hash the finding applies to (``None`` for store-level
        findings).
    schedule : str | None
        Schedule name the finding applies to (``None`` for task- or
        store-level findings).
    variable : str | None
        Data variable the finding applies to, if any.
    count : int | None
        Number of offending elements, where applicable.
    """

    severity: str
    check: str
    message: str
    task_id: str | None = None
    schedule: str | None = None
    variable: str | None = None
    count: int | None = None


@dataclass(frozen=True)
class DatabankVerificationReport:
    """Structured result of :func:`verify_databank`.

    Attributes
    ----------
    dataset_name : str
        Name of the verified dataset.
    store_path : pathlib.Path
        Path of the consolidated zarr store.
    n_tasks : int
        Number of tasks in the verified selection.
    n_schedules : int
        Number of schedules in the verified selection.
    n_pairs_expected : int
        ``n_tasks * n_schedules``.
    n_pairs_populated : int
        Cells with a non-null ``success`` (actually written pairs).
    missing_pairs : tuple[tuple[str, str], ...]
        Fully-unwritten ``(taskID, schedule)`` cells (fine by design).
    findings : tuple[VerificationFinding, ...]
        Every finding, all severities.
    """

    dataset_name: str
    store_path: Path
    n_tasks: int
    n_schedules: int
    n_pairs_expected: int
    n_pairs_populated: int
    missing_pairs: tuple[tuple[str, str], ...]
    findings: tuple[VerificationFinding, ...]

    @property
    def ok(self) -> bool:
        """``True`` when no :data:`ERROR`-level finding is present."""
        return not self.errors

    @property
    def errors(self) -> tuple[VerificationFinding, ...]:
        """The :data:`ERROR`-level findings."""
        return self._by_severity(ERROR)

    @property
    def warnings(self) -> tuple[VerificationFinding, ...]:
        """The :data:`WARNING`-level findings."""
        return self._by_severity(WARNING)

    @property
    def infos(self) -> tuple[VerificationFinding, ...]:
        """The :data:`INFO`-level findings."""
        return self._by_severity(INFO)

    def _by_severity(self, sev: str) -> tuple[VerificationFinding, ...]:
        return tuple(f for f in self.findings if f.severity == sev)

    def summary(self) -> str:
        """Multi-line human-readable summary (errors first, capped)."""
        verdict = "OK" if self.ok else "FAILED"
        lines = [
            f"verify_databank('{self.dataset_name}'): {verdict} -- "
            f"{len(self.errors)} error(s), {len(self.warnings)} "
            f"warning(s), {len(self.infos)} info; "
            f"{self.n_pairs_populated}/{self.n_pairs_expected} "
            f"pairs populated"
        ]
        for group in (self.errors, self.warnings, self.infos):
            for f in group[:_MAX_FINDINGS_SHOWN]:
                lines.append(
                    f"  [{f.severity.upper()}] {f.check}: {f.message}"
                )
            hidden = len(group) - _MAX_FINDINGS_SHOWN
            if hidden > 0:
                lines.append(f"  ... and {hidden} more")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _interior_dims(da: xr.DataArray) -> list[str]:
    """The dims of ``da`` other than ``(taskID, schedule)``."""
    return [d for d in da.dims if d not in _PAIR_DIMS]


def _n_interior(da: xr.DataArray) -> int:
    """Number of elements per ``(taskID, schedule)`` cell of ``da``."""
    return int(np.prod([da.sizes[d] for d in _interior_dims(da)]))


def _check_structure(ds: xr.Dataset) -> list[VerificationFinding]:
    """Metadata-only schema checks (variables, dims, coords)."""
    findings: list[VerificationFinding] = []
    for var, dims in _EXPECTED_DIMS.items():
        if var not in ds:
            if var in _CONDITIONAL_VARS:
                continue
            if var in _OPTIONAL_VARS:
                findings.append(
                    VerificationFinding(
                        INFO,
                        "structure.missing_optional_var",
                        f"variable '{var}' is absent (store written "
                        "before the policy-provenance schema)",
                        variable=var,
                    )
                )
                continue
            findings.append(
                VerificationFinding(
                    ERROR,
                    "structure.missing_var",
                    f"expected variable '{var}' is absent from the store",
                    variable=var,
                )
            )
        elif set(ds[var].dims) != dims:
            findings.append(
                VerificationFinding(
                    ERROR,
                    "structure.wrong_dims",
                    f"'{var}' has dims {sorted(ds[var].dims)}, expected "
                    f"{sorted(dims)}",
                    variable=var,
                )
            )
    for var in ds.data_vars:
        if var not in _EXPECTED_DIMS:
            findings.append(
                VerificationFinding(
                    INFO,
                    "structure.unexpected_var",
                    f"store holds an unexpected variable '{var}'",
                    variable=str(var),
                )
            )

    if BUDGET in ds.sizes:
        budget = np.asarray(ds[BUDGET].values, dtype=float)
        if budget.size < 3 or not np.all(np.diff(budget) > 0):
            findings.append(
                VerificationFinding(
                    ERROR,
                    "structure.budget_grid",
                    f"'{BUDGET}' coordinate must be strictly increasing "
                    f"with >= 3 points, got {budget.size}",
                )
            )
        elif budget.size != layout.BUDGET_STEPS:
            findings.append(
                VerificationFinding(
                    INFO,
                    "structure.budget_nondefault",
                    f"'{BUDGET}' grid has {budget.size} points (default "
                    f"BUDGET_STEPS is {layout.BUDGET_STEPS}); fine if the "
                    "store was created at that resolution",
                )
            )

    if METRIC in ds.coords:
        metrics = [str(m) for m in ds[METRIC].values]
        if metrics != [FEVALS, ITERATIONS]:
            findings.append(
                VerificationFinding(
                    ERROR,
                    "structure.metric_coord",
                    f"'{METRIC}' coordinate is {metrics}, expected "
                    f"{[FEVALS, ITERATIONS]}",
                )
            )

    if OPTIMIZATIONSTEP in ds:
        for name, recipe in zip(
            ds[SCHEDULE].values, ds[OPTIMIZATIONSTEP].values, strict=True
        ):
            try:
                json.loads(str(recipe))
            except (json.JSONDecodeError, TypeError):
                findings.append(
                    VerificationFinding(
                        ERROR,
                        "structure.recipe_json",
                        f"'{OPTIMIZATIONSTEP}' recipe of schedule "
                        f"'{name}' is not valid JSON",
                        schedule=str(name),
                        variable=OPTIMIZATIONSTEP,
                    )
                )

    if PROVENANCE in ds:
        for name, prov in zip(
            ds[SCHEDULE].values, ds[PROVENANCE].values, strict=True
        ):
            # Empty string == static optimizer (no provenance) -- valid.
            if not str(prov):
                continue
            try:
                json.loads(str(prov))
            except (json.JSONDecodeError, TypeError):
                findings.append(
                    VerificationFinding(
                        ERROR,
                        "structure.provenance_json",
                        f"'{PROVENANCE}' record of schedule '{name}' "
                        "is not valid JSON",
                        schedule=str(name),
                        variable=PROVENANCE,
                    )
                )
    return findings


def _check_tasksets(
    root: Path, dataset_name: str, store_taskids: list[str]
) -> list[VerificationFinding]:
    """Audit every taskset file against the full store taskID coord."""
    findings: list[VerificationFinding] = []
    rootdir = layout.databank_root(root, dataset_name)
    taskset_dir = rootdir / layout.TASKSET_DIR
    if not taskset_dir.is_dir():
        return findings
    known = set(store_taskids)
    for path in sorted(taskset_dir.glob("*.txt")):
        name = path.stem
        try:
            hashes = load_taskset(rootdir, name)
        except ValueError:
            findings.append(
                VerificationFinding(
                    WARNING,
                    "taskset.empty",
                    f"taskset '{name}' contains no task hashes",
                )
            )
            continue
        unknown = [h for h in hashes if h not in known]
        if unknown:
            sample = ", ".join(unknown[:_MAX_SAMPLE_IDS])
            findings.append(
                VerificationFinding(
                    WARNING,
                    "taskset.unknown_hash",
                    f"taskset '{name}' references {len(unknown)} task "
                    f"hash(es) absent from the store: {sample}",
                    count=len(unknown),
                )
            )
    return findings


def _pair_summaries(ds: xr.Dataset) -> xr.Dataset:
    """Reduce every check to small per-pair / per-task arrays.

    Builds all reductions lazily and computes them in **one** pass, so
    each ``(taskID, schedule)`` chunk is read once and shared across
    checks; the result is a tiny in-memory dataset over
    ``(taskID, schedule)`` (plus the per-task ``target_value`` checks).

    Non-finite means ``NaN`` or ``inf`` (``np.isfinite``). Comparisons
    against ``NaN`` are ``False``, so the violation counters
    (monotonicity, sign, integrality) skip missing entries; the
    integrality check is additionally gated on ``isfinite`` because
    ``NaN != NaN.round()`` would otherwise count as a violation.
    """
    out: dict[str, xr.DataArray] = {}

    success = ds[SUCCESS]
    out["populated"] = success.notnull().any(_interior_dims(success))

    for var in _LOSS_VARS:
        if var not in ds:
            continue
        arr = ds[var]
        interior = _interior_dims(arr)
        out[f"nonfinite__{var}"] = (~np.isfinite(arr)).sum(interior)
        out[f"allnull__{var}"] = arr.isnull().all(interior)
        out[f"anypresent__{var}"] = arr.notnull().any(interior)

    if OUTPUT_MIN_TRUE in ds:
        arr = ds[OUTPUT_MIN_TRUE]
        interior = _interior_dims(arr)
        out["true_nonfinite"] = (~np.isfinite(arr)).sum(interior)
        out["true_present"] = arr.notnull().any(interior)

    for var in _PERF_VARS:
        if var not in ds:
            continue
        arr = ds[var]
        out[f"perf_nonfinite__{var}"] = (~np.isfinite(arr)).sum(
            _interior_dims(arr)
        )

    fin = np.isfinite(success)
    interior = _interior_dims(success)
    out["success_nonbinary"] = (fin & (success != 0) & (success != 1)).sum(
        interior
    )
    if TARGET in success.dims:
        out["success_decreasing"] = (success.diff(TARGET) < 0).sum(interior)

    if OUTPUT_STD in ds:
        arr = ds[OUTPUT_STD]
        out["std_negative"] = (arr < 0).sum(_interior_dims(arr))

    for var in _COUNTER_VARS:
        if var not in ds:
            continue
        arr = ds[var]
        interior = _interior_dims(arr)
        out[f"counter_negative__{var}"] = (arr < 0).sum(interior)
        out[f"counter_noninteger__{var}"] = (
            np.isfinite(arr) & (arr != arr.round())
        ).sum(interior)
        if BUDGET in arr.dims:
            out[f"counter_decreasing__{var}"] = (arr.diff(BUDGET) < 0).sum(
                interior
            )

    if TARGET_VALUE in ds and TARGET in ds[TARGET_VALUE].dims:
        tv = ds[TARGET_VALUE]
        out["tv_nonfinite"] = (~np.isfinite(tv)).sum(TARGET)
        out["tv_decreasing"] = (tv.diff(TARGET) < 0).sum(TARGET)

    return xr.Dataset(out).compute()


def _pair_findings(
    ds: xr.Dataset, summary: xr.Dataset
) -> tuple[list[VerificationFinding], list[tuple[str, str]], int]:
    """Turn the computed per-pair summaries into findings.

    Plain-Python pass over the small computed arrays -- no further IO.
    Returns ``(findings, missing_pairs, n_populated)``.
    """
    taskids = [str(t) for t in ds[TASK_ID].values]
    schedules = [str(s) for s in ds[SCHEDULE].values]
    loss_vars = [v for v in _LOSS_VARS if v in ds]
    n_interior = {v: _n_interior(ds[v]) for v in ds.data_vars}

    def grid(name: str) -> np.ndarray | None:
        if name not in summary:
            return None
        return summary[name].transpose(TASK_ID, SCHEDULE).values

    populated = grid("populated")
    nonfinite = {v: grid(f"nonfinite__{v}") for v in loss_vars}
    allnull = {v: grid(f"allnull__{v}") for v in loss_vars}
    anypresent = {v: grid(f"anypresent__{v}") for v in loss_vars}
    perf_nonfinite = {
        v: grid(f"perf_nonfinite__{v}") for v in _PERF_VARS if v in ds
    }
    consistency = {
        "consistency.success_not_binary": (
            ERROR,
            SUCCESS,
            grid("success_nonbinary"),
            "{n} finite 'success' value(s) outside {{0, 1}}",
        ),
        "consistency.success_not_monotone": (
            ERROR,
            SUCCESS,
            grid("success_decreasing"),
            "'success' decreases {n} time(s) along 'target' (an easier "
            "target unsolved after a harder one was solved)",
        ),
        "consistency.negative_std": (
            ERROR,
            OUTPUT_STD,
            grid("std_negative"),
            "'output_std' has {n} negative value(s)",
        ),
    }
    for var in _COUNTER_VARS:
        if var not in ds:
            continue
        consistency[f"consistency.counter_negative.{var}"] = (
            ERROR,
            var,
            grid(f"counter_negative__{var}"),
            f"'{var}' has {{n}} negative value(s)",
        )
        consistency[f"consistency.counter_noninteger.{var}"] = (
            WARNING,
            var,
            grid(f"counter_noninteger__{var}"),
            f"'{var}' has {{n}} non-integer value(s)",
        )
        consistency[f"consistency.counter_decreasing.{var}"] = (
            WARNING,
            var,
            grid(f"counter_decreasing__{var}"),
            f"'{var}' decreases {{n}} time(s) along '{BUDGET}'",
        )

    true_nonfinite = grid("true_nonfinite")
    true_present = grid("true_present")

    findings: list[VerificationFinding] = []
    missing: list[tuple[str, str]] = []
    n_populated = 0
    for i, t in enumerate(taskids):
        with_true: list[str] = []
        without_true: list[str] = []
        for j, s in enumerate(schedules):
            pair = f"pair ({t}, {s})"
            if not populated[i, j]:
                if any(anypresent[v][i, j] for v in loss_vars):
                    findings.append(
                        VerificationFinding(
                            ERROR,
                            "pair.partial_perf_missing",
                            f"{pair}: loss_history data present but "
                            f"'{SUCCESS}' is entirely NaN "
                            "(interrupted/partial write)",
                            task_id=t,
                            schedule=s,
                        )
                    )
                else:
                    missing.append((t, s))
                continue

            n_populated += 1
            if loss_vars and all(allnull[v][i, j] for v in loss_vars):
                # One root cause, one finding: suppress the per-variable
                # all-non-finite errors for this pair.
                findings.append(
                    VerificationFinding(
                        ERROR,
                        "pair.partial_loss_missing",
                        f"{pair}: performance is populated but every "
                        "loss_history variable is entirely NaN "
                        "(interrupted/partial write)",
                        task_id=t,
                        schedule=s,
                    )
                )
            else:
                for var in loss_vars:
                    n_bad = int(nonfinite[var][i, j])
                    if n_bad == n_interior[var]:
                        findings.append(
                            VerificationFinding(
                                ERROR,
                                "loss.all_nonfinite",
                                f"{pair}: '{var}' is entirely non-finite",
                                task_id=t,
                                schedule=s,
                                variable=var,
                                count=n_bad,
                            )
                        )
                    elif n_bad > 0:
                        findings.append(
                            VerificationFinding(
                                WARNING,
                                "loss.scattered_nonfinite",
                                f"{pair}: '{var}' has {n_bad}/"
                                f"{n_interior[var]} non-finite values",
                                task_id=t,
                                schedule=s,
                                variable=var,
                                count=n_bad,
                            )
                        )
                if true_present is not None:
                    if true_present[i, j]:
                        with_true.append(s)
                        findings += _noise_free_findings(
                            pair,
                            t,
                            s,
                            int(true_nonfinite[i, j]),
                            n_interior[OUTPUT_MIN_TRUE],
                        )
                    else:
                        without_true.append(s)

            for var, counts in perf_nonfinite.items():
                n_bad = int(counts[i, j])
                if n_bad > 0:
                    findings.append(
                        VerificationFinding(
                            ERROR,
                            "perf.nonfinite",
                            f"{pair}: '{var}' has {n_bad} non-finite value(s)",
                            task_id=t,
                            schedule=s,
                            variable=var,
                            count=n_bad,
                        )
                    )

            for check, (sev, var, counts, msg) in consistency.items():
                if counts is None:
                    continue
                n_bad = int(counts[i, j])
                if n_bad > 0:
                    findings.append(
                        VerificationFinding(
                            sev,
                            check,
                            f"{pair}: " + msg.format(n=n_bad),
                            task_id=t,
                            schedule=s,
                            variable=var,
                            count=n_bad,
                        )
                    )

        if with_true and without_true:
            findings.append(
                VerificationFinding(
                    ERROR,
                    "task.noise_free_partial",
                    f"task {t}: the noise-free series '{OUTPUT_MIN_TRUE}' "
                    f"is on schedules {with_true} but not on "
                    f"{without_true}; their scores were computed from "
                    "different series (l2co ADR 0024)",
                    task_id=t,
                    variable=OUTPUT_MIN_TRUE,
                    count=len(without_true),
                )
            )

    if "tv_nonfinite" in summary:
        tv_nonfinite = summary["tv_nonfinite"].values
        tv_decreasing = summary["tv_decreasing"].values
        for i, t in enumerate(taskids):
            if int(tv_nonfinite[i]) > 0:
                findings.append(
                    VerificationFinding(
                        ERROR,
                        "task.target_value_nonfinite",
                        f"task {t}: '{TARGET_VALUE}' has "
                        f"{int(tv_nonfinite[i])} non-finite entries",
                        task_id=t,
                        variable=TARGET_VALUE,
                        count=int(tv_nonfinite[i]),
                    )
                )
            if int(tv_decreasing[i]) > 0:
                findings.append(
                    VerificationFinding(
                        ERROR,
                        "task.target_value_unsorted",
                        f"task {t}: '{TARGET_VALUE}' is not sorted "
                        "ascending (hardest to easiest) along "
                        f"'{TARGET}'",
                        task_id=t,
                        variable=TARGET_VALUE,
                    )
                )

    return findings, missing, n_populated


def _noise_free_findings(
    pair: str, task_id: str, schedule: str, n_bad: int, n_interior: int
) -> list[VerificationFinding]:
    """Non-finite values in a populated pair's noise-free series.

    Only called for a pair that has the series. Scattered ones are a
    warning, as for the other loss variables; an entirely non-finite
    series (present, so at least one ``inf``) an error.
    """
    if n_bad == 0:
        return []
    if n_bad == n_interior:
        severity, check = ERROR, "loss.all_nonfinite"
        message = f"{pair}: '{OUTPUT_MIN_TRUE}' is entirely non-finite"
    else:
        severity, check = WARNING, "loss.scattered_nonfinite"
        message = (
            f"{pair}: '{OUTPUT_MIN_TRUE}' has {n_bad}/{n_interior} "
            "non-finite values"
        )
    return [
        VerificationFinding(
            severity,
            check,
            message,
            task_id=task_id,
            schedule=schedule,
            variable=OUTPUT_MIN_TRUE,
            count=n_bad,
        )
    ]


def _subset(
    requested: list[str] | None,
    available: list[str],
    kind: str,
    check: str,
) -> tuple[list[str], list[VerificationFinding]]:
    """Intersect a requested id list with the store, warn on unknowns."""
    if requested is None:
        return available, []
    known = set(available)
    present = [x for x in requested if x in known]
    unknown = [x for x in requested if x not in known]
    findings = []
    if unknown:
        sample = ", ".join(unknown[:_MAX_SAMPLE_IDS])
        findings.append(
            VerificationFinding(
                WARNING,
                check,
                f"{len(unknown)} requested {kind}(s) absent from the "
                f"store: {sample}",
                count=len(unknown),
            )
        )
    if not present:
        raise ValueError(
            f"None of the requested {kind}s are present in the store."
        )
    return present, findings


def verify_databank(
    root: Path,
    dataset_name: str,
    *,
    task_ids: list[str] | None = None,
    taskset: str | None = None,
    schedules: list[str] | None = None,
) -> DatabankVerificationReport:
    """Verify a databank dataset and return a structured report.

    Audits the consolidated store of ``dataset_name`` per
    ``(taskID, schedule)`` pair. The severity model ("non-finite" means
    ``NaN`` or ``inf``):

    * **warning** -- scattered non-finite values inside the
      ``loss_history`` variables of a populated pair (expected from how
      the optimizer proceeds through the problem), non-integer or
      budget-decreasing counters, taskset/subset mismatches;
    * **error** -- a ``loss_history`` variable that is *entirely*
      non-finite for a populated pair, any non-finite value in the
      performance variables (``success`` / ``cost``), a half-written
      cell (performance without loss data or vice versa), a store an
      aborted append left mid-flight (``append.in_progress`` -- see
      :mod:`l2co._src.databank.writer`), schema violations, and
      internal inconsistencies (non-binary or target-non-monotone
      ``success``, negative ``output_std`` or counters, non-finite /
      unsorted ``target_value``, a task with the noise-free series on
      some populated pairs only -- ``task.noise_free_partial``);
    * **info** -- completeness (fully-unwritten pairs read back as
      ``NaN`` by design and are *not* errors), non-default budget-grid
      resolution.

    The checks run as one batch of lazy dask reductions computed in a
    single pass, so the full arrays are never loaded into memory.

    Parameters
    ----------
    root : pathlib.Path
        The databank base directory (the application supplies it).
    dataset_name : str
        Name of the dataset.
    task_ids : list[str] | None, optional
        Restrict verification to these task hashes. Mutually exclusive
        with ``taskset``. If both are ``None``, every task in the store
        is verified. Requested hashes absent from the store produce a
        warning finding and are skipped.
    taskset : str | None, optional
        Name of a taskset file to resolve the task hashes from.
    schedules : list[str] | None, optional
        Restrict verification to these schedule names (the
        human-readable ``OptimizationStep.name`` strings). If ``None``,
        every schedule in the store is verified.

    Returns
    -------
    DatabankVerificationReport
        Structured findings; ``report.ok`` is ``True`` when no
        error-level finding is present. The caller decides whether to
        log the report or raise.

    Raises
    ------
    FileNotFoundError
        The store does not exist (verifying a missing store is a usage
        error -- use :func:`l2co.databank.populated_pairs` to test for
        presence).
    ValueError
        Both ``task_ids`` and ``taskset`` are given, a filter list is
        empty, or no requested task/schedule is present in the store.
    """
    store = layout.zarr_store_path(root, dataset_name)
    ds = _open_store(
        root, dataset_name, allow_incomplete=True, check_stamp=False
    )

    findings: list[VerificationFinding] = []
    problem = protocol_problem(store)
    if problem is not None:
        findings.append(
            VerificationFinding(
                ERROR,
                "store.protocol",
                f"store {problem}; this code reads and writes protocol "
                f"'{DATABANK_PROTOCOL}', and its numbers are not comparable "
                "with this store's. Every other reader refuses it.",
            )
        )
    marker = layout.append_marker_path(root, dataset_name)
    if marker.exists():
        try:
            info = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError):
            info = {}
        findings.append(
            VerificationFinding(
                ERROR,
                "append.in_progress",
                f"store is marked mid-append (started "
                f"{info.get('started', 'unknown time')}, "
                f"+{info.get('incoming_tasks', '?')} task(s) / "
                f"+{info.get('incoming_schedules', '?')} schedule(s)); "
                "re-run the dump to complete it before trusting this store",
            )
        )
    if TASK_ID not in ds.coords or SCHEDULE not in ds.coords:
        findings.append(
            VerificationFinding(
                ERROR,
                "structure.missing_dim",
                f"store lacks the '{TASK_ID}' / '{SCHEDULE}' "
                "coordinates; not a databank store",
            )
        )
        return DatabankVerificationReport(
            dataset_name=dataset_name,
            store_path=store,
            n_tasks=0,
            n_schedules=0,
            n_pairs_expected=0,
            n_pairs_populated=0,
            missing_pairs=(),
            findings=tuple(findings),
        )

    all_taskids = [str(t) for t in ds[TASK_ID].values]
    all_schedules = [str(s) for s in ds[SCHEDULE].values]

    # Tasksets are audited against the *full* store, so a subset verify
    # does not flag tasks that were merely filtered out.
    findings += _check_tasksets(root, dataset_name, all_taskids)

    requested = _resolve_task_ids(
        root, dataset_name, all_taskids, task_ids, taskset
    )
    present_ids, f = _subset(
        requested if (task_ids or taskset) else None,
        all_taskids,
        "task",
        "subset.unknown_task",
    )
    findings += f
    if schedules is not None and not schedules:
        raise ValueError("'schedules' must contain at least one entry.")
    present_names, f = _subset(
        schedules, all_schedules, "schedule", "subset.unknown_schedule"
    )
    findings += f
    ds = ds.sel({TASK_ID: present_ids, SCHEDULE: present_names})

    findings += _check_structure(ds)

    n_expected = len(present_ids) * len(present_names)
    missing_pairs: list[tuple[str, str]] = []
    n_populated = 0
    if SUCCESS in ds:
        summary = _pair_summaries(ds)
        pair_findings, missing_pairs, n_populated = _pair_findings(ds, summary)
        findings += pair_findings

        missing_msg = ""
        if missing_pairs:
            shown = ", ".join(
                f"({t}, {s})" for t, s in missing_pairs[:_MAX_MISSING_SHOWN]
            )
            extra = len(missing_pairs) - _MAX_MISSING_SHOWN
            more = f" (+{extra} more)" if extra > 0 else ""
            missing_msg = f"; missing: {shown}{more}"
        findings.append(
            VerificationFinding(
                INFO,
                "completeness.summary",
                f"{n_populated}/{n_expected} (taskID, schedule) cells "
                f"populated{missing_msg}",
                count=n_populated,
            )
        )
        if n_populated == 0:
            findings.append(
                VerificationFinding(
                    WARNING,
                    "completeness.empty",
                    "no populated (taskID, schedule) cell in the "
                    "verified selection",
                )
            )

    return DatabankVerificationReport(
        dataset_name=dataset_name,
        store_path=store,
        n_tasks=len(present_ids),
        n_schedules=len(present_names),
        n_pairs_expected=n_expected,
        n_pairs_populated=n_populated,
        missing_pairs=tuple(missing_pairs),
        findings=tuple(findings),
    )

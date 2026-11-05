---
status: accepted
---

# Stores carry a protocol stamp

A store's numbers depend on the run protocol that made them: the
evaluation budget, the starting points, how noisy tasks are scored, and
the task hashes. Two stores written under different protocols are not
comparable. Nothing in a store recorded which protocol made it, so the
databank had no way to tell old and new stores apart.

That matters now. The dataset paper's protocol revision re-runs every
store, and the new stores keep their suite names (`bbob`, `cec2005`, ...)
under a new root (the re-run plan's decision D1). With nothing to tell
them apart:
- a launch pointed at the old root would read the old store as "cells
  already done";
- its dump would append new-protocol cells into an old store, or with
  `mode="replace"` delete it;
- a reader could put old and new numbers side by side.

## Decision

**Every new store carries a stamp.** `_template_create` writes
`l2co_stamp.json` inside the store, next to the append marker:

```json
{"protocol": "dataset-paper-2026-11", "l2co_version": "...", "created": "..."}
```

Only `protocol` is compared. The l2co version and creation time are
there for a reader's information: an editable install reports a stale
version, so the version is not evidence of anything.

**The protocol is a hand-set constant.** `DATABANK_PROTOCOL` in
`_src/databank/stamp.py`. Whoever changes what a stored number means
bumps it in the same change. It is not derived from a package version:
the protocol spans l2co-tasks, l2co-optimizers and l2co, and editable
installs report stale versions.

**The stamp is a plain file, not a zarr root attribute.** With xarray
2025.10.1 and zarr 3.2.1, every `Dataset.to_zarr(mode="a")` replaces the
root attributes. The writer makes such writes in `_finalize_small`, in
the grow path of an append and in `dump_databank_finalize`, so an
attribute would have to be set again after each of them. Any future
write path that forgot would then make a valid store unreadable. A file
in the store is written once and never touched again. It travels with
the store when it is copied, and `mode="replace"` deletes it along with
the store. Showing it in a report takes one call to `store_stamp`.

**What refuses a store whose stamp is missing or names another protocol**
(`StoreProtocolError`, a `RuntimeError`):
- **Every reader.** `_open_store` checks before it opens anything. So
  `populated_pairs` and `populated_iterations` refuse, and a launch
  pointed at the wrong root stops at its create step before any cell
  runs. So does every `load_databank*` loader: numbers from two
  protocols are never read together. A store without a stamp was
  written by l2co 1.8.x or earlier, and the message says to read it with
  that version. `load_loss_raw` reads the raw sidecar, not the store, so
  it checks the store's stamp itself: the sidecar was written under the
  store's protocol.
- **`mode="append"`,** checked in `_dump_template_runs` before a
  leftover rollback is restored, anything is snapshotted or the store
  grows. Nothing is written.
- **`mode="replace"`,** checked before `shutil.rmtree`. Nothing is
  deleted. An old store is removed by hand, as a deliberate step (the
  re-run plan's clean-up checklist), never as a side effect of a dump.
  Replacing a store of the same protocol works as before.

`mode="error"` still raises `FileExistsError` for any existing store.

**`verify_databank` audits any store.** It opens with the check off,
runs the full audit and reports a missing or different stamp as an
ERROR finding, `store.protocol`. Auditing stores that are not in the
expected state is verify's job, as with the mid-append marker. It also
lets old stores be inspected with current code before they are deleted.

## Consequences

- Every store that exists today has no stamp, so this code refuses all
  of them. Reading them takes l2co 1.8.x, until the paper has moved to
  the re-run's stores. This is intended: with the new l2co-tasks, the
  same code already computes task hashes that do not match the old
  stores.
- Opening a store costs one small file read.
- A store whose creation crashed between laying out the template and
  writing the stamp is refused, and has to be deleted by hand. The two
  steps are adjacent in `_template_create`.

## Not done

- **Provenance fields.** Package versions and git commits of every repo
  are left to l2co_experiments' dump, as the version-bump plan suggests.
  l2co stays generic.
- **Gap storage.** Recording whether a store holds raw losses or gaps
  f − f\* comes with that change (decision 14 of the protocol revision).
- **ERTD reports** do not copy the stamp yet.

## Rejected

- **A zarr root attribute:** wiped by every append-mode write (above).
- **Deriving the protocol from l2co's major version:** stale in editable
  installs, and it would invalidate stores on unrelated major bumps.
- **Taking the protocol from the application's config:** it would put
  the value in a config key rather than in the code that implements the
  protocol, so a stale checkout would carry a stale value.
- **Refusing only where cells are planned or written:** it would still
  stop a wrong-root launch, but rl2co's evaluation and overlay reads
  could quietly mix old and new numbers.
- **An opt-in keyword or environment variable for reading old stores:**
  a new parameter on every public loader, or hidden state.
- **Letting `mode="replace"` overwrite anything:** the one path that
  would destroy an old store without opening it.

# Optimizers stay in-repo behind a hardened authoring facade; the l2co meta-optimizer self-registers like rl2co

We considered extracting the optimizer implementations (optax/evosax
wrappers, custom L-BFGS, SHADE, random search) into a standalone
`l2co-optimizers` repository, parallel to `l2co-tasks`. We rejected
this. The motivation is only future extensibility and conceptual
hygiene — there is **no consumer that wants the base optimizers
without l2co** — and the `l2co-tasks` analogy does not hold: tasks
import zero l2co code, whereas the optimizers sit on a shared `_src`
substrate (`typing`, `opt_history`, `tree_utils`, `sampler`,
`optimizer_schedule`) that the rest of l2co also uses, and the
optimizer registry deliberately mixes base optimizers with the
`"l2co"` meta-optimizer, which points *back* into `L2COModel` and
`strategy_wrapper`. A repo split would pay the full cost of untangling
both for a benefit an internal seam already delivers. Instead we
**harden the existing subpackage boundary**, which also makes a future
repo split mechanical *if* a real external consumer ever appears.

We decided four things. **(1)** The registry cycle is broken by
evicting `UpdateClass.l2co_update` from `_src/optimizers/update_class.py`
into a new `_src/meta_optimizer.py` as an `L2COUpdateClass(UpdateClass)`
subclass — an exact mirror of rl2co's `RL2COUpdateClass` — that calls
`register_optimizer("l2co", …)` at module top level and is pulled in by
a side-effecting import in `l2co/__init__.py`. The base
`optimizer_mapping` literal loses its hardcoded `{"l2co": …}` entry, so
after this the base registry knows about *no* meta-optimizer and both
`"l2co"` and `"rl2co"` wire themselves in identically from the
models/strategy side. The lazy `L2COModel` / `strategy_wrapper` imports
inside `l2co_update` (the workaround for the circular import) become
ordinary top-level imports. **(2)** `sub_optimizer` does **not** move.
The cycle was `l2co_update → strategy_wrapper → sub_optimizer`, caused
solely by `l2co_update`; `sub_optimizer` itself imports only downward
(`optimizer_schedule`, the optax/evosax/SHADE implementations, `Task`)
and is a shared primitive that l2co's `strategy_wrapper` **and** rl2co's
`optax_wrapper`/`evosax_wrapper` both build on. Moving it to the
"strategy side" would break rl2co's re-export shim and mislocate a
primitive that is not l2co-private. **(3)** The optimizer-authoring
surface — today scattered across `l2co.optimization` (`UpdateClass`),
`l2co.naming` (contract typing aliases), `l2co.history` (`OptHistory`),
and `l2co.tree_utils`, with the `sub_optimizer` primitives and the base
step-fn factories not exposed at all — is consolidated and widened under
`l2co.optimizers`. It re-exports the registry, `UpdateClass`, the
`InitFunction`/`StepFunction`/`StopFunction`/`InputParameters`/
`SamplerFunction`/`PopSize` contract types, `OptHistory`, the
`vmapped_loss*` helpers, the `sub_optimizer` primitives (public-renamed:
`SubOpt`, `optstep_to_subopt`, `grad_sub_optimizer`, `pop_sub_optimizer`,
`l2co_native_evosax`, `resolve_popsize`, `CONSTRUCTOR_HYPERPARAMETERS`,
with underscore aliases kept as deprecated shims for one release), and
the base step-fn factories `normalized_optax_normal` /
`normalized_optax_from_state` / `random_search_update`. This
**documents the surface three consumers already reach for**, rather than
inventing one. Rejected: a minimal facade that forces `agentic-l2co` to
rewrite rather than re-path — a facade that pushes consumers back into
`_src` is not a boundary. **(4)** The boundary is enforced with two
mechanisms, each where it fits. The **cross-repo** ban — `rl2co` and
`agentic-l2co` must never import `l2co._src.*` — is a ruff `TID251`
banned-api rule in each consumer's `pyproject.toml` (repo-global, the
natural fit; catches every submodule, no false positives on the
consumers' own `_src`). The **in-repo cycle** ban — `_src/optimizers/**`
must not import `models` / `strategy_wrapper` / `meta_optimizer` — is an
AST-walk pytest guard (`test_optimizers_no_upward_imports.py`), because
ruff cannot *scope an added ban to a directory* (`per-file-ignores` only
*removes* rules); the AST guard also matches the existing
`test_experiments_no_private_imports.py` idiom. Plus a pytest registry
guard (`test_optimizer_registry_guard.py`, in the spirit of
`l2co-tasks`' `test_task_registry_guard.py`) asserting every registered
optimizer resolves through `optimizer_mapping` and that `import l2co` /
`import rl2co` register `"l2co"` / `"rl2co"`.

## Consequences

- Adding a new base optimizer is low-ceremony: implement in
  `_src/optimizers/`, register via `register_optimizer`, expose through
  `l2co.optimizers`, add a registry-guard case. Adding a new
  *meta*-optimizer follows the `L2COUpdateClass` / `RL2COUpdateClass`
  template — a self-contained, side-effect-registered module.
- Three repos change in one coordinated pass: l2co (the eviction +
  facade), `rl2co` (`update_class_wrapper.py` import block and the
  `sub_optimizer.py` re-export shim move to `l2co.optimizers`), and
  `agentic-l2co` (`schemas.py` stops deep-importing
  `optax_implementations` / `random_search` / `update_class`). Land the
  l2co facade first; migrate the consumers second; sync the editable
  path-deps.
- The wider facade freezes more as contract — `normalized_optax_*` and
  `random_search_update` are now promised stable. This is accepted as
  the price of a boundary consumers can actually live behind.
- One deprecation cycle carries the underscore→public `sub_optimizer`
  rename (l2co 1.4.0 → 1.5.0); the underscore shims are removed the
  release after.
- The `_src/optimizers` internal layout becomes free to change without
  breaking siblings, since the only supported entry point is
  `l2co.optimizers`. This is the same lift-and-shift that a future
  `l2co-optimizers` repo split would need, so the split — should a real
  external consumer ever justify it — reduces to relocating a package
  that already has a clean, enforced seam.

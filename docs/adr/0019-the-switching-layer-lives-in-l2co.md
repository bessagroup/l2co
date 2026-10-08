# The switching layer lives in l2co

ADR 0016 moved the "strategy-facing layer" into `l2co-optimizers`
together with the bare optimizers: `sub_optimizer` (`SubOpt`,
`optstep_to_subopt`), `handshake_policy`, `state_transfer` and
`menu_eval`. Its reasoning was that these hold per-optimizer properties
and contain no strategy policy. That was half right. **l2co now holds
both the bridge between tasks and optimizers (ADR 0018) and the
machinery that switches between optimizers.** `l2co-optimizers` keeps
only what an optimizer exposes to that machinery. This supersedes 0016's
placement of `sub_optimizer`, `handshake_policy` and `menu_eval`; the
rest of 0016 stands.

**The boundary.** An optimizer knows how to describe its own state as a
`TransferBundle` and how to absorb one. l2co decides *when* control
passes, *which population* the incoming optimizer sees and *whether* its
state survives, and dispatches across the menu. In code:

| Module | Home | Why |
|---|---|---|
| `state_transfer` | `l2co-optimizers`, unchanged | Every bare factory calls `build_transfer_fns` to fill `UpdateClass.transfer_read_fn` / `transfer_write_fn`. The writers need values that exist only inside a factory (the learning rate, Rprop's clip bounds, the `ravel_fn` of an evosax algorithm). These are the optimizer's own ports, and `own_ask` is one of its properties (ADR 0014). |
| `handshake_policy` | l2co (`_src/switching/`), whole | A handshake exists only where something switches. The table is keyed by optimizer name, but an absent name gets `DEFAULT_HANDSHAKE_POLICY`, so a new bare optimizer works without an entry. No bare factory imports it. |
| `menu_eval` | l2co (`_src/switching/`), whole | It references nothing optimizer-specific: it groups a tuple of `SubOpt`s by evaluation width (ADR 0011). |
| `sub_optimizer` | split | See below. |

**The split of `sub_optimizer`.** The module was two things. The
*generic* half is `SubOpt`, `grad_sub_optimizer` and
`pop_sub_optimizer`: wrap an optax transform or an `(init, ask, tell)`
triple, and attach a handshake. The *optimizer-specific* half is
`optstep_to_subopt`'s body: resolve a name against five registries,
split constructor hyperparameters from `Params`, write the evosax
closures (+inf placeholder fitness), set the line-search feval bound,
and build the transfer ports. Moving the whole module would have made
l2co import about eight names that `l2co-optimizers` keeps private
(`stock_lbfgs`, the `SHADE` / `TuRBO` / `RBFTrustRegion` classes,
`normalized_optax_from_state`, …). So the cut is a new public seam in
`l2co-optimizers`:

```python
optimizer_parts(opt_step, *, model, loss_fn, pass_rng) -> GradientParts | PopulationParts
```

`GradientParts` carries the optax transform and its feval bound;
`PopulationParts` carries `init_fn` / `ask_fn` / `tell_fn` and `own_ask`.
Both carry popsize, family, transfer ports and the step's hash. Neither
carries a handshake. l2co's `optstep_to_subopt` is now: unpack the
parts, look up `handshake_policy_for(opt_step)`, wrap.
`step_fevals` (the per-step feval count of an optax transform, ADR
0010) and `vmapped_loss_and_grad_with_rng` become public in
`l2co-optimizers` because the moved `grad_sub_optimizer` and
`menu_loss_and_grad` consume them. `resolve_popsize` and
`CONSTRUCTOR_HYPERPARAMETERS` become private; the popsize an optimizer
gets is `optimizer_parts(...).popsize`.

**Public path.** Unchanged for consumers: `SubOpt`, the handshake names
and `menu_loss_and_grad` were already exported from `l2co.optimizers`,
and rl2co already imported them from there. They change from re-exports
into l2co's own objects. `l2co.optimizers` also re-exports
`optimizer_parts`, `GradientParts` and `PopulationParts`. There is no
new public namespace. `l2co-optimizers` cannot keep compatibility
re-exports, since that would mean importing l2co.

**Enforcement.** `l2co-optimizers`' `test_switching_layer_lives_in_l2co`
fails if a switching-layer name is defined in its `src/` again. Its
`test_optimizer_parts.py` checks every name that can be unpacked against
the `UpdateClass` factory (popsize, family, `own_ask`). Here,
`tests/unit/test_sub_optimizer.py` checks that the names are defined in
`l2co._src.switching` and that the handshake comes from the table. The
handshake-policy and menu-evaluation tests moved here unchanged, with a
copy of `l2co-optimizers`' `toy_problems.py` so their exact assertions
survive.

**Consequences:**
- **Breaking** for anyone importing a switching-layer name, or
  `resolve_popsize` / `CONSTRUCTOR_HYPERPARAMETERS`, from
  `l2co_optimizers`. Within the ecosystem that was only l2co itself and
  rl2co's two `sub_optimizer` shim aliases.
- A new bare optimizer that needs a non-default handshake gets its
  entry in l2co's `HANDSHAKE_POLICY`, not next to its factory. A
  `register_handshake_policy` hook can be added here if that ever
  becomes common.
- ADRs 0003, 0009 and 0011 stand as written; their code is back in
  l2co.

**Considered and rejected:**
- *Moving `state_transfer` too.* Either `l2co-optimizers` imports l2co
  (a cycle), or `UpdateClass` loses its four handshake hooks and l2co
  rebuilds them from outside by name and hyperparameters. That second
  outside path is exactly the problem the `optimizer_parts` seam solves
  for `sub_optimizer`.
- *Moving all of `sub_optimizer` and making its private names public
  again.* It would put library internals into the public API that
  `l2co-optimizers` had just cut down.
- *Moving all of it and importing `l2co_optimizers._src`.* That breaks
  the rule that l2co imports only the `l2co_optimizers` root.
- *A first-class "schedule" concept* (a declarative optimizer sequence
  that fixed baselines and learned policies would share). Out of scope:
  this ADR only moves code. "Schedule" keeps its databank meaning, one
  `OptimizationStep`.

**Follow-up.** `optimizer_parts` and the `UpdateClass` factories are two
construction paths for one optimizer. `test_optimizer_parts.py` keeps
them in step, but collapsing them (building a `SubOpt` from an
`UpdateClass`, or the factories from the parts) would remove the
duplication. That was left out of this move.

# The _SubOpt sub-optimizer adapter is promoted from rl2co into l2co

The `_SubOpt` adapter (static optax-vs-evosax dispatch for
`lax.switch`-able optimizer menus, popsize resolution via
`optimizer_mapping`, SHADE constructor-hyperparameter routing) was
authored in `rl2co._src.sub_optimizer` but imports only l2co
internals — zero rl2co code. Both the rl2co policy wrapper and the
l2co strategy wrapper need it, and l2co cannot depend on rl2co, so
it moves to `l2co._src.optimizers.sub_optimizer`;
`rl2co._src.sub_optimizer` becomes a thin re-export shim. Rejected:
keeping a duplicated copy in each repo (guaranteed drift on the next
SHADE-style special case) and rebuilding on `UpdateClass`
(its `step_fn`s evaluate the loss internally, which violates the
ask/tell caller-evaluates contract).

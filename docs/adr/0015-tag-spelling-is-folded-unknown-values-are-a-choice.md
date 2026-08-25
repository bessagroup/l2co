# Tag spelling is folded; an unknown value is a choice

`FeatureProcessor` is built from the tasks a policy trains on and then
asked to encode tasks it was not built for. Four things a foreign task
tag can do were reachable before this change, and only one of them ended
well: an extra feature is ignored, a *silent* feature has `impute` and
`tolerate_missing`, and the two remaining ones — a value spelled
differently, and a value the vocabulary has never seen — both raised
`KeyError` from the same line. They should not share an answer.

**A spelling is not information, so folding it loses nothing.** The same
boolean reached `transform` as `True` from a live `Task.tag`, `np.True_`
from a NumPy label array, `"True"` from a store that stringified its
labels, and `1` or `1.0` from a config or a float label matrix — and only
`True` and `"true"` encoded. `_normalize_value` now folds all of them
onto the vocabulary key, and the same fold runs in both directions
between a numeric category and its string (`10` ↔ `"10"`), which is what
a store round-trip and every artifact written before vocabulary keys kept
their type produce. A task that fails to encode on a spelling difference
is a bug in the encoder, not a fact about the task.

The fold is confined to what the vocabulary itself says is foldable:
boolean spellings only where the vocabulary is a boolean one — a
categorical schema is free to list `"True"` and `"true"` as two
categories — and a numeric spelling only where it names a key that
already exists, so no fold can invent one.

**An unseen value *is* information, so discarding it is opt-in.**
`unknown_as_absent` routes an out-of-vocabulary categorical or boolean
value down the absent path: the feature's `impute` fill if it has one,
otherwise the `ABSENT` row, otherwise a raise naming both. It stays off
by default. Filling a silent feature asserts a value nobody stated;
collapsing `helmholtz` onto `ABSENT` discards one the tag stated
explicitly, and the model then reads "nothing is known about the family"
for a task whose family is known and merely foreign. Both are lies; only
the second contradicts the tag.

Turn it on when the alternative is not running at all — a trained
checkpoint on a task set whose vocabulary cannot change without
invalidating the checkpoint — and leave it off while training, where an
unseen value means the schema or the task set is wrong and the raise is
the useful outcome. Like `impute` it leaves every vocabulary and the
feature count alone, so the artifact still loads; unlike
`tolerate_missing`, which widens both.

**What it does not buy, and must be reported.** The `ABSENT` row it
routes to is only as good as the training that reached it, and on a
training set where every task carries every feature *nothing ever
indexes that row* — it keeps its initialization and contributes an
untrained vector, measurably so: the gradient of that row is exactly
zero for every batch. Absence and unknown-ness then encode as the same
arbitrary direction. The flag buys a run that completes, not a run whose
answer on foreign values means anything. Any result computed with it set
has to say so, the way an imputed one does. Making that row meaningful
needs feature dropout during training — dropping features from training
tags so the row gets gradient — which is a change to the training loop,
not to the processor, and is deliberately not part of this.

**Two smaller decisions in the same area.**

- A value that is not finite counts as absent, not as a value. `inf` is
  no more usable to a `Linear` than `nan`, and both arrive in numeric
  and string form from a dense databank label matrix. A processor with
  neither mechanism still passes an explicit null-ish value through
  untouched, so what an existing checkpoint encodes is unchanged.
- `_is_deterministic` reads a `noise` value that says nothing — absent,
  `None`, `nan`, `inf` — as deterministic, agreeing with the missing-key
  default it always documented. `nan` is how a dense label matrix
  represents "this suite does not tag noise", and the suites that do not
  tag it are the noise-free ones; before this, every task in a store
  that merely *contained* a noisy suite read as stochastic. An
  uninterpretable non-null token (`"heavy"`) still reads as stochastic:
  the tag is asserting something about noise and it is not "none", so
  guessing noise-freedom would be the wrong direction — a policy that
  trusted it would put a deterministic optimizer on a noisy loss.

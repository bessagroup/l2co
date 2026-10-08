"""Feature processing, normalization, and mixed embeddings for task inputs."""

# Standard
from __future__ import annotations

# Third-party
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (m.p.vanderschelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

Schema = dict[str, str | list]


# =============================================================================
#                                                            Value predicates
# =============================================================================
# Shared by the derived features below and by the encoder itself: both have to
# decide whether the tag actually said something about a feature.

#: Returned by :meth:`FeatureProcessor._resolve` when a feature is neither in
#: the tag nor derivable. Distinct from a tag that carries an explicit
#: null-ish value, which is absent in a different way (see :func:`_is_absent`).
_MISSING = object()


def _unwrap_scalar(val: Any) -> Any:
    """Unwrap a zero-dimensional array to the Python scalar it holds.

    Tags reach the processor from three places -- a live ``Task.tag``, a
    databank label matrix, and a config -- so the same value arrives as
    a Python object, a NumPy scalar, or a zero-dimensional JAX array
    depending on the route. Only the Python form is comparable against
    a vocabulary key or foldable to a boolean spelling, so every value
    is put in that form before anything looks at it. ``np.str_`` and
    ``np.bool_`` unwrap to ``str`` and ``bool``; a NumPy or JAX float
    unwraps to ``float``.

    Parameters
    ----------
    val : Any
        Raw value.

    Returns
    -------
    Any
        The Python scalar the value wraps, or ``val`` unchanged when it
        is not a zero-dimensional array.
    """
    if getattr(val, "ndim", None) == 0 and hasattr(val, "item"):
        return val.item()
    return val


def _is_absent(val: Any) -> bool:
    """Whether a resolved feature value carries no information.

    Covers the ways a feature goes missing. A tag may simply not have
    the key (``_MISSING``). Or it may carry a null-ish value: a
    databank store keeps one dense ``(task, feature)`` label matrix, so
    a feature only some tasks tag round-trips as ``nan`` for the rest,
    and feeding that to a continuous feature's ``Linear`` would put NaN
    through the whole network rather than fail.

    ``inf`` counts as absent for the same reason ``nan`` does -- it is
    not a value the network can use, and a fill is a better answer than
    an infinity propagating through a ``Linear``. Both are recognized
    in numeric *and* string form, since a store round-trips them as
    ``"nan"`` / ``"inf"``.

    Parameters
    ----------
    val : Any
        Resolved value, or ``_MISSING``.

    Returns
    -------
    bool
    """
    val = _unwrap_scalar(val)
    if val is _MISSING or val is None:
        return True
    if isinstance(val, str):
        token = val.strip().lower()
        if token in ("nan", "none", ""):
            return True
        try:
            return not math.isfinite(float(token))
        except ValueError:
            return False
    if isinstance(val, bool):
        return False
    try:
        return not bool(jnp.isfinite(jnp.asarray(val, dtype=float)))
    except (TypeError, ValueError):
        return False


# =============================================================================
#                                                            Derived features
# =============================================================================
# Features a schema may request that are *computed from* the task tag rather
# than stored in it. Deriving keeps ``Task.hash`` — and therefore every
# databank ``taskID`` — unchanged, so an existing store gains the feature
# without re-running a single trajectory. A tag that already carries the key
# wins over the derivation, so promoting one of these into the tag later is
# backwards compatible.


def _is_deterministic(tag: dict[str, Any]) -> bool:
    """Whether the task's loss is a deterministic function of its input.

    Read off the ``noise`` tag, which the task factories set for every
    benchmark suite: a noise *level* for the plain and embedded BBOB
    suites (``0.0`` when none is injected) and ``True`` for the
    inherently stochastic noisy suite. Values arrive either as native
    Python objects (live ``Task.tag``) or as the strings a databank
    store round-trips, so both spellings are accepted.

    A ``noise`` value that says nothing -- the key is absent, or it
    carries ``None`` / ``nan`` / ``inf`` in either form -- is read as
    deterministic. Absence is how a dense databank label matrix
    represents "this suite does not tag noise", and the suites that do
    not tag it are the noise-free ones, so this agrees with the missing
    key. An uninterpretable *non-null* token (``"heavy"``) is read as
    stochastic instead: the tag is asserting something about noise and
    it is not "none", so claiming noise-freedom would be the wrong
    direction to guess in.

    Parameters
    ----------
    tag : dict[str, Any]
        Raw task tag (or the stringified ``task_labels`` row).

    Returns
    -------
    bool
        True when the task is noise-free.
    """
    noise = tag.get("noise", _MISSING)

    if _is_absent(noise):
        return True

    noise = _unwrap_scalar(noise)

    if isinstance(noise, bool):
        return not noise

    if isinstance(noise, str):
        token = noise.strip().lower()
        if token == "false":
            return True
        if token == "true":
            return False
        try:
            return float(token) == 0.0
        except ValueError:
            return False

    try:
        return float(noise) == 0.0
    except (TypeError, ValueError):
        return True


DERIVED_FEATURES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "deterministic": _is_deterministic,
}


# =============================================================================
#                                                     Missing-feature encoding
# =============================================================================
# A schema spanning several benchmark suites asks for features that only some
# of them tag: cec2005 has no `separable`, cec2013lsgo no `unimodal`,
# quadratic neither. `FeatureProcessor.init(..., tolerate_missing=True)` makes
# that expressible by encoding absence explicitly instead of raising:
#
#   categorical / boolean -> one extra vocabulary entry, ABSENT, so "the suite
#       does not tag this" is its own embedding row rather than a value the
#       model confuses with `false`.
#   continuous            -> a companion boolean feature `<name>__present`;
#       the value slot carries 0.0 when absent and the companion says so.
#
# Both fall out of `vocabs`, which is the single source of truth for
# `MixedEmbedding.from_feature_processor` (vocabulary size per embedding,
# feature count for the projection) -- so nothing downstream changes.
#
# Off by default, and a processor built without it keeps raising KeyError on a
# missing feature. Existing stored processors deserialize their vocabs
# verbatim and are therefore unaffected.
#
# WHAT THIS DOES NOT BUY. Absence is a property of the *suite*, not of the
# task, so the missingness pattern is close to a suite identifier. A gate is
# free to route on it and thereby look like it generalizes while really having
# learned "which benchmark is this". Read any cross-suite result against that.

# =============================================================================
#                                                        Missing-feature impute
# =============================================================================
# The other way to survive a missing feature, and the one that keeps an
# ALREADY-TRAINED model usable. `tolerate_missing=True` above widens the
# encoding -- an extra vocabulary entry per categorical, a companion per
# continuous -- which changes both the per-feature embedding size and the
# feature count, and therefore the shapes `MixedEmbedding` builds. A
# checkpoint trained before the widening cannot load into it. That is the
# right trade when training a new policy on a union schema; it is no use at
# all when the question is "can the bbob-trained policy run on quadratic".
#
# Imputation answers that one instead. The processor keeps exactly the vocabs
# it was built with, and a feature the task tag does not carry is filled with
# a value the model already saw during training -- so every shape is
# unchanged and the checkpoint loads as-is.
#
# WHY NOT A SPARE INDEX. Because it fails silently. `eqx.nn.Embedding` is a
# gather, and JAX clamps out-of-bounds gathers rather than raising: index 2
# into a two-row table returns row 1. Encoding "missing" as an unused index
# would quietly tell the model `separable=true` for every task that omits it.
# An imputed value therefore has to be a real vocabulary entry, which
# `_validate_impute` checks at construction time rather than leaving to a
# lookup deep inside a jitted rollout.
#
# WHAT THIS DOES NOT BUY -- and it is a bigger caveat than the one above.
# The model is told the feature has a specific value. It has no way to
# express doubt about it, and nothing downstream records that the value was
# invented. Choose the fill so the assertion is defensible for the task
# family it will be applied to -- ideally its true value, otherwise the
# training-set marginal -- and report which features were imputed alongside
# any result, because a reader cannot recover that from the encoded vector.
#
# THE OTHER HALF OF THE CROSS-SUITE PROBLEM is a tag that disagrees with the
# training schema not by being silent about a feature -- that is this section
# -- but by giving one a value the vocabulary has never seen. That is not
# absence, so no fill fires here; see the next section.

# =============================================================================
#                                                      Unknown-value handling
# =============================================================================
# `unknown_as_absent` routes an out-of-vocabulary categorical or boolean value
# down the absent path above: the fill if the feature has one, otherwise the
# ABSENT row, otherwise a raise naming both. Off by default, and the default
# raises -- which is what a processor did before the flag existed.
#
# WHY IT IS A CHOICE AND NOT THE DEFAULT. The two disagreements are not equally
# innocent. Filling a *silent* feature asserts a value nobody stated;
# collapsing `helmholtz` onto ABSENT discards a value the tag stated
# explicitly, and the model then reads "no information about the family" for a
# task whose family is known and simply foreign. Both are lies, but only the
# second one contradicts the tag.
#
# WHEN TO TURN IT ON. When the alternative is not running at all: a trained
# checkpoint evaluated on a task set whose vocabulary you cannot change without
# invalidating the checkpoint. It keeps every shape, so the artifact still
# loads. Leave it off while *training*, where an unseen value means the schema
# or the task set is wrong and the raise is the useful outcome.
#
# WHAT THIS DOES NOT BUY. The ABSENT row it routes to is only as good as the
# training that reached it -- and on a training set where every task carries
# every feature, nothing ever indexes that row, so it keeps its initialization
# and contributes an untrained vector. Absence and unknown-ness are then both
# encoded as the same arbitrary direction. The flag buys a run that completes,
# not a run whose answer on foreign values means anything; a processor whose
# vocabulary spans both task sets is still the honest answer.

#: Vocabulary entry standing for "this feature is absent from the task tag".
ABSENT = "__absent__"

#: Suffix of the companion boolean generated for each continuous feature.
PRESENCE_SUFFIX = "__present"

#: The two entries a ``'boolean'`` feature's vocabulary is keyed by.
_BOOLEAN_KEYS = frozenset({"false", "true"})


def _is_boolean_vocab(vocab: dict[object, int]) -> bool:
    """Whether ``vocab`` is a ``'boolean'`` feature's vocabulary.

    Boolean features get the fixed two-entry vocabulary :meth:`init`
    builds (plus :data:`ABSENT` under ``tolerate_missing``), so the
    entries identify the kind. A *categorical* feature that happens to
    list ``"true"`` among other values is not one, and is left alone --
    the schema meant those strings literally.

    Parameters
    ----------
    vocab : dict[object, int]
        A feature's vocabulary.

    Returns
    -------
    bool
    """
    keys = set(vocab)
    return _BOOLEAN_KEYS <= keys and keys <= _BOOLEAN_KEYS | {ABSENT}


def _normalize_boolean(val: Any) -> Any:
    """Fold a value onto the ``"false"`` / ``"true"`` spellings.

    The same boolean reaches the processor spelled several ways
    depending on where the tag came from: ``True`` from a live
    ``Task.tag``, ``np.True_`` from a NumPy label array, ``"True"``
    from a store that stringified it, ``1`` or ``1.0`` from a config or
    from a float label matrix. All of them mean the vocabulary entry
    ``"true"``, so all of them are folded onto it -- otherwise a
    perfectly well-specified task fails to encode on a spelling
    difference.

    Anything that is not one of those spellings is returned unchanged,
    for :meth:`FeatureProcessor.transform` to treat as an unknown
    value.

    Parameters
    ----------
    val : Any
        Raw value, already unwrapped by :func:`_unwrap_scalar`.

    Returns
    -------
    Any
        ``"false"``, ``"true"``, or ``val`` unchanged.
    """
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, str):
        token = val.strip().lower()
        if token in _BOOLEAN_KEYS or token == ABSENT:
            return token
        try:
            val = float(token)
        except ValueError:
            return token
    if isinstance(val, int | float):
        if val == 0:
            return "false"
        if val == 1:
            return "true"
    return val


def _in_vocab(val: Any, vocab: dict[object, int]) -> bool:
    """Whether ``val`` is a key of ``vocab``, unhashable values included.

    Parameters
    ----------
    val : Any
        Candidate key.
    vocab : dict[object, int]
        The feature's vocabulary.

    Returns
    -------
    bool
        False for a value that cannot be a dict key at all, rather than
        raising ``TypeError`` from the membership test.
    """
    try:
        return val in vocab
    except TypeError:
        return False


def _match_number_spelling(val: Any, vocab: dict[object, int]) -> Any:
    """Fold a number onto a vocabulary key spelled the other way.

    A categorical vocabulary can be keyed by numbers -- a
    dimensionality declared as a category -- while the tag carries the
    string a store round-tripped, or the reverse: an artifact written
    before vocabulary keys kept their type has ``"10"`` where the live
    tag says ``10``. Both are the same category, so both are folded
    onto whichever spelling the vocabulary actually uses.

    Only spellings that name an existing key are returned, so this can
    never invent one.

    Parameters
    ----------
    val : Any
        Raw value, not a key of ``vocab``.
    vocab : dict[object, int]
        The feature's vocabulary.

    Returns
    -------
    Any
        A key of ``vocab``, or ``val`` unchanged when no spelling of it
        is one.
    """
    candidates: list[Any] = []
    if isinstance(val, str):
        token = val.strip()
        candidates.append(token)
        for parse in (int, float):
            try:
                candidates.append(parse(token))
            except ValueError:
                pass
    elif isinstance(val, int | float):
        candidates.append(str(val))
        if isinstance(val, float) and val.is_integer():
            candidates.append(int(val))
            candidates.append(str(int(val)))
    for candidate in candidates:
        if _in_vocab(candidate, vocab):
            return candidate
    return val


def _normalize_value(val: Any, vocab: dict[object, int]) -> Any:
    """Put a value in the form ``vocab`` is keyed by.

    Zero-dimensional arrays are unwrapped first, so a NumPy or JAX
    scalar is compared as the Python value it holds. Boolean features
    then fold onto their two spellings (see
    :func:`_normalize_boolean`), and a value that is not a key of a
    categorical vocabulary gets one more chance as a number spelled the
    other way (see :func:`_match_number_spelling`).

    Categorical *strings* are otherwise left alone: a categorical schema
    is free to declare ``"True"`` as a literal category, and
    case-folding every string would make that entry unreachable. A
    Python ``bool`` is still folded, since ``True`` cannot be a JSON
    vocabulary key.

    Parameters
    ----------
    val : Any
        Raw value.
    vocab : dict[object, int]
        The feature's vocabulary.

    Returns
    -------
    Any
        The key to look up in ``vocab``.
    """
    val = _unwrap_scalar(val)
    if _is_boolean_vocab(vocab):
        return _normalize_boolean(val)
    if isinstance(val, bool):
        return str(val).lower()
    if _in_vocab(val, vocab):
        return val
    return _match_number_spelling(val, vocab)


def _coerce_fill(val: Any) -> Any:
    """Put a fill value in plain-Python form.

    A fill that arrives as a NumPy scalar or a zero-dimensional JAX
    array is numerically what the caller meant, but storing it as-is
    puts an array inside a static field -- which Equinox warns about,
    and which stops comparing and serializing the way a plain number
    does. Scalars are unwrapped; anything else, a multi-element array
    included, is left for :func:`_validate_impute` to reject.

    Parameters
    ----------
    val : Any
        Raw fill value.

    Returns
    -------
    Any
        The Python scalar the value wraps, or ``val`` unchanged.
    """
    return _unwrap_scalar(val)


def _presence_base(
    vocabs: dict[str, dict[object, int] | None], name: str
) -> str | None:
    """The continuous feature ``name`` is the presence flag of.

    Parameters
    ----------
    vocabs : dict[str, dict[object, int] | None]
        The processor's vocabularies.
    name : str
        A key of ``vocabs``.

    Returns
    -------
    str | None
        The shadowed feature name, or ``None`` when ``name`` is an
        ordinary feature. A schema feature that genuinely ends in
        ``__present`` is only treated as a companion when the name it
        would shadow is itself a continuous feature here, so a real
        tag of that name is not hijacked.
    """
    if not name.endswith(PRESENCE_SUFFIX):
        return None
    base = name[: -len(PRESENCE_SUFFIX)]
    if vocabs.get(base, "missing") is None:
        return base
    return None


def _validate_impute(
    vocabs: dict[str, dict[object, int] | None], impute: dict[str, Any]
) -> None:
    """Reject fill values that would not survive :meth:`transform`.

    Checked when the processor is built, not when it encodes, because
    the failure it prevents is invisible at encode time: an index
    outside a vocabulary's range does not raise inside
    :class:`MixedEmbedding`, it silently clamps to the last row (see
    the module note above).

    Parameters
    ----------
    vocabs : dict[str, dict[object, int] | None]
        The processor's vocabularies.
    impute : dict[str, Any]
        Feature name to fill value.

    Raises
    ------
    KeyError
        If a fill value names a feature the processor does not encode.
    ValueError
        If a categorical or boolean fill value is not in that feature's
        vocabulary, a continuous one is not a finite number, or the
        named feature is a ``<name>__present`` companion, which
        :meth:`FeatureProcessor.transform` reads off the tag and never
        from here.
    """
    for name, val in impute.items():
        if name not in vocabs:
            raise KeyError(
                f"Cannot impute {name!r}: the processor does not encode "
                f"it (features: {sorted(vocabs)})."
            )
        base = _presence_base(vocabs, name)
        if base is not None:
            raise ValueError(
                f"Cannot impute {name!r}: it is the presence companion "
                f"of {base!r} and always reports whether the *tag* "
                f"carried a value, so a fill here would be dropped "
                f"without a word. Impute {base!r} instead -- the "
                f"companion then reads false beside the filled value, "
                f"which is the honest encoding."
            )
        vocab = vocabs[name]
        if vocab is None:
            if isinstance(val, bool) or not isinstance(val, int | float):
                raise ValueError(
                    f"Fill value for continuous feature {name!r} must be "
                    f"a number, got {val!r}."
                )
            if not math.isfinite(val):
                raise ValueError(
                    f"Fill value for continuous feature {name!r} must be "
                    f"finite, got {val!r}."
                )
            continue
        if not _in_vocab(_normalize_value(val, vocab), vocab):
            raise ValueError(
                f"Fill value {val!r} for feature {name!r} is not in its "
                f"vocabulary {sorted(map(str, vocab))}. An out-of-vocabulary "
                f"index does not raise downstream -- it clamps to the last "
                f"embedding row -- so it is rejected here instead."
            )


#: Vocabulary key types JSON can carry as a *value* and hand back
#: unchanged. A key outside this set cannot survive a round trip at all,
#: so :func:`_typed_vocab_pairs` refuses to write one.
_JSON_SCALARS = (str, bool, int, float, type(None))


def _typed_vocab_pairs(
    vocabs: dict[str, dict[object, int] | None],
) -> dict[str, list[list]]:
    """Vocabularies whose keys do not survive JSON, as ordered pairs.

    A vocabulary is stored as a JSON *object*, and JSON object keys are
    strings -- so a categorical feature keyed by the integers ``2, 3, 5,
    10`` (a dimensionality declared as a category, say) reloads keyed by
    ``"2", "3", "5", "10"`` and stops matching the integers a live
    ``Task.tag`` carries. Encoding those vocabularies a second time as
    ``[[key, index], ...]`` puts the keys in *value* position, where
    JSON keeps the difference between ``2`` and ``"2"``.

    Only the affected vocabularies are emitted, so a processor whose
    keys are all strings -- every processor built from a `'boolean'` or
    string-categorical schema, which is all of them today -- serializes
    to the bytes it always did.

    Parameters
    ----------
    vocabs : dict[str, dict[object, int] | None]
        The processor's vocabularies.

    Returns
    -------
    dict[str, list[list]]
        Feature name to ``[[key, index], ...]``, for the vocabularies
        that need it. Empty when none do.

    Raises
    ------
    ValueError
        If a vocabulary key is not a JSON scalar. Such a key cannot
        round-trip in either position, and writing the artifact anyway
        would produce one that fails to load.
    """
    typed: dict[str, list[list]] = {}
    for name, vocab in vocabs.items():
        if vocab is None:
            continue
        bad = [k for k in vocab if not isinstance(k, _JSON_SCALARS)]
        if bad:
            raise ValueError(
                f"Vocabulary of feature {name!r} is keyed by {bad!r}, "
                f"which JSON cannot carry, so the stored processor "
                f"could not be loaded back. Vocabulary keys must be "
                f"strings, numbers, booleans or None."
            )
        if any(not isinstance(k, str) for k in vocab):
            typed[name] = [[k, i] for k, i in vocab.items()]
    return typed


# =============================================================================


class FeatureProcessor(eqx.Module):
    """Encodes a raw feature dict into a flat JAX array.

    Continuous values are passed through as-is. Categorical and
    boolean values are mapped to integer indices via a per-feature
    vocabulary; booleans use the fixed mapping
    ``{"false": 0, "true": 1}``.

    A schema may also name an entry of :data:`DERIVED_FEATURES` (e.g.
    ``deterministic``), which :meth:`transform` computes from the rest
    of the tag when the key is absent.

    Boolean and categorical values are folded onto their vocabulary
    keys before lookup, so the same value spelled ``True``,
    ``np.True_``, ``"True"``, ``1`` or ``1.0`` all encode alike (see
    :func:`_normalize_value`).

    Parameters
    ----------
    vocabs : dict[str, dict[object, int] | None]
        Per-feature vocabulary mapping. ``None`` marks a continuous
        feature; a dict gives the value→index lookup for boolean or
        categorical features. Insertion order of ``vocabs`` defines
        the feature ordering used by :meth:`transform`.
    impute : dict[str, Any], optional
        Fill values for features a task tag does not carry, as
        ``{feature: value}``. Default empty.
    unknown_as_absent : bool, optional
        Encode an out-of-vocabulary categorical or boolean value as an
        absent one instead of raising. Default ``False``.

    Attributes
    ----------
    vocabs : dict[str, dict[object, int] | None]
        As above; static.
    impute : dict[str, Any]
        As above; static.
    unknown_as_absent : bool
        As above; static.
    """

    vocabs: dict[str, dict[object, int] | None] = eqx.field(static=True)
    impute: dict[str, Any] = eqx.field(static=True, default_factory=dict)
    unknown_as_absent: bool = eqx.field(static=True, default=False)

    def __post_init__(self) -> None:
        # Coerced before validation so a fill that arrived as a NumPy
        # or JAX scalar is judged -- and stored -- as the plain number
        # it stands for. Equinox's own field converter runs after this
        # hook, too late to keep an array out of a static field.
        object.__setattr__(
            self,
            "impute",
            {name: _coerce_fill(v) for name, v in self.impute.items()},
        )
        _validate_impute(self.vocabs, self.impute)

    @classmethod
    def init(
        cls,
        schema: Schema,
        *,
        tolerate_missing: bool = False,
        impute: dict[str, Any] | None = None,
        unknown_as_absent: bool = False,
    ) -> FeatureProcessor:
        """Build a FeatureProcessor from a schema.

        Parameters
        ----------
        schema : Schema
            Mapping of feature names to type specs:
              - ``'continuous'`` : float, passed through as-is
              - ``'boolean'``    : bool, mapped to 0 or 1
              - iterable         : categorical, gives the full
                vocabulary in the desired index order
        tolerate_missing : bool, optional
            Encode a feature the task tag does not carry instead of
            raising. Adds an :data:`ABSENT` entry to every categorical
            and boolean vocabulary, and a companion
            ``<name>__present`` boolean beside every continuous
            feature. Default ``False``, which keeps
            :meth:`transform` raising ``KeyError`` on a missing
            feature. Turn it on for a schema spanning benchmark suites
            that tag different things -- and see the module-level note
            on why the resulting missingness pattern is close to a
            suite identifier.
        impute : dict[str, Any] | None, optional
            Fill values for features a task tag may not carry, as
            ``{feature: value}``. Unlike ``tolerate_missing`` this
            leaves every vocabulary and the feature count untouched, so
            a model trained against this schema still loads --- the
            absent feature is simply encoded as ``value``. Each value
            must already be in its feature's vocabulary (or be a finite
            number, for a continuous feature), and no entry may name a
            ``<name>__present`` companion -- ``transform`` reads those
            off the tag, so a fill there would go nowhere. Anything
            else is rejected here. Default ``None``.
        unknown_as_absent : bool, optional
            Treat a categorical or boolean value that is *not* in its
            vocabulary as an absent one -- filled from ``impute`` if
            that feature has a fill, else encoded as :data:`ABSENT` if
            the vocabulary has that entry, else raising a message that
            names both. Like ``impute`` this leaves every vocabulary
            and the feature count untouched, so a trained model still
            loads. Default ``False``, which keeps :meth:`transform`
            raising on an unseen value -- see the module note on why
            that is the default.

        Returns
        -------
        FeatureProcessor
        """
        vocabs: dict[str, dict[object, int] | None] = {}

        for name, spec in schema.items():
            # Guarded on ``str`` because an array-valued spec answers an
            # equality test elementwise, and ``if`` on the result raises
            # before the categorical branch is ever reached.
            is_type_name = isinstance(spec, str)
            if is_type_name and spec == "continuous":
                vocabs[name] = None
                if tolerate_missing:
                    # Directly after its own feature, so the encoded
                    # vector stays readable in schema order.
                    vocabs[name + PRESENCE_SUFFIX] = {"false": 0, "true": 1}
            elif is_type_name and spec == "boolean":
                vocabs[name] = {"false": 0, "true": 1}
            elif hasattr(spec, "__iter__"):
                # Unwrapped so a spec that arrived as a NumPy array is
                # keyed by the Python values it holds: a ``np.int64``
                # key is not something JSON can carry, and a vocabulary
                # that cannot be stored is one no trained model can be
                # reloaded against.
                categories = [_unwrap_scalar(v) for v in spec]
                vocabs[name] = {v: i for i, v in enumerate(categories)}
            else:
                raise ValueError(
                    f"Unknown feature spec for {name!r}: {spec!r}. "
                    f"Expected 'continuous', 'boolean', or an iterable "
                    f"of categories."
                )

            if tolerate_missing and vocabs[name] is not None:
                vocabs[name][ABSENT] = len(vocabs[name])

        return cls(
            vocabs=vocabs,
            impute=dict(impute or {}),
            unknown_as_absent=unknown_as_absent,
        )

    def transform(self, x: dict[str, Any]) -> jax.Array:
        """Encode a feature dict to a default-precision JAX array.

        Parameters
        ----------
        x : dict[str, Any]
            Raw feature dict. Every key in ``vocabs`` must either be
            present, name an entry of :data:`DERIVED_FEATURES` (which
            is computed from ``x`` when the key is absent), or be
            covered by :attr:`impute` / ``tolerate_missing``. Keys of
            ``x`` the processor does not encode are ignored, so a tag
            richer than the schema passes through unchanged.

        Returns
        -------
        jax.Array
            Shape (num_features,); dtype follows JAX's default float
            precision (``float32`` by default, ``float64`` when
            ``jax_enable_x64`` is set).

        Raises
        ------
        KeyError
            If a feature is neither present in ``x``, nor derivable,
            nor encodable as absent (see :meth:`init`'s ``impute`` and
            ``tolerate_missing``); or if a categorical feature carries
            a value outside its vocabulary and the processor was not
            built with ``unknown_as_absent`` (see the note below).

        Notes
        -----
        When a processor carries both an :attr:`impute` entry and an
        ``ABSENT`` vocabulary entry for the same feature, the fill wins
        --- it is the more specific of the two. A ``<name>__present``
        companion still reports whether the *tag* carried a value, so
        it reads ``false`` even where the value slot was filled.

        A tag can disagree with the schema a second way: it can carry
        a categorical feature with a value the vocabulary has never
        seen. That is not absence, so no fill fires by default, and the
        encoding stops with a ``KeyError`` naming the feature and its
        vocabulary. ``unknown_as_absent`` routes it down the absent
        path instead -- the fill if there is one, else the ``ABSENT``
        row -- which keeps an already-trained checkpoint usable on a
        foreign task set at the cost of telling the model nothing about
        a feature the tag did state. See the module note on
        unknown-value handling.

        A value that is not finite -- ``nan`` or ``inf``, in numeric or
        string form -- counts as absent rather than as a value, so it
        takes the fill and reads ``false`` on a presence companion. A
        dense databank label matrix stores an untagged feature as
        ``nan``, and an infinity is not something a ``Linear`` can use.
        """
        feats = []
        for name, vocab in self.vocabs.items():
            base = _presence_base(self.vocabs, name)
            if base is not None:
                # Companion flag: does the continuous feature it shadows
                # actually carry a value for this task?
                resolved = self._resolve(x, base)
                feats.append(
                    vocab["false" if _is_absent(resolved) else "true"]
                )
                continue

            val = self._resolve(x, name)

            if _is_absent(val):
                if name in self.impute:
                    # Fill, then fall through to the ordinary encoding
                    # path below. Checked first so that a processor
                    # carrying both mechanisms answers with the specific
                    # value rather than the generic "unknown" -- the
                    # caller asked for this feature by name.
                    val = self.impute[name]
                elif vocab is not None and ABSENT in vocab:
                    feats.append(vocab[ABSENT])
                    continue
                elif vocab is None and name + PRESENCE_SUFFIX in self.vocabs:
                    # The companion flag carries the information; the
                    # value slot only has to be finite and constant.
                    feats.append(0.0)
                    continue
                elif val is _MISSING:
                    raise KeyError(
                        f"Feature {name!r} is absent from the task tag "
                        f"and is not derivable (derived features: "
                        f"{sorted(DERIVED_FEATURES)}). Pass "
                        f"impute={{{name!r}: ...}} to fill it with a "
                        f"fixed value, or build the processor with "
                        f"tolerate_missing=True to encode it as absent."
                    )
                # A processor with neither mechanism keeps its old
                # behaviour for an explicit null-ish value: pass it
                # through untouched rather than change what existing
                # checkpoints encode.

            if vocab is None:
                feats.append(val)
            else:
                key = _normalize_value(val, vocab)
                if not _in_vocab(key, vocab):
                    key = self._resolve_unknown(name, val, vocab)
                feats.append(vocab[key])
        return jnp.array(feats, dtype=float)

    def _resolve_unknown(
        self, name: str, val: Any, vocab: dict[object, int]
    ) -> Any:
        """The vocabulary key to encode an out-of-vocabulary value as.

        Reached when the tag *does* carry the feature but gives it a
        value the vocabulary has never seen -- which no fill covers by
        default, since imputation stands in for a feature the tag omits
        rather than for one it contradicts.

        Parameters
        ----------
        name : str
            Feature name.
        val : Any
            The value the tag carried.
        vocab : dict[object, int]
            The feature's vocabulary.

        Returns
        -------
        Any
            A key of ``vocab``, when :attr:`unknown_as_absent` is set
            and the feature has either a fill or an :data:`ABSENT`
            entry to fall back on.

        Raises
        ------
        KeyError
            Otherwise -- naming the feature, its vocabulary, and the
            two ways to make the value encodable.
        """
        if self.unknown_as_absent:
            if name in self.impute:
                return _normalize_value(self.impute[name], vocab)
            if ABSENT in vocab:
                return ABSENT
            raise KeyError(
                f"Feature {name!r} is tagged {val!r}, which is not in "
                f"the vocabulary this processor was built with "
                f"({sorted(map(str, vocab))}). unknown_as_absent is "
                f"set, but this feature has nothing to fall back on: "
                f"it carries no impute entry and its vocabulary has no "
                f"{ABSENT!r} row (only a processor built with "
                f"tolerate_missing=True has one). Pass "
                f"impute={{{name!r}: ...}} to say what an unknown "
                f"value should read as."
            )
        raise KeyError(
            f"Feature {name!r} is tagged {val!r}, which is not in the "
            f"vocabulary this processor was built with "
            f"({sorted(map(str, vocab))}). The tag is not silent about "
            f"this feature, so no fill applies -- imputation covers a "
            f"feature the tag omits, not one it gives an unseen value. "
            f"Encoding it takes a processor whose {name!r} vocabulary "
            f"lists {val!r}: train against a schema that spans both "
            f"task sets, or leave the feature out of the schema. To "
            f"read an unseen value as an absent one instead, build the "
            f"processor with unknown_as_absent=True (see the module "
            f"note on what that does and does not buy)."
        )

    def _resolve(self, x: dict[str, Any], name: str) -> Any:
        """Look ``name`` up in the tag, falling back to its derivation.

        Parameters
        ----------
        x : dict[str, Any]
            Raw feature dict.
        name : str
            Feature to resolve.

        Returns
        -------
        Any
            The value, or :data:`_MISSING` when the tag has no such key
            and the feature is not derivable.
        """
        if name in x:
            return x[name]
        if name in DERIVED_FEATURES:
            return DERIVED_FEATURES[name](x)
        return _MISSING

    def __call__(self, x: dict[str, Any]) -> jax.Array:
        return self.transform(x)

    def swap_names(self, name_mapping: dict[str, str]) -> FeatureProcessor:
        """Rename features according to a mapping.

        Names absent from ``name_mapping`` are kept as-is; vocabulary
        contents and feature ordering are unchanged. Called by the
        stage models' ``swap_optimizers`` (this method existed before
        the module-structure refactor and was lost in it — restored
        because the ``"l2co"`` registry path's explicit
        ``swap_optimizers`` list exercises it).

        Names the processor does not encode are ignored, so a mapping
        may name features belonging to other processors (the stage-1
        models pass one keyed by optimizer name).

        Parameters
        ----------
        name_mapping : dict[str, str]
            Mapping of old feature names to new feature names.

        Returns
        -------
        FeatureProcessor
            New instance with renamed features.

        Raises
        ------
        ValueError
            If the mapping sends two features to the same name. The
            renamed vocabulary would then hold one entry where there
            were two -- a silently shorter feature vector, which
            :class:`MixedEmbedding` was built for the old length of.
        """
        new_names = [name_mapping.get(name, name) for name in self.vocabs]
        collisions = sorted(
            {name for name in new_names if new_names.count(name) > 1}
        )
        if collisions:
            raise ValueError(
                f"Renaming would send more than one feature to each of "
                f"{collisions}, leaving {len(set(new_names))} features "
                f"where there were {len(new_names)}. A shorter feature "
                f"vector does not match the MixedEmbedding built for "
                f"this processor, so the collision is rejected here "
                f"rather than silently dropping a feature."
            )
        new_vocabs = dict(zip(new_names, self.vocabs.values(), strict=True))
        new_impute = {
            name_mapping.get(name, name): val
            for name, val in self.impute.items()
        }
        return FeatureProcessor(
            vocabs=new_vocabs,
            impute=new_impute,
            unknown_as_absent=self.unknown_as_absent,
        )

    def with_imputation(self, impute: dict[str, Any]) -> FeatureProcessor:
        """Copy of this processor that fills the named absent features.

        The intended entry point for reusing a stored processor on a
        task set it was not built for: vocabularies and feature order
        are untouched, so a model trained against this processor still
        loads, and only the handling of an absent feature changes.

        Parameters
        ----------
        impute : dict[str, Any]
            Feature name to fill value, replacing any existing
            :attr:`impute` map. Validated as in :meth:`init`.

        Returns
        -------
        FeatureProcessor
            New instance; ``self`` is unchanged.

        Examples
        --------
        Run a processor built for bbob over a task set that tags
        neither ``separable`` nor ``unimodal``::

            fp = FeatureProcessor.load(path).with_imputation(
                {"separable": False, "unimodal": True}
            )
        """
        return FeatureProcessor(
            vocabs=self.vocabs,
            impute=dict(impute),
            unknown_as_absent=self.unknown_as_absent,
        )

    def with_unknown_as_absent(
        self, unknown_as_absent: bool = True
    ) -> FeatureProcessor:
        """Copy of this processor that survives an unseen value.

        The companion of :meth:`with_imputation` for the other half of
        the cross-suite problem: where that one covers a feature the tag
        omits, this one covers a feature the tag gives a value the
        vocabulary has never seen. Vocabularies, feature order and fills
        are untouched, so a model trained against this processor still
        loads.

        Parameters
        ----------
        unknown_as_absent : bool, optional
            What to set the flag to. Default ``True``.

        Returns
        -------
        FeatureProcessor
            New instance; ``self`` is unchanged.

        Examples
        --------
        Run a bbob-trained processor over a task set that tags a
        ``family`` its vocabulary does not list, reading the unseen
        family as no information at all::

            fp = (
                FeatureProcessor.load(path)
                .with_unknown_as_absent()
            )

        Read it as a stated value instead, which is the honest choice
        when the foreign family really does behave like a known one::

            fp = (
                FeatureProcessor.load(path)
                .with_imputation({"family": "bbob"})
                .with_unknown_as_absent()
            )
        """
        return FeatureProcessor(
            vocabs=self.vocabs,
            impute=dict(self.impute),
            unknown_as_absent=unknown_as_absent,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        # Every optional key is omitted when it carries nothing, so a
        # processor built the ordinary way serializes to exactly the
        # bytes it always did -- the stored artifacts under models/ are
        # compared and re-read by older code paths.
        out: dict = {"vocabs": self.vocabs}
        if self.impute:
            out["impute"] = self.impute
        if self.unknown_as_absent:
            out["unknown_as_absent"] = True
        typed = _typed_vocab_pairs(self.vocabs)
        if typed:
            out["vocab_pairs"] = typed
        return out

    @classmethod
    def reconstruct(
        cls,
        vocabs: dict,
        impute: dict | None = None,
        unknown_as_absent: bool = False,
        vocab_pairs: dict | None = None,
    ) -> FeatureProcessor:
        # Every keyword past ``vocabs`` defaults, so artifacts written
        # before it existed still load. ``vocab_pairs`` wins where it is
        # present: it is the only copy of a vocabulary whose keys are
        # not strings (see :func:`_typed_vocab_pairs`).
        vocabs = dict(vocabs)
        for name, pairs in (vocab_pairs or {}).items():
            vocabs[name] = {key: int(index) for key, index in pairs}
        return cls(
            vocabs=vocabs,
            impute=dict(impute or {}),
            unknown_as_absent=unknown_as_absent,
        )

    def store(self, path: str) -> str:
        _path = Path(path)
        with open(_path.with_suffix(".eqx"), "wb") as f:
            f.write((json.dumps(self.to_dict()) + "\n").encode())
            eqx.tree_serialise_leaves(f, self)
        return str(_path)

    @classmethod
    def load(cls, path: str) -> FeatureProcessor:
        _path = Path(path)
        with open(_path.with_suffix(".eqx"), "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            fp = cls.reconstruct(**hyperparams)
            return eqx.tree_deserialise_leaves(f, fp)


class MixedEmbedding(eqx.Module):
    """Embeds a mixed-type feature array and projects to output_dim.

    Each feature gets its own layer:
      - continuous  → Linear(1, embed_dim)
      - boolean     → Embedding(2, embed_dim)
      - categorical → Embedding(vocab_size, embed_dim)

    Outputs are concatenated and passed through a final
    Linear(num_features * embed_dim, output_dim) projection.

    Parameters
    ----------
    embedders : list[eqx.Module]
        One layer per feature, in schema order.
    final_proj : eqx.nn.Linear
        Projection from concatenated embeddings to output_dim.
    embed_dim : int
        Embedding size of each individual layer.
    """

    embedders: list[eqx.Module]
    final_proj: eqx.nn.Linear
    embed_dim: int = eqx.field(static=True)

    @property
    def out_size(self) -> int:
        return self.final_proj.out_features

    @classmethod
    def from_feature_processor(
        cls,
        feature_processor: FeatureProcessor,
        embed_dim: int,
        output_dim: int,
        *,
        key: jax.Array,
    ) -> MixedEmbedding:
        """Build a MixedEmbedding from a FeatureProcessor.

        Parameters
        ----------
        feature_processor : FeatureProcessor
        embed_dim : int
            Size of each individual embedding / linear output.
        output_dim : int
            Dimensionality of the final projected output.
        key : jax.Array

        Returns
        -------
        MixedEmbedding
        """
        num_features = len(feature_processor.vocabs)
        keys = jax.random.split(key, num_features + 1)

        embedders: list[eqx.Module] = []
        for (_name, vocab), k in zip(
            feature_processor.vocabs.items(), keys[:-1], strict=True
        ):
            if vocab is None:
                embedders.append(eqx.nn.Linear(1, embed_dim, key=k))
            else:  # boolean or categorical
                embedders.append(
                    eqx.nn.Embedding(
                        num_embeddings=len(vocab),
                        embedding_size=embed_dim,
                        key=k,
                    )
                )

        final_proj = eqx.nn.Linear(
            embed_dim * num_features,
            output_dim,
            key=keys[-1],
        )
        return cls(
            embedders=embedders,
            final_proj=final_proj,
            embed_dim=embed_dim,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Embed features and project to output_dim.

        Parameters
        ----------
        x : jax.Array
            Default-precision float array of shape (num_features,) as
            produced by FeatureProcessor.transform.

        Returns
        -------
        jax.Array
            Shape (output_dim,).
        """
        outputs = []
        for xi, layer in zip(x, self.embedders, strict=True):
            if isinstance(layer, eqx.nn.Embedding):
                outputs.append(layer(xi.astype(int)))
            else:
                outputs.append(layer(xi[None]))
        return self.final_proj(jnp.concatenate(outputs, axis=0))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        embed_list = []
        for layer in self.embedders:
            if isinstance(layer, eqx.nn.Embedding):
                embed_list.append(
                    {
                        "_type": "embedding",
                        "num_embeddings": layer.num_embeddings,
                        "embedding_size": layer.embedding_size,
                    }
                )
            else:
                embed_list.append({"_type": "linear"})
        return {
            "embedders": embed_list,
            "embed_dim": self.embed_dim,
            "output_dim": self.final_proj.out_features,
        }

    @classmethod
    def reconstruct(
        cls, embedders, embed_dim, output_dim, **kwargs
    ) -> MixedEmbedding:
        key = jr.key(0)
        k, *layer_keys = jax.random.split(key, len(embedders) + 1)

        embedders_list: list[eqx.Module] = []
        for emb, lk in zip(embedders, layer_keys, strict=True):
            if emb["_type"] == "embedding":
                embedders_list.append(
                    eqx.nn.Embedding(
                        num_embeddings=emb["num_embeddings"],
                        embedding_size=emb["embedding_size"],
                        key=lk,
                    )
                )
            else:
                embedders_list.append(eqx.nn.Linear(1, embed_dim, key=lk))

        proj = eqx.nn.Linear(
            in_features=embed_dim * len(embedders),
            out_features=output_dim,
            key=k,
        )
        return cls(
            embedders=embedders_list,
            final_proj=proj,
            embed_dim=embed_dim,
        )

    def store(self, path: str) -> str:
        _path = Path(path)
        with open(_path.with_suffix(".eqx"), "wb") as f:
            f.write((json.dumps(self.to_dict()) + "\n").encode())
            eqx.tree_serialise_leaves(f, self)
        return str(_path)

    @classmethod
    def load(cls, path: str) -> MixedEmbedding:
        _path = Path(path)
        with open(_path.with_suffix(".eqx"), "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            me = cls.reconstruct(**hyperparams)
            return eqx.tree_deserialise_leaves(f, me)

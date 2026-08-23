"""Feature processing, normalization, and mixed embeddings for task inputs."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

Schema = dict[str, str | list]


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
    store round-trips, so both spellings are accepted. A missing
    ``noise`` key is read as deterministic.

    Parameters
    ----------
    tag : dict[str, Any]
        Raw task tag (or the stringified ``task_labels`` row).

    Returns
    -------
    bool
        True when the task is noise-free.
    """
    noise = tag.get("noise", 0.0)

    if isinstance(noise, bool):
        return not noise

    if isinstance(noise, str):
        token = noise.strip().lower()
        if token in ("", "none", "false"):
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
# THE OTHER HALF OF THE CROSS-SUITE PROBLEM, which imputation does not touch.
# A tag can disagree with the training schema by being silent about a feature
# -- that is this section -- or by giving one a value the vocabulary has never
# seen. The second is not absence, so no fill fires; `transform` raises
# instead. It is not an oversight: filling a silent feature asserts a value
# nobody stated, while remapping `helmholtz` onto `euler` would assert one the
# tag explicitly contradicts. A processor whose vocabulary spans both task
# sets -- or a schema that leaves the feature out -- is the honest answer, and
# the raised message says so.

#: Vocabulary entry standing for "this feature is absent from the task tag".
ABSENT = "__absent__"

#: Suffix of the companion boolean generated for each continuous feature.
PRESENCE_SUFFIX = "__present"

#: Returned by :meth:`FeatureProcessor._resolve` when a feature is neither in
#: the tag nor derivable. Distinct from a tag that carries an explicit
#: null-ish value, which is absent in a different way (see :func:`_is_absent`).
_MISSING = object()


def _normalize_categorical(val: Any) -> Any:
    """Put a value in the form vocabularies are keyed by.

    Booleans are stored as the strings ``"false"`` / ``"true"``, so a
    Python ``bool`` has to be folded to that form before any vocabulary
    lookup.

    Deliberately does not touch strings. A categorical schema is free
    to declare ``"True"`` as a literal category, and case-folding every
    string would make that entry unreachable.

    Parameters
    ----------
    val : Any
        Raw value.

    Returns
    -------
    Any
        ``val`` unchanged, or its lowercased string form when it is a
        ``bool``.
    """
    if isinstance(val, bool):
        return str(val).lower()
    return val


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
    if getattr(val, "ndim", None) == 0 and hasattr(val, "item"):
        return val.item()
    return val


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
        if _normalize_categorical(val) not in vocab:
            raise ValueError(
                f"Fill value {val!r} for feature {name!r} is not in its "
                f"vocabulary {sorted(map(str, vocab))}. An out-of-vocabulary "
                f"index does not raise downstream -- it clamps to the last "
                f"embedding row -- so it is rejected here instead."
            )


def _is_absent(val: Any) -> bool:
    """Whether a resolved feature value carries no information.

    Covers the two ways a feature goes missing. A tag may simply not
    have the key (``_MISSING``). Or it may carry a null-ish value: a
    databank store keeps one dense ``(task, feature)`` label matrix, so
    a feature only some tasks tag round-trips as ``nan`` for the rest,
    and feeding that to a continuous feature's ``Linear`` would put NaN
    through the whole network rather than fail.

    Parameters
    ----------
    val : Any
        Resolved value, or ``_MISSING``.

    Returns
    -------
    bool
    """
    if val is _MISSING or val is None:
        return True
    if isinstance(val, str):
        return val.strip().lower() in ("nan", "none", "")
    if isinstance(val, bool):
        return False
    try:
        return bool(jnp.isnan(jnp.asarray(val, dtype=float)))
    except (TypeError, ValueError):
        return False


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

    Parameters
    ----------
    vocabs : dict[str, dict[object, int] | None]
        Per-feature vocabulary mapping. ``None`` marks a continuous
        feature; a dict gives the value→index lookup for boolean or
        categorical features. Insertion order of ``vocabs`` defines
        the feature ordering used by :meth:`transform`.
    """

    vocabs: dict[str, dict[object, int] | None] = eqx.field(static=True)
    impute: dict[str, Any] = eqx.field(static=True, default_factory=dict)

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

        Returns
        -------
        FeatureProcessor
        """
        vocabs: dict[str, dict[object, int] | None] = {}

        for name, spec in schema.items():
            if spec == "continuous":
                vocabs[name] = None
                if tolerate_missing:
                    # Directly after its own feature, so the encoded
                    # vector stays readable in schema order.
                    vocabs[name + PRESENCE_SUFFIX] = {"false": 0, "true": 1}
            elif spec == "boolean":
                vocabs[name] = {"false": 0, "true": 1}
            elif hasattr(spec, "__iter__"):
                categories = list(spec)
                vocabs[name] = {v: i for i, v in enumerate(categories)}
            else:
                raise ValueError(
                    f"Unknown feature spec for {name!r}: {spec!r}. "
                    f"Expected 'continuous', 'boolean', or an iterable "
                    f"of categories."
                )

            if tolerate_missing and vocabs[name] is not None:
                vocabs[name][ABSENT] = len(vocabs[name])

        return cls(vocabs=vocabs, impute=dict(impute or {}))

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
            a value outside its vocabulary, which no fill covers (see
            the note below).

        Notes
        -----
        When a processor carries both an :attr:`impute` entry and an
        ``ABSENT`` vocabulary entry for the same feature, the fill wins
        --- it is the more specific of the two. A ``<name>__present``
        companion still reports whether the *tag* carried a value, so
        it reads ``false`` even where the value slot was filled.

        A tag can disagree with the schema a second way, which neither
        mechanism covers: it can carry a categorical feature with a
        value the vocabulary has never seen. That is not absence, so
        no fill fires, and the encoding stops with a ``KeyError``
        naming the feature and its vocabulary. See the module note on
        imputation for why remapping the value instead would be worse
        than the gap it closes.
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
                key = _normalize_categorical(val)
                if key not in vocab:
                    raise KeyError(
                        f"Feature {name!r} is tagged {val!r}, which is "
                        f"not in the vocabulary this processor was "
                        f"built with ({sorted(map(str, vocab))}). The "
                        f"tag is not silent about this feature, so no "
                        f"fill applies -- imputation covers a feature "
                        f"the tag omits, not one it gives an unseen "
                        f"value. Encoding it takes a processor whose "
                        f"{name!r} vocabulary lists {val!r}: train "
                        f"against a schema that spans both task sets, "
                        f"or leave the feature out of the schema."
                    )
                feats.append(vocab[key])
        return jnp.array(feats, dtype=float)

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

        Parameters
        ----------
        name_mapping : dict[str, str]
            Mapping of old feature names to new feature names.

        Returns
        -------
        FeatureProcessor
            New instance with renamed features.
        """
        new_vocabs = {
            name_mapping.get(name, name): vocab
            for name, vocab in self.vocabs.items()
        }
        new_impute = {
            name_mapping.get(name, name): val
            for name, val in self.impute.items()
        }
        return FeatureProcessor(vocabs=new_vocabs, impute=new_impute)

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
        return FeatureProcessor(vocabs=self.vocabs, impute=dict(impute))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        # ``impute`` is omitted when empty so that a processor built the
        # ordinary way serializes to exactly the bytes it always did --
        # the stored artifacts under models/ are compared and re-read by
        # older code paths.
        out: dict = {"vocabs": self.vocabs}
        if self.impute:
            out["impute"] = self.impute
        return out

    @classmethod
    def reconstruct(
        cls, vocabs: dict, impute: dict | None = None
    ) -> FeatureProcessor:
        # ``impute`` defaults so that artifacts written before it existed
        # still load.
        return cls(vocabs=vocabs, impute=dict(impute or {}))

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

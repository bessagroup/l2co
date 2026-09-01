"""Filling in features a task tag does not carry.

The case these cover: a policy trained on one benchmark suite, asked to
run on a task set that tags fewer features. ``tolerate_missing`` cannot
serve it -- widening the encoding changes the shapes
``MixedEmbedding`` builds, so the trained checkpoint no longer loads.
Imputation keeps every shape and fills the gap with a stated value.

The shape assertions are the point of the whole mechanism, so they are
tested directly rather than left implied by a round-trip.
"""

import warnings

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from l2co.features import FeatureProcessor, MixedEmbedding

SCHEMA = {
    "dimensionality": "continuous",
    "separable": "boolean",
    "unimodal": "boolean",
}

# A tag from a suite that tags neither boolean -- the quadratic shape.
SPARSE_TAG = {"dimensionality": 10, "n_observations": 10, "normalize": False}

# A tag from the suite the processor was built for.
DENSE_TAG = {"dimensionality": 10, "separable": True, "unimodal": False}

FILL = {"separable": False, "unimodal": True}


@pytest.fixture
def plain():
    return FeatureProcessor.init(SCHEMA)


@pytest.fixture
def imputing(plain):
    return plain.with_imputation(FILL)


def test_absent_feature_still_raises_without_imputation(plain):
    """The default is unchanged: no silent fill for anyone."""
    with pytest.raises(KeyError, match="separable"):
        plain.transform(SPARSE_TAG)


def test_imputed_features_encode_to_the_stated_values(imputing):
    """false -> index 0, true -> index 1, in schema order."""
    assert jnp.array_equal(
        imputing.transform(SPARSE_TAG), jnp.array([10.0, 0.0, 1.0])
    )


def test_imputation_does_not_touch_a_tag_that_has_the_features(
    plain, imputing
):
    """A fill applies only where the tag is silent.

    Guards the failure that would make every imputing run wrong at
    once: a fill overriding values the task actually carries.
    """
    assert jnp.array_equal(
        plain.transform(DENSE_TAG), imputing.transform(DENSE_TAG)
    )


def test_encoding_shapes_are_unchanged(plain, imputing):
    """The reason to impute rather than widen: the checkpoint fits.

    Both the per-feature embedding sizes and the projection's input
    width have to match, or a model trained against ``plain`` cannot
    load against ``imputing``.
    """
    kw = dict(embed_dim=8, output_dim=16, key=jr.key(0))
    a = MixedEmbedding.from_feature_processor(plain, **kw)
    b = MixedEmbedding.from_feature_processor(imputing, **kw)

    assert len(plain.vocabs) == len(imputing.vocabs)
    assert a.final_proj.in_features == b.final_proj.in_features
    assert [getattr(x, "num_embeddings", None) for x in a.embedders] == [
        getattr(x, "num_embeddings", None) for x in b.embedders
    ]


def test_tolerate_missing_by_contrast_changes_the_shapes():
    """Pins the incompatibility that motivates imputation.

    If widening ever stops changing shapes this test should fail and
    the imputation path can be reconsidered.
    """
    kw = dict(embed_dim=8, output_dim=16, key=jr.key(0))
    plain = MixedEmbedding.from_feature_processor(
        FeatureProcessor.init(SCHEMA), **kw
    )
    widened = MixedEmbedding.from_feature_processor(
        FeatureProcessor.init(SCHEMA, tolerate_missing=True), **kw
    )
    assert plain.final_proj.in_features != widened.final_proj.in_features


def test_out_of_vocabulary_fill_is_rejected_at_construction(plain):
    """Rejected early because it cannot be caught late.

    An out-of-range index into ``eqx.nn.Embedding`` clamps to the last
    row instead of raising, so a bad fill would encode as a real
    category and never surface.
    """
    with pytest.raises(ValueError, match="not in its vocabulary"):
        plain.with_imputation({"separable": "maybe"})


def test_fill_for_an_unknown_feature_is_rejected(plain):
    with pytest.raises(KeyError, match="does not encode"):
        plain.with_imputation({"nonexistent": True})


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), "10", True, np.array([1.0, 2.0])]
)
def test_continuous_fill_must_be_a_finite_number(plain, bad):
    with pytest.raises(ValueError, match="continuous feature"):
        plain.with_imputation({"dimensionality": bad})


def test_a_fill_aimed_at_a_presence_companion_is_rejected():
    """Refused up front because it would be dropped without a word.

    ``transform`` answers a ``__present`` flag from the tag before it
    ever reaches the fill map, so a fill named after the companion
    encodes nothing and reports nothing.
    """
    fp = FeatureProcessor.init(
        {"dimensionality": "continuous"}, tolerate_missing=True
    )
    with pytest.raises(ValueError, match="presence companion"):
        fp.with_imputation({"dimensionality__present": True})


@pytest.mark.parametrize("wrapped", [np.float64(7.0), jnp.array(7.0)])
def test_an_array_scalar_fill_is_stored_as_a_plain_number(plain, wrapped):
    """A fill read off a computation means the number it wraps.

    Keeping the array would put one inside a static field, which
    Equinox warns about and which no longer serializes to JSON.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fp = plain.with_imputation({"dimensionality": wrapped})

    assert type(fp.impute["dimensionality"]) is float
    assert fp.impute["dimensionality"] == 7.0
    assert not [w for w in caught if "static" in str(w.message)]


def test_an_array_scalar_fill_survives_a_round_trip(tmp_path):
    """The reason the coercion has to happen: JSON cannot hold an array."""
    path = str(tmp_path / "fp")
    fp = FeatureProcessor.init({"dimensionality": "continuous"})
    fp.with_imputation({"dimensionality": np.float64(7.0)}).store(path)

    assert FeatureProcessor.load(path).transform({})[0] == 7.0


# A pair of PDE schemas from l2co_tasks, which declare `task_name` as a
# one-entry categorical: identical feature counts and embedding widths, so
# a checkpoint crosses between them cleanly and only `transform` refuses.
EULER = {"dimensionality": "continuous", "task_name": ["euler"]}
HELMHOLTZ_TAG = {"dimensionality": 20, "task_name": "helmholtz"}


def test_a_value_outside_the_vocabulary_names_the_feature_and_the_vocabulary():
    """The other way a tag disagrees with the schema it is read against.

    A bare ``vocab[val]`` used to raise ``KeyError: 'helmholtz'`` --
    no feature, no vocabulary, nothing to act on.
    """
    fp = FeatureProcessor.init(EULER)
    with pytest.raises(KeyError, match="not in the vocabulary") as excinfo:
        fp.transform(HELMHOLTZ_TAG)

    assert "task_name" in str(excinfo.value)
    assert "euler" in str(excinfo.value)


def test_a_fill_does_not_rescue_a_value_outside_the_vocabulary():
    """Pins the boundary between the two failures.

    Imputation fires only where the tag is silent, so a fill for the
    same feature changes nothing here -- and a fill *of* the unseen
    value is refused at construction.
    """
    fp = FeatureProcessor.init(EULER).with_imputation({"task_name": "euler"})
    with pytest.raises(KeyError, match="not in the vocabulary"):
        fp.transform(HELMHOLTZ_TAG)

    with pytest.raises(ValueError, match="not in its vocabulary"):
        FeatureProcessor.init(EULER).with_imputation(
            {"task_name": "helmholtz"}
        )


def test_clamping_is_why_validation_exists(plain):
    """Documents the silent-failure mode the validator prevents."""
    table = MixedEmbedding.from_feature_processor(
        plain, embed_dim=4, output_dim=2, key=jr.key(0)
    ).embedders[1]
    assert table.num_embeddings == 2
    assert jnp.array_equal(table(jnp.array(2)), table(jnp.array(1)))


def test_round_trip_preserves_the_fill(tmp_path, imputing):
    path = str(tmp_path / "fp")
    imputing.store(path)
    loaded = FeatureProcessor.load(path)

    assert loaded.impute == FILL
    assert jnp.array_equal(
        loaded.transform(SPARSE_TAG), imputing.transform(SPARSE_TAG)
    )


def test_processor_without_a_fill_serializes_as_before(tmp_path, plain):
    """Stored artifacts under models/ are read by older code paths."""
    path = str(tmp_path / "fp")
    plain.store(path)

    with open(path + ".eqx", "rb") as f:
        assert "impute" not in f.readline().decode()


def test_swap_names_carries_the_fill(imputing):
    renamed = imputing.swap_names({"separable": "is_separable"})

    assert renamed.impute["is_separable"] is False
    assert "separable" not in renamed.impute


def test_a_fill_wins_over_an_absent_marker():
    """The specific value beats the generic "unknown" when both exist."""
    fp = FeatureProcessor.init(SCHEMA, tolerate_missing=True).with_imputation(
        {"separable": True}
    )
    encoded = fp.transform(SPARSE_TAG)
    names = list(fp.vocabs)

    assert encoded[names.index("separable")] == 1.0
    # 'unimodal' has no fill, so it still lands on the ABSENT row.
    assert encoded[names.index("unimodal")] == 2.0

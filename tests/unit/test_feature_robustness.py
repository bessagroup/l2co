"""Surviving a task tag that disagrees with the training schema.

A processor is built from the tasks a policy trains on, and then asked
to encode tasks it was not built for. Four disagreements are possible,
and only the first was already benign:

  * the tag carries *more* features than the schema -- ignored;
  * it is silent about one -- ``impute`` / ``tolerate_missing``, covered
    by ``test_feature_impute.py``;
  * it spells a value differently (``np.True_``, ``"True"``, ``1``) --
    folded onto the vocabulary key, here;
  * it gives a value the vocabulary has never seen -- ``KeyError`` by
    default, or the absent path under ``unknown_as_absent``, here.

The distinction the last two draw is the point: a *spelling* difference
is not information, so folding it loses nothing, while an unseen
*value* is information, so discarding it is a decision the caller has
to opt into.
"""

import json

import jax.numpy as jnp
import numpy as np
import pytest

from l2co.features import FeatureProcessor

SCHEMA = {
    "dimensionality": "continuous",
    "separable": "boolean",
    "family": ["bbob", "quadratic"],
}

TAG = {"dimensionality": 10, "separable": True, "family": "bbob"}


@pytest.fixture
def plain():
    return FeatureProcessor.init(SCHEMA)


@pytest.fixture
def tolerant():
    return FeatureProcessor.init(SCHEMA, tolerate_missing=True)


#                                             Spelling of a boolean value
# =============================================================================


@pytest.mark.parametrize(
    "spelling",
    [True, np.True_, "true", "True", " TRUE ", 1, 1.0, "1", np.int64(1)],
)
def test_every_spelling_of_true_encodes_as_true(plain, spelling):
    """The same task, tagged by a different route, encodes alike.

    A live ``Task.tag`` says ``True``, a NumPy label array says
    ``np.True_``, a store that stringified its labels says ``"True"``,
    and a config or a float label matrix says ``1`` / ``1.0``. None of
    those is a different task.
    """
    assert plain.transform({**TAG, "separable": spelling}).tolist() == (
        plain.transform({**TAG, "separable": True}).tolist()
    )


@pytest.mark.parametrize(
    "spelling", [False, np.False_, "false", "False", 0, 0.0, "0"]
)
def test_every_spelling_of_false_encodes_as_false(plain, spelling):
    assert plain.transform({**TAG, "separable": spelling}).tolist() == (
        plain.transform({**TAG, "separable": False}).tolist()
    )


def test_a_zero_dimensional_array_is_unwrapped(plain):
    """JAX scalars reach the processor from a jitted caller's tag."""
    encoded = plain.transform({**TAG, "separable": jnp.array(True)})
    assert encoded.tolist() == plain.transform(TAG).tolist()


def test_a_number_that_is_not_zero_or_one_is_not_a_boolean(plain):
    """Only the two spellings fold; ``2`` is an unseen value."""
    with pytest.raises(KeyError, match="separable"):
        plain.transform({**TAG, "separable": 2})


def test_a_categorical_true_is_still_taken_literally():
    """Case-folding is confined to *boolean* features.

    A categorical schema is free to declare ``"True"`` and ``"true"`` as
    two distinct categories, and folding either would make one of them
    unreachable.
    """
    fp = FeatureProcessor.init({"flag": ["True", "true", "other"]})
    assert fp.transform({"flag": "True"}).tolist() == [0.0]
    assert fp.transform({"flag": "true"}).tolist() == [1.0]


def test_numpy_scalars_encode_as_the_values_they_hold(plain):
    encoded = plain.transform(
        {
            "dimensionality": np.float32(10),
            "separable": np.True_,
            "family": np.str_("bbob"),
        }
    )
    assert encoded.tolist() == plain.transform(TAG).tolist()


#                                             A value outside the vocabulary
# =============================================================================


def test_an_unseen_value_still_raises_by_default(plain):
    """The default is unchanged, and the message names the way out."""
    with pytest.raises(KeyError, match="unknown_as_absent"):
        plain.transform({**TAG, "family": "helmholtz"})


def test_an_unseen_value_takes_the_absent_row_when_asked(tolerant):
    absent_index = tolerant.vocabs["family"]["__absent__"]
    encoded = tolerant.with_unknown_as_absent().transform(
        {**TAG, "family": "helmholtz"}
    )
    assert encoded[-1] == absent_index


def test_an_unseen_value_takes_the_fill_when_there_is_one(plain):
    """A stated fill beats the generic "no information" row."""
    fp = plain.with_imputation({"family": "quadratic"})
    encoded = fp.with_unknown_as_absent().transform(
        {**TAG, "family": "helmholtz"}
    )
    assert encoded[-1] == plain.vocabs["family"]["quadratic"]


def test_the_fill_wins_over_the_absent_row_for_an_unseen_value(tolerant):
    fp = tolerant.with_imputation({"family": "quadratic"})
    encoded = fp.with_unknown_as_absent().transform(
        {**TAG, "family": "helmholtz"}
    )
    assert encoded[-1] == tolerant.vocabs["family"]["quadratic"]


def test_with_nothing_to_fall_back_on_the_flag_says_so(plain):
    """The flag alone cannot invent a row that is not in the vocabulary.

    A strict processor has no ``ABSENT`` entry, and an out-of-range
    index would clamp silently inside the embedding rather than raise --
    so this has to stop here, naming the fill as the remedy.
    """
    with pytest.raises(KeyError, match="nothing to fall back on"):
        plain.with_unknown_as_absent().transform(
            {**TAG, "family": "helmholtz"}
        )


def test_an_unseen_boolean_spelling_is_an_unseen_value(tolerant):
    """Whatever does not fold is treated as a value, not as a spelling."""
    encoded = tolerant.with_unknown_as_absent().transform(
        {**TAG, "separable": "maybe"}
    )
    assert encoded[2] == tolerant.vocabs["separable"]["__absent__"]


def test_a_known_value_is_untouched_by_the_flag(tolerant):
    assert (
        tolerant.with_unknown_as_absent().transform(TAG).tolist()
        == tolerant.transform(TAG).tolist()
    )


def test_the_flag_leaves_every_shape_alone(plain, tolerant):
    """Which is the whole reason it exists next to ``tolerate_missing``.

    Widening the encoding is what a trained checkpoint cannot survive;
    this mechanism changes only what an unseen value encodes *as*.
    """
    for fp in (plain, tolerant):
        assert list(fp.with_unknown_as_absent().vocabs) == list(fp.vocabs)
        assert fp.with_unknown_as_absent().vocabs == fp.vocabs


def test_the_flag_is_carried_by_the_other_copiers(plain):
    fp = plain.with_unknown_as_absent()
    assert fp.with_imputation({"family": "bbob"}).unknown_as_absent
    assert fp.swap_names({"family": "fam"}).unknown_as_absent


def test_the_flag_survives_a_round_trip(tmp_path, tolerant):
    fp = tolerant.with_unknown_as_absent()
    fp.store(str(tmp_path / "fp"))
    reloaded = FeatureProcessor.load(str(tmp_path / "fp"))
    assert reloaded.unknown_as_absent
    assert reloaded.transform({**TAG, "family": "helmholtz"}).tolist() == (
        fp.transform({**TAG, "family": "helmholtz"}).tolist()
    )


def test_a_processor_without_the_flag_serializes_as_before(plain):
    """No new key in the header unless the processor uses the feature."""
    assert plain.to_dict() == {"vocabs": plain.vocabs}


#                                                   Values that are not values
# =============================================================================


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -float("inf"), "nan", "inf", "-Infinity"],
)
def test_a_non_finite_value_takes_the_fill(plain, value):
    """``inf`` is no more usable to a ``Linear`` than ``nan`` is."""
    fp = plain.with_imputation({"dimensionality": 7.0})
    assert fp.transform({**TAG, "dimensionality": value})[0] == 7.0


def test_a_non_finite_value_reads_false_on_the_presence_companion(tolerant):
    encoded = tolerant.transform({**TAG, "dimensionality": float("inf")})
    assert encoded[0] == 0.0
    assert encoded[1] == tolerant.vocabs["dimensionality__present"]["false"]


def test_a_finite_value_is_left_alone(plain):
    fp = plain.with_imputation({"dimensionality": 7.0})
    assert fp.transform(TAG)[0] == 10.0


#                                                        Derived features
# =============================================================================


@pytest.mark.parametrize(
    "tag",
    [
        {},
        {"noise": None},
        {"noise": float("nan")},
        {"noise": "nan"},
        {"noise": ""},
        {"noise": 0.0},
        {"noise": "0.0"},
        {"noise": False},
    ],
)
def test_a_noise_tag_that_says_nothing_reads_as_deterministic(tag):
    """A dense label matrix stores an untagged feature as ``nan``.

    The suites that do not tag ``noise`` are the noise-free ones, so
    ``nan`` has to agree with a missing key -- otherwise every task in a
    store that merely *contains* a noisy suite reads as stochastic.
    """
    fp = FeatureProcessor.init({"deterministic": "boolean"})
    assert fp.transform(tag).tolist() == [1.0]


@pytest.mark.parametrize(
    "tag",
    [
        {"noise": True},
        {"noise": 0.5},
        {"noise": "0.5"},
        {"noise": "heavy"},
    ],
)
def test_a_noise_tag_that_says_something_else_reads_as_stochastic(tag):
    """Including a token nobody can parse.

    The tag is asserting something about noise and it is not "none", so
    guessing noise-freedom is the wrong direction: a policy that trusts
    it would pick a deterministic optimizer for a noisy loss.
    """
    fp = FeatureProcessor.init({"deterministic": "boolean"})
    assert fp.transform(tag).tolist() == [0.0]


#                                                                  Renaming
# =============================================================================


def test_renaming_two_features_onto_one_name_is_rejected(plain):
    """It would silently shorten the feature vector.

    ``MixedEmbedding`` is built for one embedder per vocabulary entry,
    so a processor that quietly lost a feature no longer matches the
    model trained against it.
    """
    with pytest.raises(ValueError, match="separable"):
        plain.swap_names({"family": "separable"})


def test_renaming_leaves_the_feature_count_alone(plain):
    renamed = plain.swap_names({"family": "fam"})
    assert list(renamed.vocabs) == ["dimensionality", "separable", "fam"]


def test_a_mapping_of_unrelated_names_is_a_no_op(plain):
    """The stage-1 models pass one keyed by optimizer name."""
    assert list(plain.swap_names({"adam": "adam"}).vocabs) == list(
        plain.vocabs
    )


#                                                    Vocabulary round-trips
# =============================================================================


def test_a_numeric_vocabulary_survives_a_round_trip(tmp_path):
    """JSON object keys are strings; vocabulary keys need not be.

    Without the pair encoding the reloaded vocabulary is keyed by
    ``"10"`` and stops matching the ``10`` a live ``Task.tag`` carries
    -- a processor that encoded fine before it was stored and raises
    after.
    """
    fp = FeatureProcessor.init({"dimensionality": [2, 3, 5, 10]})
    fp.store(str(tmp_path / "fp"))
    reloaded = FeatureProcessor.load(str(tmp_path / "fp"))

    assert reloaded.vocabs == fp.vocabs
    assert reloaded.transform({"dimensionality": 10}).tolist() == [3.0]


@pytest.mark.parametrize("spelling", [10, 10.0, "10", " 10 ", np.int64(10)])
def test_a_numeric_category_is_matched_however_it_is_spelled(spelling):
    fp = FeatureProcessor.init({"dimensionality": [2, 10]})
    assert fp.transform({"dimensionality": spelling}).tolist() == [1.0]


def test_a_stringified_vocabulary_still_matches_a_numeric_tag():
    """The artifacts already on disk are keyed by strings.

    They were written before the pair encoding existed, and their
    integer categories are gone -- so the fold has to work in this
    direction too, or those processors stay broken.
    """
    fp = FeatureProcessor.reconstruct(
        vocabs={"dimensionality": {"2": 0, "10": 1}}
    )
    assert fp.transform({"dimensionality": 10}).tolist() == [1.0]


def test_a_genuinely_unseen_category_is_not_invented(plain):
    """Folding a spelling must never reach a key that is not there."""
    fp = FeatureProcessor.init({"dimensionality": [2, 10]})
    with pytest.raises(KeyError, match="dimensionality"):
        fp.transform({"dimensionality": 7})


def test_a_string_keyed_processor_gains_no_extra_header(plain):
    """Every processor built today is one, so the bytes do not move."""
    assert "vocab_pairs" not in plain.to_dict()
    assert json.loads(json.dumps(plain.to_dict())) == {"vocabs": plain.vocabs}


def test_a_vocabulary_key_json_cannot_carry_is_rejected():
    """Rather than writing an artifact that fails to load."""
    fp = FeatureProcessor(vocabs={"f": {(1, 2): 0}})
    with pytest.raises(ValueError, match="JSON cannot carry"):
        fp.to_dict()


def test_a_numpy_categorical_spec_is_stored_as_python_values(tmp_path):
    """A schema read off a NumPy array is still a storable processor.

    ``np.int64`` is not one of the types JSON can carry, so a
    vocabulary keyed by one could not be written at all.
    """
    fp = FeatureProcessor.init({"dimensionality": np.array([2, 10])})
    assert [type(k) for k in fp.vocabs["dimensionality"]] == [int, int]

    fp.store(str(tmp_path / "fp"))
    reloaded = FeatureProcessor.load(str(tmp_path / "fp"))
    assert reloaded.transform({"dimensionality": 10}).tolist() == [1.0]

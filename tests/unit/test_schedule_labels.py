"""Unit tests for the schedule display labels in ``LABEL_MAP``.

Covers how a label is looked up (full name first, then the optimizer
before the first underscore), that the labels stay safe to hand to
LaTeX, and that the bbob databank portfolio gets a distinct, readable
name for every schedule.
"""

import re

import pytest

from l2co._src.visualize.colors import LABEL_MAP, display_name

#: The schedules of l2co_experiments' `optimizers=bbob_databank`, as the
#: databank store spells them, plus the two reference lines a report adds.
BBOB_DATABANK_SCHEDULES = [
    "ars", "asebo", "cmaes", "crfmnes", "des", "esmc",
    "gradientlessdescent", "guidedes", "hillclimbing", "iamalgamfull",
    "iamalgamunivariate", "lmmaes", "maes", "noisereusees", "openes",
    "persistentes", "pgpe", "rmes", "sepcmaes", "simulatedannealing",
    "simplees", "snes", "xnes", "differentialevolution",
    "diffusionevolution", "gesmrga", "mr15ga", "pso", "samrga", "simplega",
    "adabelief_learning_rate=0.001", "adadelta", "adan_learning_rate=0.001",
    "adafactor", "adagrad_learning_rate=0.001", "adam_learning_rate=0.001",
    "adamw_learning_rate=0.001",
    "adamax_learning_rate=0.001", "adamaxw_learning_rate=0.001",
    "amsgrad_learning_rate=0.001", "fromage_learning_rate=0.001",
    "lamb_learning_rate=0.001", "lars_learning_rate=0.001",
    "lion_learning_rate=0.001", "nadam_learning_rate=0.001",
    "nadamw_learning_rate=0.001", "noisysgd_learning_rate=0.001",
    "novograd_learning_rate=0.001",
    "optimisticgradientdescent_learning_rate=0.001",
    "optimisticadam_learning_rate=0.001", "radam_learning_rate=0.001",
    "rmsprop_learning_rate=0.001", "rprop_learning_rate=0.001",
    "sgd_learning_rate=0.001", "signsgd_learning_rate=0.001",
    "sm3_learning_rate=0.001", "yogi_learning_rate=0.001", "lbfgs",
    "rbf_trust_region", "turbo", "randomsearch", "vbs",
]  # fmt: skip


def _outside_math(label: str) -> str:
    return re.sub(r"\$[^$]*\$", "", label)


def test_every_bbob_databank_schedule_has_a_label():
    unlabelled = [
        name for name in BBOB_DATABANK_SCHEDULES if display_name(name) == name
    ]
    assert unlabelled == []


def test_bbob_databank_labels_are_distinct():
    # Every schedule must be unique on its own label; the label column's
    # collision fallback is for sweeps, not for this portfolio.
    labels = [display_name(name) for name in BBOB_DATABANK_SCHEDULES]
    assert len(set(labels)) == len(labels)


@pytest.mark.parametrize("label", sorted(LABEL_MAP.values()))
def test_labels_are_safe_to_render_with_latex(label):
    # The dissertation style sets text.usetex, where a bare _ & % or #
    # outside math mode fails the whole figure.
    assert not re.search(r"[_&%#]", _outside_math(label))


def test_a_variant_falls_back_to_its_optimizer_label():
    assert display_name("adabelief_learning_rate=0.001") == "AdaBelief"
    assert display_name("adam_learning_rate=0.5") == "Adam"


def test_an_optimizer_with_an_underscore_is_found_by_its_full_name():
    # The fallback would look up "rbf", which is no optimizer.
    assert display_name("rbf_trust_region") == "RBF trust region"


def test_an_unknown_schedule_keeps_its_raw_name():
    assert display_name("brand_new_optimizer=1") == "brand_new_optimizer=1"

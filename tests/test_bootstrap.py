import numpy as np

from luad.bootstrap import cohort_draws, paired_bootstrap


def _folds(y, seed=0):
    rng = np.random.default_rng(seed)
    folds = []
    for r in range(2):                      # two repeats: patients recur across folds
        idx = rng.permutation(len(y))
        for part in np.array_split(idx, 3):
            train = np.setdiff1d(idx, part)
            good = -y["time"][part] + rng.normal(0, 500, len(part))
            folds.append({"fold": f"r{r}", "train": train, "test": part,
                          "risk": {"good": good, "same": good.copy(),
                                   "noise": rng.normal(size=len(part))}})
    return folds


def test_draws_are_reproducible():
    np.testing.assert_array_equal(cohort_draws(50, 20, 1), cohort_draws(50, 20, 1))
    assert not np.array_equal(cohort_draws(50, 20, 1), cohort_draws(50, 20, 2))


def test_identical_arms_give_exactly_zero(cohort):
    """The pairing control: the same predictions under two names must differ
    by exactly zero in every replicate, or the arms are not sharing a draw."""
    _, con = paired_bootstrap(_folds(cohort.y), cohort.y, [("good", "same")],
                              n_boot=100, seed=0, min_events=3)
    assert (con["difference"] == 0).all()
    assert (con["ci_low"] == 0).all() and (con["ci_high"] == 0).all()


def test_a_real_difference_is_detected(cohort):
    arms, con = paired_bootstrap(_folds(cohort.y), cohort.y, [("good", "noise")],
                                 n_boot=200, seed=0, min_events=3)
    row = con[con["metric"] == "harrell_c"].iloc[0]
    assert row["ci_low"] > 0 and row["verdict"] == "a better"
    assert arms.set_index(["arm", "metric"]).loc[("good", "harrell_c"), "c_index"] > 0.6

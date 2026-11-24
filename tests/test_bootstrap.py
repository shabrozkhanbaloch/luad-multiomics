"""Tests of the paired bootstrap (src/luad/bootstrap.py).

The bootstrap gives every reported interval, so three things are checked:
the random draws are the same on every run, two arms really are scored on the
same draw (the "pairing"), and a difference that is known to exist is found.

No model is fitted here. The "arms" are risks made up from the true survival
times, so the right answer is known in advance:

    good   the true order of the patients plus some noise: a good predictor.
    same   an exact copy of `good` under another name.
    noise  random numbers: a predictor that knows nothing.
"""
import numpy as np

from luad.bootstrap import cohort_draws, paired_bootstrap


def _folds(y, seed=0):
    """Made-up cross-validation results in the form `paired_bootstrap` reads.

    y     outcome of every patient (structured array: event, time).
    seed  random seed.

    Returns a list of six fold dicts (two repeats of three folds), each with
    "fold", "train" and "test" (row numbers) and "risk" ({arm: risks of the
    test patients}) for the three arms described at the top of this file.
    """
    rng = np.random.default_rng(seed)
    folds = []
    for repeat in range(2):                      # two repeats: patients recur across folds
        # A new random order of the patients, cut into three test sets.
        shuffled_rows = rng.permutation(len(y))
        for test_rows in np.array_split(shuffled_rows, 3):
            train_rows = np.setdiff1d(shuffled_rows, test_rows)
            # A shorter time to the event should mean a higher risk, so the
            # true time with its sign turned is a good risk. Noise is added
            # so that it is good but not perfect.
            good = -y["time"][test_rows] + rng.normal(0, 500, len(test_rows))
            folds.append({"fold": f"r{repeat}", "train": train_rows, "test": test_rows,
                          "risk": {"good": good, "same": good.copy(),
                                   "noise": rng.normal(size=len(test_rows))}})
    return folds


def test_draws_are_reproducible():
    """The same seed must give the same draws, and another seed other draws.

    Catches: draws that depend on something that changes between runs (the
    clock, or Python's per-session hashing), or a seed that is ignored. If
    the draws changed from run to run, so would every published interval, and
    nobody could reproduce the tables.
    """
    np.testing.assert_array_equal(cohort_draws(50, 20, 1), cohort_draws(50, 20, 1))
    assert not np.array_equal(cohort_draws(50, 20, 1), cohort_draws(50, 20, 2))


def test_identical_arms_give_exactly_zero(cohort):
    """The pairing control: the same predictions under two names must differ
    by exactly zero in every replicate, or the arms are not sharing a draw.

    Catches: each arm being resampled on its own. Two identical arms would
    then differ by chance in every replicate and the interval would have a
    width. Intervals for real contrasts would be too wide in the same way, and
    true differences would be missed.
    """
    _, contrasts = paired_bootstrap(_folds(cohort.y), cohort.y, [("good", "same")],
                                    n_boot=100, seed=0, min_events=3)
    assert (contrasts["difference"] == 0).all()
    assert (contrasts["ci_low"] == 0).all() and (contrasts["ci_high"] == 0).all()


def test_a_real_difference_is_detected(cohort):
    """A good predictor against random numbers must come out as "a better".

    Catches: a bootstrap that can never resolve anything (intervals far too
    wide), a difference taken the wrong way round (b minus a), or risks lined
    up with the wrong patients, which would turn the good predictor into
    noise. The last line checks that the good arm itself scores well above
    chance.
    """
    arms, contrasts = paired_bootstrap(_folds(cohort.y), cohort.y, [("good", "noise")],
                                       n_boot=200, seed=0, min_events=3)
    row = contrasts[contrasts["metric"] == "harrell_c"].iloc[0]
    assert row["ci_low"] > 0 and row["verdict"] == "a better"
    assert arms.set_index(["arm", "metric"]).loc[("good", "harrell_c"), "c_index"] > 0.6

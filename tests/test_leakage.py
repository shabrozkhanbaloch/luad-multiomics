"""The fold-internal transforms must not read the test rows.

What "leakage" is
    In each fold the models are trained on the training rows and scored on the
    test rows. Before a model is fitted, the data are prepared: the most
    variable genes and probes are chosen, the recurrently mutated genes are
    chosen, and missing clinical values are filled in. These steps are the
    "transforms" (src/luad/transforms.py). If a transform looked at the test
    rows while making its choices, information about the test patients would
    leak into training, and every reported score would be too good. A score is
    only honest if the test patients were unseen at every step.

How it is tested
    By changing the data and watching what moves. If the test rows are replaced
    by nonsense, nothing that was fitted may change. As a counter-check, if the
    training rows are changed, the fit must change.

Words used here
    build_blocks  builds the four blocks for one fold; returns {block name:
                  Block}, where a Block has X (all patients x features) and
                  names (one per feature).
    training rows / test rows   row numbers of the patients, as arrays.
"""
import numpy as np

from luad.transforms import build_blocks, concat, impute_missing


def _split(cohort):
    """One fixed split of the 120 synthetic patients: 90 training rows and 30
    test rows, each as a sorted array of row numbers."""
    shuffled_rows = np.random.default_rng(1).permutation(cohort.n)
    return np.sort(shuffled_rows[:90]), np.sort(shuffled_rows[90:])


def test_test_rows_do_not_change_what_is_fitted(cohort, cfg):
    """Corrupt every test row; the chosen features and the training rows of
    every block must be identical.

    Catches: any transform that reads the test rows, for example choosing the
    most variable genes over all patients, or fitting the imputation of
    clinical values on everyone. This is the main guard against leakage: if it
    fails, no score in the report can be trusted.
    """
    from dataclasses import replace
    train_rows, test_rows = _split(cohort)
    rng = np.random.default_rng(2)
    # A copy of the cohort (so the shared one stays intact) in which the test
    # rows of every data table are replaced by values unrelated to the real ones.
    corrupted = replace(cohort, rna=cohort.rna.copy(), meth=cohort.meth.copy(),
                        mut_gene=cohort.mut_gene.copy(), clinical=cohort.clinical.copy(),
                        rna_counts=cohort.rna_counts.copy())
    corrupted.rna[test_rows] = rng.random(corrupted.rna[test_rows].shape)
    corrupted.meth[test_rows] = rng.random(corrupted.meth[test_rows].shape)
    corrupted.mut_gene[test_rows] = 1 - corrupted.mut_gene[test_rows]
    corrupted.clinical[test_rows] = rng.normal(size=corrupted.clinical[test_rows].shape)
    corrupted.rna_counts[test_rows] = rng.poisson(500, corrupted.rna_counts[test_rows].shape)

    # Same training rows, same settings; only the test rows differ.
    original_blocks = build_blocks(cohort, train_rows, cfg, 0, 20)
    corrupted_blocks = build_blocks(corrupted, train_rows, cfg, 0, 20)
    for name in original_blocks:
        # The same features must be chosen, and the training rows must hold
        # the same values (this covers the imputed clinical values).
        assert original_blocks[name].names == corrupted_blocks[name].names, \
            f"{name}: selection read the test rows"
        np.testing.assert_array_equal(original_blocks[name].X[train_rows],
                                      corrupted_blocks[name].X[train_rows])


def test_selection_does_depend_on_training_rows(cohort, cfg):
    """The opposite check: if the selection never moved, the test above would
    pass for the wrong reason.

    Catches: a selection that ignores the data altogether (say, always the
    first 20 genes). Such a selection cannot leak, so the test above would
    pass, but it would prove nothing. Here another set of training rows must
    give another choice of genes.
    """
    train_rows, _ = _split(cohort)
    other_train_rows = np.sort(np.random.default_rng(9).permutation(cohort.n)[:90])
    blocks = build_blocks(cohort, train_rows, cfg, 0, 20)
    other_blocks = build_blocks(cohort, other_train_rows, cfg, 0, 20)
    assert blocks["rna"].names != other_blocks["rna"].names


def test_outcome_is_never_read_by_transforms(cohort, cfg):
    """Shuffle the outcome among the patients; every block must be identical.

    Catches: a transform that uses survival to prepare the features, for
    example keeping the genes most related to the outcome. Choosing features
    by the outcome is allowed only inside the model tuning, where the inner
    cross-validation accounts for it. Done in a transform, it would make the
    inner scores too good.
    """
    train_rows, _ = _split(cohort)
    shuffled = cohort.with_outcome(cohort.y[np.random.default_rng(3).permutation(cohort.n)])
    blocks = build_blocks(cohort, train_rows, cfg, 0, 20)
    shuffled_blocks = build_blocks(shuffled, train_rows, cfg, 0, 20)
    for name in blocks:
        np.testing.assert_array_equal(blocks[name].X, shuffled_blocks[name].X)


def test_imputation_uses_training_means_only(cohort, cfg):
    """A block declared missing must be filled with its training means, and
    nothing else may be touched.

    This is what early fusion does when a test patient lacks a block. Catches:
    means taken over all patients (the test rows would leak into the fill-in
    values), the wrong columns being filled, or columns of other blocks being
    changed. Any of these would make the missing-block scenarios wrong.
    """
    train_rows, test_rows = _split(cohort)
    X, owner = concat(build_blocks(cohort, train_rows, cfg, 0, 20), ["clinical", "rna", "mutation"])
    imputed = impute_missing(X, owner, train_rows, ["rna"])
    rna_columns = owner == "rna"
    # Every test patient gets the same row for the missing block: the mean of
    # each of its columns over the training patients.
    np.testing.assert_allclose(
        imputed[test_rows][:, rna_columns],
        np.tile(X[train_rows][:, rna_columns].mean(0), (len(test_rows), 1)))
    # The columns of the blocks that are present are exactly as they were.
    np.testing.assert_array_equal(imputed[:, ~rna_columns], X[:, ~rna_columns])

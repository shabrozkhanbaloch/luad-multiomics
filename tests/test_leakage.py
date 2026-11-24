"""The fold-internal transforms must not read the test rows."""
import numpy as np

from luad.transforms import build_blocks, concat, impute_missing


def _split(c):
    idx = np.random.default_rng(1).permutation(c.n)
    return np.sort(idx[:90]), np.sort(idx[90:])


def test_test_rows_do_not_change_what_is_fitted(cohort, cfg):
    """Corrupt every test row; the chosen features and the training rows of
    every block must be identical."""
    from dataclasses import replace
    tr, te = _split(cohort)
    rng = np.random.default_rng(2)
    bad = replace(cohort, rna=cohort.rna.copy(), meth=cohort.meth.copy(),
                  mut_gene=cohort.mut_gene.copy(), clinical=cohort.clinical.copy(),
                  rna_counts=cohort.rna_counts.copy())
    bad.rna[te] = rng.random(bad.rna[te].shape)
    bad.meth[te] = rng.random(bad.meth[te].shape)
    bad.mut_gene[te] = 1 - bad.mut_gene[te]
    bad.clinical[te] = rng.normal(size=bad.clinical[te].shape)
    bad.rna_counts[te] = rng.poisson(500, bad.rna_counts[te].shape)

    a = build_blocks(cohort, tr, cfg, 0, 20)
    b = build_blocks(bad, tr, cfg, 0, 20)
    for name in a:
        assert a[name].names == b[name].names, f"{name}: selection read the test rows"
        np.testing.assert_array_equal(a[name].X[tr], b[name].X[tr])


def test_selection_does_depend_on_training_rows(cohort, cfg):
    """The opposite check: if the selection never moved, the test above would
    pass for the wrong reason."""
    tr, _ = _split(cohort)
    other = np.sort(np.random.default_rng(9).permutation(cohort.n)[:90])
    a = build_blocks(cohort, tr, cfg, 0, 20)
    b = build_blocks(cohort, other, cfg, 0, 20)
    assert a["rna"].names != b["rna"].names


def test_outcome_is_never_read_by_transforms(cohort, cfg):
    tr, _ = _split(cohort)
    shuffled = cohort.with_outcome(cohort.y[np.random.default_rng(3).permutation(cohort.n)])
    a, b = build_blocks(cohort, tr, cfg, 0, 20), build_blocks(shuffled, tr, cfg, 0, 20)
    for name in a:
        np.testing.assert_array_equal(a[name].X, b[name].X)


def test_imputation_uses_training_means_only(cohort, cfg):
    tr, te = _split(cohort)
    X, owner = concat(build_blocks(cohort, tr, cfg, 0, 20), ["clinical", "rna", "mutation"])
    out = impute_missing(X, owner, tr, ["rna"])
    cols = owner == "rna"
    np.testing.assert_allclose(out[te][:, cols], np.tile(X[tr][:, cols].mean(0), (len(te), 1)))
    np.testing.assert_array_equal(out[:, ~cols], X[:, ~cols])

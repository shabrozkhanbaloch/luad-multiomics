import numpy as np

from luad import models as M
from luad.integrate import fit_mofa, late_fusion
from luad.transforms import build_blocks, concat


def _setup(cohort, cfg):
    idx = np.random.default_rng(1).permutation(cohort.n)
    tr, te = np.sort(idx[:90]), np.sort(idx[90:])
    return tr, te, M.inner_splits(cohort.y, tr, 3, 0), build_blocks(cohort, tr, cfg, 0, 20)


def test_higher_risk_means_worse_prognosis(cohort, cfg):
    """Stage drives the synthetic hazard, so both models must score above 0.5
    on held-out rows. A flipped sign would give well below 0.5."""
    tr, te, splits, B = _setup(cohort, cfg)
    X, _ = concat(B, ["clinical"])
    for model in M.MODELS:
        f = M.tune_and_fit(model, X, cohort.y, tr, splits, cfg, 0, 1, quick=True)
        assert M.harrell_c(cohort.y[te], f.predict(X[te])) > 0.6, model


def test_late_fusion_weights_and_missing_blocks(cohort, cfg):
    tr, te, splits, B = _setup(cohort, cfg)
    fits, risks = {}, {}
    for b in ("clinical", "rna"):
        X, _ = concat(B, [b])
        fits[b] = M.tune_and_fit("coxnet", X, cohort.y, tr, splits, cfg, 0, 1, quick=True)
        risks[b] = fits[b].predict(X[te])
    # With the only other block missing, late fusion is the clinical model
    # on another scale: identical ranking.
    alone = late_fusion(fits, risks, missing=["rna"])
    assert np.array_equal(np.argsort(alone, kind="stable"),
                          np.argsort(risks["clinical"], kind="stable"))
    both = late_fusion(fits, risks)
    assert both.shape == alone.shape and np.all(np.isfinite(both))


def test_mofa_projection_recovers_the_shared_factor(cohort, cfg):
    """The synthetic expression and methylation share one latent variable.
    MOFA fitted on training rows must recover it on held-out rows, with both
    views and with either one missing."""
    tr, te, _, _ = _setup(cohort, cfg)
    wide = build_blocks(cohort, tr, cfg, 0, 100)
    views = {v: wide[v].X for v in ("rna", "mutation", "methylation")}
    mofa = fit_mofa({v: views[v][tr] for v in views}, cfg, 0)
    assert len(mofa.active) >= 1

    # The factor projected from expression alone and from methylation alone
    # must agree, since both views were generated from the same variable.
    test_views = {v: views[v][te] for v in views}
    full = mofa.project(test_views)
    from_rna = mofa.project(test_views, missing=["methylation", "mutation"])
    from_meth = mofa.project(test_views, missing=["rna", "mutation"])
    # The shared factor is the one that explains variance in BOTH views.
    both = [mofa.views.index("rna"), mofa.views.index("methylation")]
    top = int(np.argmax(mofa.r2[both][:, mofa.active].min(axis=0)))
    assert abs(np.corrcoef(from_rna[:, top], from_meth[:, top])[0, 1]) > 0.5
    assert abs(np.corrcoef(full[:, top], from_rna[:, top])[0, 1]) > 0.7
    # No view at all gives the prior mean.
    assert np.allclose(mofa.project(test_views, missing=list(views)), 0.0)

"""Tests of the survival models and of the two integration methods.

Three things are checked on the synthetic cohort (see conftest.py), where the
truth is known because it was put in:

    the models     penalised Cox ("coxnet") and the random survival forest
                   ("rsf") give HIGHER risk to patients who do worse.
    late fusion    one model per block, risks averaged; a missing block's
                   model is left out.
    MOFA           finds the hidden variable that expression and methylation
                   share, also for new patients and with a block missing.

Words used here
    risk         the number a model gives a patient; higher means a worse
                 prognosis is predicted.
    concordance  (C) the share of patient pairs the risks put in the right
                 order: 1.0 is perfect, 0.5 is chance, below 0.5 is worse
                 than chance (the order is the wrong way round).
    factor       one of the hidden variables MOFA learns; each patient gets a
                 value on each factor.
    view         MOFA's word for a block.
    projection   computing a patient's factor values from their data with an
                 already fitted MOFA model, without fitting again.
"""
import numpy as np

from luad import models as M
from luad.integrate import fit_mofa, late_fusion
from luad.transforms import build_blocks, concat


def _setup(cohort, cfg):
    """What every test here needs: one split of the patients and the blocks.

    Returns (training rows, test rows, inner cross-validation splits of the
    training rows, {block name: Block} built on the training rows).
    """
    shuffled_rows = np.random.default_rng(1).permutation(cohort.n)
    train_rows, test_rows = np.sort(shuffled_rows[:90]), np.sort(shuffled_rows[90:])
    return (train_rows, test_rows, M.inner_splits(cohort.y, train_rows, 3, 0),
            build_blocks(cohort, train_rows, cfg, 0, 20))


def test_higher_risk_means_worse_prognosis(cohort, cfg):
    """Stage drives the synthetic hazard, so both models must score above 0.5
    on held-out rows. A flipped sign would give well below 0.5.

    Catches: a model whose output means the opposite of risk (for example a
    predicted survival time, where higher is better), or a model that learns
    nothing. Every concordance in the project assumes "higher risk, earlier
    event"; with the sign flipped, a good model would be reported as worse
    than chance.
    """
    train_rows, test_rows, splits, blocks = _setup(cohort, cfg)
    X, _ = concat(blocks, ["clinical"])
    for model in M.MODELS:
        fitted = M.tune_and_fit(model, X, cohort.y, train_rows, splits, cfg, 0, 1, quick=True)
        assert M.harrell_c(cohort.y[test_rows], fitted.predict(X[test_rows])) > 0.6, model


def test_late_fusion_weights_and_missing_blocks(cohort, cfg):
    """Late fusion with a block missing must use the remaining blocks only,
    and with all blocks present must give one finite risk per patient.

    Catches: a missing block's model still taking part in the average, or the
    rescaling of the risks changing their order. If either happened, the
    missing-block results for late fusion would describe something else than
    what the report says.
    """
    train_rows, test_rows, splits, blocks = _setup(cohort, cfg)
    # Two single-block models: one on clinical, one on expression.
    fits, risks = {}, {}
    for block in ("clinical", "rna"):
        X, _ = concat(blocks, [block])
        fits[block] = M.tune_and_fit("coxnet", X, cohort.y, train_rows, splits, cfg, 0, 1,
                                     quick=True)
        risks[block] = fits[block].predict(X[test_rows])
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
    views and with either one missing.

    Catches: a projection of new patients that is wrong (factor values that
    do not match what the model learned), or one that falls apart when a view
    is missing. The MOFA arm and all its missing-block scenarios, and the
    external cohorts, depend on this projection.
    """
    train_rows, test_rows, _, _ = _setup(cohort, cfg)
    # MOFA gets wider blocks than the models do, as in the real pipeline.
    wide = build_blocks(cohort, train_rows, cfg, 0, 100)
    views = {v: wide[v].X for v in ("rna", "mutation", "methylation")}
    mofa = fit_mofa({v: views[v][train_rows] for v in views}, cfg, 0)
    # At least one factor must have been kept.
    assert len(mofa.active) >= 1

    # The factor projected from expression alone and from methylation alone
    # must agree, since both views were generated from the same variable.
    test_views = {v: views[v][test_rows] for v in views}
    full = mofa.project(test_views)
    from_rna = mofa.project(test_views, missing=["methylation", "mutation"])
    from_meth = mofa.project(test_views, missing=["rna", "mutation"])
    # The shared factor is the one that explains variance in BOTH views.
    both = [mofa.views.index("rna"), mofa.views.index("methylation")]
    top = int(np.argmax(mofa.r2[both][:, mofa.active].min(axis=0)))
    # The sign of a factor is arbitrary, so the size of the correlation is
    # what is tested, not its sign.
    assert abs(np.corrcoef(from_rna[:, top], from_meth[:, top])[0, 1]) > 0.5
    assert abs(np.corrcoef(full[:, top], from_rna[:, top])[0, 1]) > 0.7
    # No view at all gives the prior mean.
    assert np.allclose(mofa.project(test_views, missing=list(views)), 0.0)

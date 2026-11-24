"""A small synthetic cohort so the tests run in seconds and need no data.

What this file is for
    pytest reads conftest.py before any test. What is defined here is shared by
    all the test files:

    synthetic  builds a made-up cohort of 120 patients with the same four
               blocks as the real one, only much smaller.
    cohort     a fixture that hands that cohort to a test.
    cfg        a fixture that hands the project configuration to a test, with
               the slow settings turned down.

    A fixture is pytest's way of giving a test what it needs: a test that has
    an argument called `cohort` receives what the function `cohort` below
    returns. scope="session" means it is built once and reused by every test.

Why a synthetic cohort
    The answer is known, because it was put in. Survival is made to depend on
    tumour stage and on one hidden variable, and that hidden variable also
    leaves a trace in expression and in methylation. The tests can then check
    that the models find the stage effect and that MOFA finds the shared
    variable. It also means the tests need none of the real data files.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Make `import luad` work without installing the package: add src/ to the
# places Python looks for modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luad import load_config  # noqa: E402
from luad.data import Cohort, make_y  # noqa: E402


def synthetic(n=120, seed=0) -> Cohort:
    """Build a made-up cohort with a known link between the data and survival.

    n     number of patients.
    seed  random seed; the same seed always gives the same cohort.

    Returns a Cohort with all four blocks: 4 clinical covariates, 300 genes,
    400 methylation probes, and 200 mutated genes plus 3 mutation summaries.
    """
    rng = np.random.default_rng(seed)

    # ---- Survival. The hazard (the rate of events) rises with stage and with
    # one hidden variable, `latent`. Each patient gets a time to the event and
    # a time at which follow-up stops (censoring); whichever comes first is
    # what is observed. Nothing is followed beyond 1826 days (5 years).
    stage = rng.integers(1, 5, n).astype(float)
    latent = rng.normal(size=n)                      # one shared omics factor
    hazard = np.exp(0.6 * stage + 0.5 * latent)
    t_event = rng.exponential(2000 / hazard)
    t_cens = rng.exponential(1500, n)
    time = np.minimum(np.minimum(t_event, t_cens), 1826.0) + 1
    event = (t_event <= t_cens) & (t_event < 1826)

    # ---- Clinical block: age, sex, stage and one covariate of pure noise.
    # A tenth of the noise covariate is made missing so that the imputation
    # has something to do.
    clinical = np.column_stack([rng.normal(65, 10, n), rng.integers(0, 2, n), stage,
                                rng.normal(size=n)])
    clinical[rng.random(n) < 0.1, 3] = np.nan

    # ---- Omics blocks. About a fifth of the genes and of the probes follow
    # the hidden variable; the rest are noise. Mutations are unrelated to it.
    n_genes, n_probes, n_mut_genes = 300, 400, 200
    rna = (latent[:, None] * rng.normal(size=n_genes) * (rng.random(n_genes) < 0.2)
           + rng.normal(size=(n, n_genes)))
    # Expression is stored as within-patient ranks in (0, 1], as in the real data.
    rna = (rna.argsort(1).argsort(1) + 1) / n_genes
    # Methylation is squeezed into (0, 1), like real beta values.
    meth = 1 / (1 + np.exp(-(latent[:, None] * rng.normal(size=n_probes)
                             * (rng.random(n_probes) < 0.2)
                             + rng.normal(size=(n, n_probes)))))
    # Each gene is mutated in 1% to 20% of patients, at random.
    mut = (rng.random((n, n_mut_genes)) < rng.uniform(0.01, 0.2, n_mut_genes)).astype(np.float32)
    return Cohort(
        patients=np.array([f"P{i:03d}" for i in range(n)]), y=make_y(event, time),
        clinical=clinical, clinical_names=["age", "sex_male", "stage_ordinal", "other"],
        rna=rna.astype(np.float32),
        rna_counts=rng.poisson(50, (n, n_genes)).astype(np.float32),
        rna_genes=np.array([f"G{i}" for i in range(n_genes)]),
        mut_gene=mut, mut_gene_names=np.array([f"M{i}" for i in range(n_mut_genes)]),
        mut_feat=rng.normal(size=(n, 3)).astype(np.float32),
        mut_feat_names=["tmb_log10", "pw_a", "pw_b"],
        meth=meth.astype(np.float32), meth_probes=np.array([f"cg{i}" for i in range(n_probes)]))


@pytest.fixture(scope="session")
def cohort():
    """The synthetic cohort, built once and shared by all tests. Tests must
    not change it in place; a test that needs a changed cohort makes a copy."""
    return synthetic()


@pytest.fixture(scope="session")
def cfg():
    """The project configuration with small settings, so the tests are fast:
    fewer MOFA factors and iterations, narrower blocks, one candidate width."""
    config = load_config()
    config["mofa"].update(factors=4, top_k=100, iterations=60)
    config["features"]["k_grid"] = [20]
    return config

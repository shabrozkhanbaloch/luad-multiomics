"""A small synthetic cohort so the tests run in seconds and need no data."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luad import load_config  # noqa: E402
from luad.data import Cohort, make_y  # noqa: E402


def synthetic(n=120, seed=0) -> Cohort:
    rng = np.random.default_rng(seed)
    stage = rng.integers(1, 5, n).astype(float)
    latent = rng.normal(size=n)                      # one shared omics factor
    hazard = np.exp(0.6 * stage + 0.5 * latent)
    t_event = rng.exponential(2000 / hazard)
    t_cens = rng.exponential(1500, n)
    time = np.minimum(np.minimum(t_event, t_cens), 1826.0) + 1
    event = (t_event <= t_cens) & (t_event < 1826)
    clinical = np.column_stack([rng.normal(65, 10, n), rng.integers(0, 2, n), stage,
                                rng.normal(size=n)])
    clinical[rng.random(n) < 0.1, 3] = np.nan
    g, p, m = 300, 400, 200
    rna = (latent[:, None] * rng.normal(size=g) * (rng.random(g) < 0.2)
           + rng.normal(size=(n, g)))
    rna = (rna.argsort(1).argsort(1) + 1) / g
    meth = 1 / (1 + np.exp(-(latent[:, None] * rng.normal(size=p) * (rng.random(p) < 0.2)
                             + rng.normal(size=(n, p)))))
    mut = (rng.random((n, m)) < rng.uniform(0.01, 0.2, m)).astype(np.float32)
    return Cohort(
        patients=np.array([f"P{i:03d}" for i in range(n)]), y=make_y(event, time),
        clinical=clinical, clinical_names=["age", "sex_male", "stage_ordinal", "other"],
        rna=rna.astype(np.float32),
        rna_counts=rng.poisson(50, (n, g)).astype(np.float32),
        rna_genes=np.array([f"G{i}" for i in range(g)]),
        mut_gene=mut, mut_gene_names=np.array([f"M{i}" for i in range(m)]),
        mut_feat=rng.normal(size=(n, 3)).astype(np.float32),
        mut_feat_names=["tmb_log10", "pw_a", "pw_b"],
        meth=meth.astype(np.float32), meth_probes=np.array([f"cg{i}" for i in range(p)]))


@pytest.fixture(scope="session")
def cohort():
    return synthetic()


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    c["mofa"].update(factors=4, top_k=100, iterations=60)
    c["features"]["k_grid"] = [20]
    return c

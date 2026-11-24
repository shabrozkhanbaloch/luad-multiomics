"""Load the four blocks and the outcome for one cohort, aligned by patient."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 365.25 / 12.0
OMICS = ("rna", "mutation", "methylation")


@dataclass
class Cohort:
    """Every array shares the row order of `patients`."""
    patients: np.ndarray        # patient barcodes
    y: np.ndarray               # structured (event: bool, time: float, days)
    clinical: np.ndarray        # patients x covariates, NaN where missing
    clinical_names: list[str]
    rna: np.ndarray             # patients x genes, within-sample ranks in (0, 1]
    rna_counts: np.ndarray      # patients x genes, raw counts (for the expression filter)
    rna_genes: np.ndarray       # gene symbols
    mut_gene: np.ndarray        # patients x genes, 0/1 non-silent mutation
    mut_gene_names: np.ndarray
    mut_feat: np.ndarray        # patients x (TMB + pathway indicators)
    mut_feat_names: list[str]
    meth: np.ndarray            # patients x probes, beta values
    meth_probes: np.ndarray

    @property
    def n(self) -> int:
        return len(self.patients)

    def with_outcome(self, y: np.ndarray) -> "Cohort":
        return replace(self, y=y)


def make_y(event, time) -> np.ndarray:
    y = np.empty(len(time), dtype=[("event", bool), ("time", float)])
    y["event"], y["time"] = np.asarray(event, bool), np.asarray(time, float)
    return y


def censor_at(event, time, horizon_days: float):
    """Administrative censoring: nothing after the horizon counts."""
    event = np.asarray(event, bool) & (np.asarray(time, float) <= horizon_days)
    return event, np.minimum(np.asarray(time, float), horizon_days)


def cohort_flow(cfg: dict, endpoint: str | None = None) -> pd.DataFrame:
    """How the analysis cohort is reached: one row per inclusion rule, applied
    in order, with the patients left and the patients removed by that rule."""
    import pyarrow.parquet as pq

    endpoint = (endpoint or cfg["endpoint"]["primary"]).lower()
    proc, interim = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["interim"])

    def ids(name: str, by_column: bool) -> set[str]:
        if by_column:                       # one column per patient: read the names only
            return set(pq.ParquetFile(proc / name).schema.names[1:])
        return set(pd.read_parquet(proc / name, columns=["patient_barcode"])["patient_barcode"])

    endp = pd.read_parquet(interim / "endpoints.parquet").set_index("patient_barcode")
    ev, tm = f"{endpoint}_event", f"{endpoint}_time"
    left = set(endp.index[endp[ev].notna() & endp[tm].notna() & (endp[tm] > 0)])
    rows = [{"step": f"TCGA-LUAD patients with a usable {endpoint.upper()} endpoint",
             "patients": len(left), "excluded": 0}]
    for label, name, by_column in (("clinical covariates", "clinical_features.parquet", False),
                                   ("expression", "rna_rank.parquet", True),
                                   ("mutation", "mut_gene_binary.parquet", False),
                                   ("methylation", "meth_beta.parquet", True)):
        kept = left & ids(name, by_column)
        rows.append({"step": f"with {label}", "patients": len(kept), "excluded": len(left) - len(kept)})
        left = kept
    return pd.DataFrame(rows)


def load_cohort(cfg: dict, endpoint: str | None = None, verbose: bool = True) -> Cohort:
    """The analysis cohort: patients with an endpoint and all four blocks."""
    endpoint = (endpoint or cfg["endpoint"]["primary"]).lower()
    proc, interim = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["interim"])

    rank = pd.read_parquet(proc / "rna_rank.parquet").set_index("gene_id")
    counts = pd.read_parquet(proc / "rna_counts.parquet").set_index("gene_id")
    meta = pd.read_parquet(proc / "rna_gene_meta.parquet").set_index("gene_id")
    mut_gene = pd.read_parquet(proc / "mut_gene_binary.parquet").set_index("patient_barcode")
    mut_feat = pd.read_parquet(proc / "mut_features.parquet").set_index("patient_barcode")
    clinical = pd.read_parquet(proc / "clinical_features.parquet").set_index("patient_barcode")
    meth = pd.read_parquet(proc / "meth_beta.parquet").set_index("probe_id")
    endp = pd.read_parquet(interim / "endpoints.parquet").set_index("patient_barcode")

    ev, tm = f"{endpoint}_event", f"{endpoint}_time"
    endp = endp[endp[ev].notna() & endp[tm].notna() & (endp[tm] > 0)]

    keep = (set(endp.index) & set(rank.columns) & set(mut_gene.index)
            & set(clinical.index) & set(meth.columns))
    patients = np.array(sorted(keep))
    if len(patients) == 0:
        raise SystemExit("no patient has an endpoint and all four blocks")

    horizon = float(cfg["endpoint"]["admin_censor_months"]) * DAYS_PER_MONTH
    event, time = censor_at(endp.loc[patients, ev], endp.loc[patients, tm], horizon)

    feat_cols = [c for c in mut_feat.columns if c == "tmb_log10" or c.startswith("pw_")]
    symbols = meta["gene_name"].reindex(rank.index).fillna(pd.Series(rank.index, rank.index))

    cohort = Cohort(
        patients=patients,
        y=make_y(event, time),
        clinical=clinical.loc[patients].to_numpy(np.float64),
        clinical_names=list(clinical.columns),
        rna=rank[patients].to_numpy(np.float32).T,
        rna_counts=counts.loc[rank.index, patients].to_numpy(np.float32).T,
        rna_genes=symbols.to_numpy(str),
        mut_gene=mut_gene.loc[patients].to_numpy(np.float32),
        mut_gene_names=mut_gene.columns.to_numpy(str),
        mut_feat=mut_feat.loc[patients, feat_cols].to_numpy(np.float32),
        mut_feat_names=feat_cols,
        meth=meth[patients].to_numpy(np.float32).T,
        meth_probes=meth.index.to_numpy(str),
    )
    if verbose:
        print(f"  cohort   : {cohort.n} patients with an endpoint and all four blocks")
        print(f"  endpoint : {endpoint.upper()}, censored at "
              f"{cfg['endpoint']['admin_censor_months']} months; "
              f"{int(event.sum())} events ({event.mean():.1%})")
    return cohort

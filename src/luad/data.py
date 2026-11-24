"""Load the four blocks and the outcome for one cohort, aligned by patient.

What this file is for
    Everything downstream works on one object, a `Cohort`: the list of
    patients, their outcome, and one matrix per data block. In every matrix,
    row i is patient i. This file builds that object from the prepared files
    in data/processed and data/interim.

The four blocks
    clinical      age, stage, sex and so on (10 covariates)
    rna           gene expression
    mutation      which genes are mutated, plus tumour mutation burden (TMB)
                  and pathway indicators
    methylation   DNA methylation at CpG probes ("beta values", between 0 and 1)

The outcome, and what "censored" means
    The outcome is a time and a yes/no event. For progression-free interval
    the event is progression or death. A patient who was last seen without
    the event is "censored": we know only that the event had not happened by
    that time. Survival methods use both kinds of patient.

    All patients are also censored at a fixed horizon (60 months in the
    configuration). An event after the horizon is treated as "no event yet at
    the horizon". This is called administrative censoring.

Main functions
    load_cohort   the analysis cohort: patients with an endpoint and all four blocks
    cohort_flow   how many patients each inclusion rule removes (for the flow diagram)
    make_y        pack event and time into the array format the survival library expects
    censor_at     apply the administrative censoring

Who is in the cohort
    A patient is kept only if they have a usable endpoint and all four blocks.
    This is why every arm is compared on exactly the same patients.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

# Times in the data are in days; the censoring horizon is given in months.
DAYS_PER_MONTH = 365.25 / 12.0

# The three blocks that are "omics". The fourth block is clinical.
OMICS = ("rna", "mutation", "methylation")


@dataclass
class Cohort:
    """One cohort of patients: the outcome and the four blocks.

    Every array shares the row order of `patients`: row i of every matrix is
    patient i. Nothing in the pipeline reorders the rows; a fold is described
    by row numbers into these arrays.

    The mutation block is stored as two matrices, `mut_gene` (one 0/1 column
    per gene) and `mut_feat` (tumour mutation burden and pathway indicators),
    because only the gene columns are filtered inside a fold.
    """
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
        """The number of patients."""
        return len(self.patients)

    def with_outcome(self, y: np.ndarray) -> "Cohort":
        """A copy of this cohort with a different outcome and the same blocks.

        y is a structured array as made by `make_y`, one entry per patient.
        Used by the shuffled-outcome check, where the outcome is permuted so
        that no model should be able to predict it.
        """
        return replace(self, y=y)


def make_y(event, time) -> np.ndarray:
    """Pack event and time into one structured array, the format scikit-survival expects.

    event
        One True/False (or 1/0) per patient: True if the event was observed,
        False if the patient is censored.
    time
        One number per patient: days to the event or to censoring.

    Returns an array of length patients with two named fields, so that
    `y["event"]` is the boolean array and `y["time"]` is the float array.
    """
    y = np.empty(len(time), dtype=[("event", bool), ("time", float)])
    y["event"], y["time"] = np.asarray(event, bool), np.asarray(time, float)
    return y


def censor_at(event, time, horizon_days: float):
    """Administrative censoring: nothing after the horizon counts.

    event, time
        As in `make_y`: one entry per patient.
    horizon_days
        The horizon, in days.

    Returns (event, time). A patient whose time is beyond the horizon becomes
    "no event, followed for exactly the horizon". Patients inside the horizon
    are unchanged.
    """
    # An event counts only if it happened by the horizon.
    event = np.asarray(event, bool) & (np.asarray(time, float) <= horizon_days)
    # No follow-up time is longer than the horizon.
    return event, np.minimum(np.asarray(time, float), horizon_days)


def cohort_flow(cfg: dict, endpoint: str | None = None) -> pd.DataFrame:
    """How the analysis cohort is reached: one row per inclusion rule, applied
    in order, with the patients left and the patients removed by that rule.

    cfg
        The configuration, from `load_config`.
    endpoint
        "pfi" or "os" (any case). Left empty, it is the primary endpoint of
        the configuration.

    Returns a table with the columns `step` (the rule, in words), `patients`
    (how many remain after it) and `excluded` (how many it removed). The last
    row has the same number of patients as `load_cohort` returns.

    Only patient identifiers are read, not the data, so this is fast.
    """
    import pyarrow.parquet as pq

    endpoint = (endpoint or cfg["endpoint"]["primary"]).lower()
    processed_dir, interim_dir = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["interim"])

    def ids(name: str, by_column: bool) -> set[str]:
        """The patients present in one file of data/processed.

        by_column says how the file is laid out: True if each patient is a
        column (expression, methylation), False if each patient is a row.
        """
        if by_column:                       # one column per patient: read the names only
            return set(pq.ParquetFile(processed_dir / name).schema.names[1:])
        return set(pd.read_parquet(processed_dir / name, columns=["patient_barcode"])["patient_barcode"])

    # Start from the patients with a usable endpoint: event and time both
    # recorded, and a follow-up time above zero.
    endpoints = pd.read_parquet(interim_dir / "endpoints.parquet").set_index("patient_barcode")
    event_column, time_column = f"{endpoint}_event", f"{endpoint}_time"
    left = set(endpoints.index[endpoints[event_column].notna() & endpoints[time_column].notna()
                               & (endpoints[time_column] > 0)])
    rows = [{"step": f"TCGA-LUAD patients with a usable {endpoint.upper()} endpoint",
             "patients": len(left), "excluded": 0}]

    # Then require each block in turn, and count who is lost at each step.
    for label, name, by_column in (("clinical covariates", "clinical_features.parquet", False),
                                   ("expression", "rna_rank.parquet", True),
                                   ("mutation", "mut_gene_binary.parquet", False),
                                   ("methylation", "meth_beta.parquet", True)):
        kept = left & ids(name, by_column)
        rows.append({"step": f"with {label}", "patients": len(kept), "excluded": len(left) - len(kept)})
        left = kept
    return pd.DataFrame(rows)


def load_cohort(cfg: dict, endpoint: str | None = None, verbose: bool = True) -> Cohort:
    """The analysis cohort: patients with an endpoint and all four blocks.

    cfg
        The configuration, from `load_config`.
    endpoint
        "pfi" or "os" (any case). Left empty, it is the primary endpoint of
        the configuration.
    verbose
        If True, print the size of the cohort and the number of events.

    Returns a `Cohort`. Patients are in alphabetical order of barcode, so the
    row order is the same on every run.
    """
    endpoint = (endpoint or cfg["endpoint"]["primary"]).lower()
    processed_dir, interim_dir = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["interim"])

    # Read every file. The expression and methylation files have one row per
    # gene or probe and one column per patient; the others have one row per patient.
    rank = pd.read_parquet(processed_dir / "rna_rank.parquet").set_index("gene_id")
    counts = pd.read_parquet(processed_dir / "rna_counts.parquet").set_index("gene_id")
    gene_meta = pd.read_parquet(processed_dir / "rna_gene_meta.parquet").set_index("gene_id")
    mut_gene = pd.read_parquet(processed_dir / "mut_gene_binary.parquet").set_index("patient_barcode")
    mut_feat = pd.read_parquet(processed_dir / "mut_features.parquet").set_index("patient_barcode")
    clinical = pd.read_parquet(processed_dir / "clinical_features.parquet").set_index("patient_barcode")
    meth = pd.read_parquet(processed_dir / "meth_beta.parquet").set_index("probe_id")
    endpoints = pd.read_parquet(interim_dir / "endpoints.parquet").set_index("patient_barcode")

    # A usable endpoint: event and time both recorded, and a follow-up time above zero.
    event_column, time_column = f"{endpoint}_event", f"{endpoint}_time"
    endpoints = endpoints[endpoints[event_column].notna() & endpoints[time_column].notna()
                          & (endpoints[time_column] > 0)]

    # Keep the patients found in the endpoint table and in all four blocks.
    # Sorting fixes the row order of every array for the rest of the pipeline.
    keep = (set(endpoints.index) & set(rank.columns) & set(mut_gene.index)
            & set(clinical.index) & set(meth.columns))
    patients = np.array(sorted(keep))
    if len(patients) == 0:
        raise SystemExit("no patient has an endpoint and all four blocks")

    # Censor everyone at the horizon (given in months, the data are in days).
    horizon = float(cfg["endpoint"]["admin_censor_months"]) * DAYS_PER_MONTH
    event, time = censor_at(endpoints.loc[patients, event_column],
                            endpoints.loc[patients, time_column], horizon)

    # From the mutation feature table, use tumour mutation burden and the
    # pathway indicators (columns starting with "pw_") and nothing else.
    feat_cols = [c for c in mut_feat.columns if c == "tmb_log10" or c.startswith("pw_")]
    # Name each gene by its symbol; where a gene has no symbol, keep its identifier.
    symbols = gene_meta["gene_name"].reindex(rank.index).fillna(pd.Series(rank.index, rank.index))

    # Put every block in the same patient order. Expression and methylation
    # are turned (.T) so that patients become the rows.
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

#!/usr/bin/env python
"""Step 10. Survival curves for patients grouped by their predicted risk.

    python scripts/10_survival_curves.py                # primary endpoint
    python scripts/10_survival_curves.py --endpoint os

ADDED AFTER THE FIRST RUN. This step is not part of the plan that was fixed in
config/config.yaml before the analysis. It was added on 24 November 2026, after
the results were known, as a description of what the models' discrimination
looks like in practice. No claim of the report rests on it, and it is not one
of the pre-declared contrasts.

What it is for
    A concordance of 0.60 is hard to picture. This step shows it: the patients
    are split into three equal groups by the risk the model gave them (low,
    middle, high), and a Kaplan-Meier survival curve is drawn for each group.
    The further apart the three curves, the better the model separates
    patients.

    Every risk used here comes from a model that had never seen that patient:
    it is the risk the cross-validation of step 1 gave the patient when the
    patient was in the test fold. Nothing is fitted in this step.

How one risk per patient is made
    In step 1 every patient is tested five times (five repeats of five folds),
    each time by a different model, and risks from different models are not on
    the same scale. So inside each test fold a patient's risk is first turned
    into its rank among that fold's test patients, as a share from 0 to 1. The
    five shares of a patient are then averaged. Patients are sorted by that
    average and cut into three equal groups.

What it reads
    config/config.yaml                                which arms, how many groups, which months
    results/tables/cv_<endpoint>_predictions.parquet  the risks step 1 wrote
    data/processed/, data/interim/                    the outcome of each patient

What it writes
    results/tables/survival_curves_<endpoint>.csv  every step of every curve: time in months,
                                                   share still event-free, 95% interval,
                                                   patients still at risk
    results/tables/survival_groups_<endpoint>.csv  one row per group: patients, events,
                                                   event-free share at the chosen months, and
                                                   the log-rank test across the three groups

How long it takes
    A few seconds.

Words used here
    Kaplan-Meier curve  the share of patients who have not yet had the event,
                        over time, allowing for patients whose follow-up ended
                        early (censored patients)
    log-rank test       a test of whether the curves of several groups differ
                        by more than chance would give. Here it is a
                        description, not a planned test.
"""
from __future__ import annotations

import argparse

from _common import load_config, out_dirs

import numpy as np
import pandas as pd
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator

from luad.data import DAYS_PER_MONTH, load_cohort
from luad.models import MODELS

GROUP_NAMES = {3: ["low", "middle", "high"]}


def one_risk_per_patient(predictions: pd.DataFrame, model: str, arm: str, n_patients: int) -> np.ndarray:
    """The average rank of each patient's out-of-fold risk, from 0 (lowest) to 1 (highest).

    Arguments
        predictions  the table step 1 wrote: fold, model, arm, patient, risk
        model        "coxnet" or "rsf"
        arm          for example "clinical" or "mofa"
        n_patients   size of the cohort; patients are numbered 0 to n_patients - 1

    Returns an array with one value per patient.
    """
    rows = predictions[(predictions["model"] == model) & (predictions["arm"] == arm)].copy()
    # Rank inside each test fold, because every fold has its own model and the
    # risks of two models cannot be compared directly. `pct=True` gives the
    # rank as a share; tied risks share the average rank.
    rows["rank_share"] = rows.groupby("fold")["risk"].rank(pct=True, method="average")
    average = rows.groupby("patient")["rank_share"].mean()
    # Every patient must have been tested; a gap would silently shrink a group.
    assert len(average) == n_patients, f"{arm}/{model}: {len(average)} of {n_patients} patients have a risk"
    return average.reindex(range(n_patients)).to_numpy()


def split_into_groups(score: np.ndarray, n_groups: int) -> np.ndarray:
    """Cut the patients into n_groups groups of equal size by their score.

    Returns an array of group numbers, 0 for the lowest scores. Ties are broken
    by patient order, so the groups differ in size by at most one patient.
    """
    order = pd.Series(score).rank(method="first").to_numpy()
    return np.minimum(((order - 1) * n_groups / len(score)).astype(int), n_groups - 1)


def describe_group(event: np.ndarray, months: np.ndarray, at_months: list[int]) -> tuple[pd.DataFrame, dict]:
    """The Kaplan-Meier curve of one group and its summary numbers.

    Arguments
        event      True where the patient had the event
        months     time to the event or to the end of follow-up, in months
        at_months  the time points to report, for example [36, 60]

    Returns
        curve    one row per step of the curve
        summary  patients, events, and the event-free share at each time point
    """
    time, survival, interval = kaplan_meier_estimator(event, months, conf_type="log-log")
    # How many patients are still being followed just before each step.
    at_risk = np.array([(months >= t).sum() for t in time])
    curve = pd.DataFrame({"months": time, "event_free": survival,
                          "ci_low": interval[0], "ci_high": interval[1], "at_risk": at_risk})

    summary = {"patients": len(event), "events": int(event.sum())}
    for month in at_months:
        # The curve is a staircase: its value at a time point is the value of
        # the last step at or before it. Before the first step everyone is
        # still event-free.
        reached = np.flatnonzero(time <= month)
        if len(reached):
            step = reached[-1]
            summary[f"event_free_{month}m"] = survival[step]
            summary[f"event_free_{month}m_low"] = interval[0][step]
            summary[f"event_free_{month}m_high"] = interval[1][step]
        else:
            summary[f"event_free_{month}m"] = 1.0
            summary[f"event_free_{month}m_low"] = 1.0
            summary[f"event_free_{month}m_high"] = 1.0
    return curve, summary


def main() -> int:
    """Group the patients, describe each group, write the two tables."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default=None,
                        help="endpoint to describe, pfi or os (default: the primary one in the config)")
    args = parser.parse_args()

    # ---- Read the plan ----
    cfg = load_config()
    settings = cfg["added_after_first_run"]["survival_curves"]
    endpoint = (args.endpoint or cfg["endpoint"]["primary"]).lower()
    n_groups = int(settings["groups"])
    group_names = GROUP_NAMES[n_groups]
    tables, _ = out_dirs(cfg)

    # ---- Load the data ----
    print(f"SURVIVAL CURVES BY PREDICTED RISK  ({endpoint})")
    cohort = load_cohort(cfg, endpoint)
    event = cohort.y["event"]
    months = cohort.y["time"] / DAYS_PER_MONTH
    predictions = pd.read_parquet(tables / f"cv_{endpoint}_predictions.parquet")

    # ---- Compute: one set of curves per model family and arm ----
    curves, groups = [], []
    for model in MODELS:
        for arm in settings["arms"]:
            score = one_risk_per_patient(predictions, model, arm, cohort.n)
            group = split_into_groups(score, n_groups)

            # Do the three curves differ by more than chance would give?
            chi_squared, p_value = compare_survival(cohort.y, group)

            for number, name in enumerate(group_names):
                members = group == number
                curve, summary = describe_group(event[members], months[members], settings["months"])
                curves.append(curve.assign(model=model, arm=arm, group=name))
                groups.append({"model": model, "arm": arm, "group": name, **summary,
                               "logrank_chi2": chi_squared, "logrank_p": p_value})

    # ---- Write the tables ----
    curves = pd.concat(curves, ignore_index=True)[
        ["model", "arm", "group", "months", "event_free", "ci_low", "ci_high", "at_risk"]]
    groups = pd.DataFrame(groups)
    curves.to_csv(tables / f"survival_curves_{endpoint}.csv", index=False)
    groups.to_csv(tables / f"survival_groups_{endpoint}.csv", index=False)

    # ---- Print the summary ----
    print(groups.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\n  wrote {tables}/survival_curves_{endpoint}.csv and survival_groups_{endpoint}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

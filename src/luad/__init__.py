"""LUAD multi-omics integration: prediction, integration, biology.

This is the library behind the numbered scripts in `scripts/`. The scripts
decide what to run and where to write it; the modules here do the work.

The modules, in the order the data passes through them
    data.py        load the four data blocks and the outcome, aligned by patient
    transforms.py  choose features and fill in missing values, on training rows only
    models.py      penalised Cox and random survival forest, tuned by inner cross-validation
    integrate.py   late fusion and MOFA
    cv.py          one cross-validation fold: every arm, every scenario
    bootstrap.py   intervals for the scores and for the differences between arms
    external.py    apply the models to the GEO cohorts
    biology.py     gene lists, the protein network and its modules

What this file itself provides
    ROOT           the folder of the repository, so every path can be written from it
    load_config    read the one configuration file, config/config.yaml

The vocabulary (block, arm, scenario, contrast) is explained in README.md.
"""
from pathlib import Path

import yaml

# This file is <repository>/src/luad/__init__.py, so two folders up is the repository.
ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Read the configuration file and return it as a dictionary.

    path
        The YAML file to read. Left empty, it is config/config.yaml in the
        repository.

    Returns the configuration exactly as written in the file, with one change:
    every entry under `paths` is turned from a path relative to the repository
    (for example `results/tables`) into a full path, as text. The scripts can
    then be started from any folder.
    """
    cfg = yaml.safe_load(open(path or ROOT / "config" / "config.yaml"))

    # The file gives paths relative to the repository; make them absolute.
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = str(ROOT / rel)
    return cfg

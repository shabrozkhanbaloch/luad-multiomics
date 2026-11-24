"""LUAD multi-omics integration: prediction, integration, biology."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    cfg = yaml.safe_load(open(path or ROOT / "config" / "config.yaml"))
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = str(ROOT / rel)
    return cfg

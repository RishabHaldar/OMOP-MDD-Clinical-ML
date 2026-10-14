"""Load the YAML configuration files in config/."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
SQL_DIR = REPO_ROOT / "sql"


def _load(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=None)
def concept_sets() -> dict:
    return _load("concept_sets.yaml")


@lru_cache(maxsize=None)
def analysis() -> dict:
    return _load("analysis.yaml")


@lru_cache(maxsize=None)
def features() -> dict:
    return _load("features.yaml")


def analytes() -> list[str]:
    """The 16 laboratory analytes, in configuration order."""
    return list(concept_sets()["labs"].keys())


def predictors() -> list[str]:
    """The 18-column predictor allowlist."""
    return list(features()["predictors"])

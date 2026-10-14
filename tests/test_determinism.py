"""Fixed seeds give identical results."""

from __future__ import annotations

from omop_mdd.run import run


def _fast(seed: int) -> dict:
    return run(seed=seed, out=None, make_figures=False, with_importance=False)


def test_same_seed_same_results():
    a, b = _fast(5), _fast(5)
    assert a["split"] == b["split"]
    assert a["cv"] == b["cv"]
    assert a["test"] == b["test"]


def test_different_seed_changes_results():
    a, b = _fast(5), _fast(6)
    assert a["test"] != b["test"]

"""Unit tests for the ES-14 visualization module.

Tests the pure data-shaping helpers (inline synthetic dicts/DataFrames, no file
I/O — repo convention) and a smoke test per JSON-based plot function that asserts
a PNG is written under ``tmp_path`` (the Agg backend keeps this CI-safe). The
figure-4 helper reads the real gitignored parquet, so its test is marked ``data``
and excluded in CI via ``-m "not data"``.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import visualize as viz


# --------------------------------------------------------------------------- #
# Synthetic records mirroring the committed results files' schema
# --------------------------------------------------------------------------- #
def _correlation_records():
    return [
        {"analysis": "primary", "return_col": "abn_return_1d", "estimate": 0.104,
         "ci_low": 0.087, "ci_high": 0.120, "n": 13429},
        {"analysis": "baseline_floor", "return_col": "abn_return_1d", "estimate": -0.002,
         "ci_low": -0.019, "ci_high": 0.015, "n": 13429},
        {"analysis": "difference", "return_col": "abn_return_1d", "estimate": 0.106,
         "ci_low": 0.082, "ci_high": 0.129, "n": 13429},
        {"analysis": "robustness", "return_col": "return_1d", "estimate": 0.103,
         "ci_low": 0.087, "ci_high": 0.120, "n": 13429},
        {"analysis": "robustness", "return_col": "abn_return_1d", "estimate": 0.104,
         "ci_low": 0.087, "ci_high": 0.120, "n": 13429},
        {"analysis": "robustness", "return_col": "return_5d", "estimate": 0.083,
         "ci_low": 0.066, "ci_high": 0.100, "n": 13507},
        {"analysis": "robustness", "return_col": "abn_return_5d", "estimate": 0.090,
         "ci_low": 0.073, "ci_high": 0.107, "n": 13507},
    ]


def _metrics():
    return {
        "confusion_matrix": [[207, 6, 1], [3, 85, 1], [2, 3, 37]],
        "per_class": {
            "neutral": {"precision": 0.976, "recall": 0.967, "f1-score": 0.972, "support": 214.0},
            "positive": {"precision": 0.904, "recall": 0.955, "f1-score": 0.929, "support": 89.0},
            "negative": {"precision": 0.949, "recall": 0.881, "f1-score": 0.914, "support": 42.0},
        },
    }


def _subgroup_records():
    terciles = [
        ("small", 0.121, 0.090, 0.151, 4455),
        ("mid", 0.101, 0.074, 0.127, 4442),
        ("large", 0.081, 0.052, 0.110, 4428),
    ]
    recs = [{"return_col": "abn_return_1d", "kind": "tercile", "tercile": t,
             "rho": rho, "ci_low": lo, "ci_high": hi, "n": n}
            for t, rho, lo, hi, n in terciles]
    recs += [{"return_col": "abn_return_1d", "kind": "pair", "pair": p,
              "ci_overlap": True, "differ": False}
             for p in ("small-mid", "small-large", "mid-large")]
    # A second window that should be ignored by primary-window helpers.
    recs.append({"return_col": "return_5d", "kind": "tercile", "tercile": "small",
                 "rho": 0.5, "ci_low": 0.4, "ci_high": 0.6, "n": 10})
    return recs


# --------------------------------------------------------------------------- #
# Data-shaping helpers
# --------------------------------------------------------------------------- #
def test_class_weights_matches_documented_values():
    w = viz.class_weights({"neutral": 1713, "positive": 709, "negative": 336})
    assert w["neutral"] == pytest.approx(0.62, abs=0.01)
    assert w["positive"] == pytest.approx(0.97, abs=0.01)
    assert w["negative"] == pytest.approx(1.41, abs=0.01)
    assert np.mean(list(w.values())) == pytest.approx(1.0)


def test_primary_ladder_pulls_the_three_rows():
    ladder = viz.primary_ladder(_correlation_records())
    assert ladder["finetuned"]["estimate"] == pytest.approx(0.104)
    assert ladder["baseline"]["estimate"] == pytest.approx(-0.002)
    assert ladder["difference"]["ci_low"] == pytest.approx(0.082)
    assert ladder["finetuned"]["n"] == 13429


def test_robustness_rows_are_ordered_and_windowed():
    rows = viz.robustness_rows(_correlation_records())
    assert [r["return_col"] for r in rows] == list(viz.ROBUSTNESS_ORDER)
    assert rows[2]["estimate"] == pytest.approx(0.083)  # return_5d


def test_confusion_and_per_class_class_order():
    matrix, per_class = viz.confusion_and_per_class(_metrics())
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == 207 and matrix[2, 2] == 37
    assert list(per_class) == list(viz.CLASS_ORDER)
    assert per_class["negative"]["f1"] == pytest.approx(0.914)
    assert per_class["neutral"]["support"] == 214


def test_tercile_points_primary_window_only():
    pts = viz.tercile_points(_subgroup_records())
    assert [p["tercile"] for p in pts] == ["small", "mid", "large"]
    assert pts[0]["estimate"] == pytest.approx(0.121)
    assert all(p["estimate"] < 0.2 for p in pts)  # the return_5d row is excluded


def test_pair_overlaps_all_overlap():
    pairs = viz.pair_overlaps(_subgroup_records())
    assert len(pairs) == 3
    assert all(p["ci_overlap"] and not p["differ"] for p in pairs)


# --------------------------------------------------------------------------- #
# Smoke tests: each JSON-based plot writes a PNG (Agg backend, CI-safe)
# --------------------------------------------------------------------------- #
def test_plot_baseline_ladder_writes_png(tmp_path):
    out = viz.plot_baseline_ladder(_correlation_records(), tmp_path / "ladder.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_classification_quality_writes_png(tmp_path):
    out = viz.plot_classification_quality(_metrics(), out_path=tmp_path / "cls.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_market_cap_subgroup_writes_png(tmp_path):
    out = viz.plot_market_cap_subgroup(_subgroup_records(), tmp_path / "sub.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_sentiment_by_return_sign_writes_png(tmp_path):
    calls = pd.DataFrame({
        "sentiment_score": [0.4, 0.2, -0.1, 0.3, -0.3, 0.1],
        "abn_return_1d": [0.02, -0.01, -0.02, 0.03, -0.04, 0.01],
    })
    out = viz.plot_sentiment_by_return_sign(calls, tmp_path / "sign.png")
    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Figure-4 frame builder — reads the real gitignored parquet (excluded in CI)
# --------------------------------------------------------------------------- #
@pytest.mark.data
def test_build_return_sign_frame_from_real_data():
    calls = viz.build_return_sign_frame()
    assert calls is not None, "scored parquet / index missing — run the pipeline first"
    assert {"sentiment_score", viz.PRIMARY_RETURN_COL} <= set(calls.columns)
    assert calls[["sentiment_score", viz.PRIMARY_RETURN_COL]].notna().all().all()
    assert len(calls) > 0

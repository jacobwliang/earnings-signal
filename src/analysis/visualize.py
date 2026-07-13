"""ES-14 visualizations: honest figures for the analysis layer.

Four figures, each rendered from a committed results file so the JSON-based ones
reproduce deterministically in CI:

1. ``es14_baseline_ladder.png`` — fine-tuned sentiment clears the chance floor and
   the signal holds across return windows (from ``correlation_results.json``).
2. ``es14_classification_quality.png`` — PhraseBank confusion matrix + per-class
   precision/recall/F1, annotated with support and class weight, framed so the
   small negative *train* class is not misread as a weak negative *test* score
   (from ``r2_finetune_metrics_full_finetune.json``).
3. ``es14_market_cap_subgroup.png`` — per-tercile rho with CIs, with the explicit
   note that every pairwise CI overlaps (from ``subgroup_market_cap_results.json``).
4. ``es14_sentiment_by_return_sign.png`` — a sanity check: call-level sentiment
   split by the sign of the realized 1-day abnormal return. Needs the gitignored
   scored parquet + index, so it runs locally only and is skipped when absent.

Guiding principle: every correlation figure renders its 95% CI and ``n`` on the
plot. Note on windows: the pipeline only computes 1d and 5d returns (no 3d), so
the issue's "3d" panel is intentionally absent.

Run from the repo root with the project venv:

    .venv/bin/python -m src.analysis.visualize
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: choose backend before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.correlate_returns import (  # noqa: E402
    FINETUNED_SCORES_PATH,
    INDEX_PATH,
    PRIMARY_RETURN_COL,
    ROOT,
    add_abnormal_returns,
    aggregate_to_call,
)
from src.data.compute_returns import compute_market_returns  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
RESULTS_DIR = ROOT / "results"
CORRELATION_JSON = RESULTS_DIR / "correlation_results.json"
METRICS_JSON = RESULTS_DIR / "r2_finetune_metrics_full_finetune.json"
SUBGROUP_JSON = RESULTS_DIR / "subgroup_market_cap_results.json"

LADDER_PNG = RESULTS_DIR / "es14_baseline_ladder.png"
CLASSIFICATION_PNG = RESULTS_DIR / "es14_classification_quality.png"
SUBGROUP_PNG = RESULTS_DIR / "es14_market_cap_subgroup.png"
RETURN_SIGN_PNG = RESULTS_DIR / "es14_sentiment_by_return_sign.png"

# --------------------------------------------------------------------------- #
# Shared style (first styled figure set in the repo). Light-surface only —
# these are PNGs, so we use the validated light-mode categorical values from the
# dataviz palette. Adjacent aqua/yellow are sub-3:1 on white, so every figure
# that uses them also carries direct value labels (the relief rule).
# --------------------------------------------------------------------------- #
INK = "#0b0b0b"       # primary text
SECONDARY = "#52514e"  # secondary text
MUTED = "#898781"      # axis/labels + the greyed baseline floor
GRID = "#e1e0d9"       # hairline gridlines

BLUE = "#2a78d6"    # categorical slot 1 — the fine-tuned signal / "up" pole
AQUA = "#1baf7a"    # categorical slot 2
YELLOW = "#eda100"  # categorical slot 3
RED = "#e34948"     # categorical slot 6 — the "down" pole

# Sentiment classes are label ids {0: neutral, 1: positive, 2: negative}.
CLASS_ORDER = ("neutral", "positive", "negative")
# BH-corrected robustness windows, in the order they read on the x-axis.
ROBUSTNESS_ORDER = ("return_1d", "abn_return_1d", "return_5d", "abn_return_5d")
# Documented Financial PhraseBank train-split counts (sentences_75agree, 80/10/10,
# seed 42), label-id order. Persisted split is gitignored, so we carry the counts
# here to keep the class-weight panel CI-safe and deterministic.
TRAIN_COUNTS = {"neutral": 1713, "positive": 709, "negative": 336}

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": SECONDARY,
    "axes.titlecolor": INK,
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
    "text.color": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
})


def _save(fig, out_path: Path) -> Path:
    """Save a figure as PNG (dpi=120), close it, and print — the repo convention."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Data shaping (pure — unit-tested without touching pixels or disk)
# --------------------------------------------------------------------------- #
def class_weights(counts: dict[str, int]) -> dict[str, float]:
    """sqrt-inverse-frequency class weights, normalized to mean 1.

    Same formula as ``compute_class_weights`` in the fine-tune module
    (``w_c = sqrt(N / N_c)``, then divide by the mean); replicated here so the
    plotting script needn't import torch/transformers. Keyed by class name.
    """
    names = list(counts)
    c = np.array([counts[n] for n in names], dtype=float)
    w = np.sqrt(c.sum() / c)
    w = w / w.mean()
    return dict(zip(names, w.tolist()))


def primary_ladder(records: list[dict]) -> dict:
    """Pull the primary-endpoint ladder from ``correlation_results`` records.

    Returns the finetuned/baseline point + CI at ``abn_return_1d`` and the paired
    fine-tuned − baseline difference row, plus the shared ``n``.
    """
    def _one(analysis: str) -> dict:
        row = next(r for r in records if r["analysis"] == analysis)
        return {"estimate": row["estimate"], "ci_low": row["ci_low"],
                "ci_high": row["ci_high"], "n": row["n"]}

    return {"finetuned": _one("primary"), "baseline": _one("baseline_floor"),
            "difference": _one("difference")}


def robustness_rows(records: list[dict]) -> list[dict]:
    """Fine-tuned rho + CI per return window, ordered as ``ROBUSTNESS_ORDER``."""
    by_col = {r["return_col"]: r for r in records if r["analysis"] == "robustness"}
    return [{"return_col": col, "estimate": by_col[col]["estimate"],
             "ci_low": by_col[col]["ci_low"], "ci_high": by_col[col]["ci_high"],
             "n": by_col[col]["n"]}
            for col in ROBUSTNESS_ORDER if col in by_col]


def confusion_and_per_class(metrics: dict) -> tuple[np.ndarray, dict]:
    """Extract the confusion matrix and a per-class metric dict in class order.

    Returns ``(matrix, per_class)`` where ``matrix`` is rows=true/cols=pred in
    ``CLASS_ORDER`` and ``per_class[name]`` has precision/recall/f1/support.
    """
    matrix = np.array(metrics["confusion_matrix"], dtype=int)
    pc = metrics["per_class"]
    per_class = {
        name: {"precision": pc[name]["precision"], "recall": pc[name]["recall"],
               "f1": pc[name]["f1-score"], "support": int(pc[name]["support"])}
        for name in CLASS_ORDER
    }
    return matrix, per_class


def tercile_points(records: list[dict], return_col: str = PRIMARY_RETURN_COL) -> list[dict]:
    """Per-tercile rho + CI + n at ``return_col``, ordered small → mid → large."""
    order = ("small", "mid", "large")
    by_t = {r["tercile"]: r for r in records
            if r.get("kind") == "tercile" and r["return_col"] == return_col}
    return [{"tercile": t, "estimate": by_t[t]["rho"], "ci_low": by_t[t]["ci_low"],
             "ci_high": by_t[t]["ci_high"], "n": int(by_t[t]["n"])}
            for t in order if t in by_t]


def pair_overlaps(records: list[dict], return_col: str = PRIMARY_RETURN_COL) -> list[dict]:
    """Pairwise CI-overlap / differ flags at ``return_col``."""
    return [{"pair": r["pair"], "ci_overlap": bool(r["ci_overlap"]),
             "differ": bool(r["differ"])}
            for r in records
            if r.get("kind") == "pair" and r["return_col"] == return_col]


# --------------------------------------------------------------------------- #
# Figure 1 — baseline ladder
# --------------------------------------------------------------------------- #
def plot_baseline_ladder(records: list[dict], out_path: Path = LADDER_PNG) -> Path:
    """Proof-over-chance: floor vs fine-tuned at the primary window + robustness."""
    ladder = primary_ladder(records)
    rob = robustness_rows(records)

    fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: baseline floor (greyed) vs fine-tuned at abn_return_1d.
    ax_p.axhline(0, color=MUTED, linewidth=1, linestyle="--", zorder=0)
    points = [("baseline\nfloor", ladder["baseline"], MUTED),
              ("fine-tuned", ladder["finetuned"], BLUE)]
    for i, (label, d, color) in enumerate(points):
        err = [[d["estimate"] - d["ci_low"]], [d["ci_high"] - d["estimate"]]]
        ax_p.errorbar(i, d["estimate"], yerr=err, fmt="o", color=color,
                      markersize=9, capsize=5, linewidth=2, zorder=3)
        ax_p.annotate(f"{d['estimate']:+.3f}", (i, d["estimate"]),
                      textcoords="offset points", xytext=(12, 0), va="center",
                      color=color, fontweight="bold")
    ax_p.set_xticks([0, 1])
    ax_p.set_xticklabels([p[0] for p in points])
    ax_p.set_xlim(-0.5, 1.6)
    ax_p.set_ylabel("Spearman ρ  (sentiment vs abn. 1-day return)")
    ax_p.set_title("Fine-tuned sentiment clears the chance floor")
    ax_p.grid(axis="y", zorder=0)

    diff = ladder["difference"]
    ax_p.text(0.5, 0.5,
              f"Δρ = {diff['estimate']:+.3f}\n95% CI "
              f"[{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}]\nn = {diff['n']:,}",
              transform=ax_p.transAxes, ha="center", va="center",
              fontsize=9, color=SECONDARY,
              bbox=dict(boxstyle="round", fc="white", ec=GRID))

    # Right: fine-tuned rho with CI across all four windows.
    xs = np.arange(len(rob))
    ests = [r["estimate"] for r in rob]
    lo = [r["estimate"] - r["ci_low"] for r in rob]
    hi = [r["ci_high"] - r["estimate"] for r in rob]
    ax_r.axhline(0, color=MUTED, linewidth=1, linestyle="--", zorder=0)
    ax_r.errorbar(xs, ests, yerr=[lo, hi], fmt="o", color=BLUE, markersize=9,
                  capsize=5, linewidth=2, zorder=3)
    ax_r.set_xticks(xs)
    ax_r.set_xticklabels([r["return_col"] for r in rob], rotation=20, ha="right")
    ax_r.set_ylim(bottom=min(0, min(ests) - 0.02))
    ax_r.set_ylabel("Spearman ρ")
    ax_r.set_title("Fine-tuned signal holds across return windows")
    ax_r.grid(axis="y", zorder=0)
    ax_r.text(0.5, 0.02,
              "baseline floor computed at abn_return_1d only (not fabricated elsewhere)",
              transform=ax_r.transAxes, ha="center", va="bottom",
              fontsize=8, color=MUTED, style="italic")

    fig.suptitle("ES-14 · Fine-tuned sentiment vs chance floor", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 2 — classification quality
# --------------------------------------------------------------------------- #
def plot_classification_quality(metrics: dict, train_counts: dict[str, int] = TRAIN_COUNTS,
                                out_path: Path = CLASSIFICATION_PNG) -> Path:
    """Confusion matrix + per-class precision/recall/F1 with support & weight."""
    matrix, per_class = confusion_and_per_class(metrics)
    weights = class_weights(train_counts)

    fig, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: confusion matrix as a single-hue (blue) sequential heatmap.
    im = ax_cm.imshow(matrix, cmap="Blues")
    ax_cm.set_xticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER)
    ax_cm.set_yticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER)
    ax_cm.set_xlabel("predicted")
    ax_cm.set_ylabel("true")
    ax_cm.set_title("Confusion matrix (PhraseBank test)")
    thresh = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax_cm.text(j, i, f"{matrix[i, j]:d}", ha="center", va="center",
                       color="white" if matrix[i, j] > thresh else INK,
                       fontweight="bold")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    # Right: grouped precision/recall/F1 bars per class.
    metric_names = ("precision", "recall", "f1")
    metric_colors = (BLUE, AQUA, YELLOW)
    x = np.arange(len(CLASS_ORDER))
    width = 0.26
    for k, (m, color) in enumerate(zip(metric_names, metric_colors)):
        vals = [per_class[c][m] for c in CLASS_ORDER]
        bars = ax_bar.bar(x + (k - 1) * width, vals, width, label=m, color=color)
        ax_bar.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color=SECONDARY)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [f"{c}\nn={per_class[c]['support']}  w={weights[c]:.2f}" for c in CLASS_ORDER])
    ax_bar.set_ylim(0, 1.12)
    ax_bar.set_ylabel("score")
    ax_bar.set_title("Per-class metrics (support n, train weight w)")
    ax_bar.legend(loc="lower right", frameon=False)
    ax_bar.grid(axis="y", zorder=0)

    neg = per_class["negative"]
    fig.text(0.5, -0.04,
             f"Negative class scores F1 {neg['f1']:.3f} / recall {neg['recall']:.3f} "
             f"in-domain — strong. The weak spot is the small train class "
             f"(n={train_counts['negative']}) and domain shift onto unlabeled earnings "
             f"calls, which this matrix cannot show.",
             ha="center", va="top", fontsize=8.5, color=SECONDARY, wrap=True)

    fig.suptitle("ES-14 · Fine-tuned classification quality", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 3 — market-cap subgroup
# --------------------------------------------------------------------------- #
def plot_market_cap_subgroup(records: list[dict], out_path: Path = SUBGROUP_PNG) -> Path:
    """Per-tercile rho with CIs and the all-pairs-overlap guardrail note."""
    points = tercile_points(records)
    overlaps = pair_overlaps(records)
    all_overlap = bool(overlaps) and all(p["ci_overlap"] and not p["differ"]
                                         for p in overlaps)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = np.arange(len(points))
    ests = [p["estimate"] for p in points]
    lo = [p["estimate"] - p["ci_low"] for p in points]
    hi = [p["ci_high"] - p["estimate"] for p in points]

    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--", zorder=0)
    ax.errorbar(xs, ests, yerr=[lo, hi], fmt="o", color=BLUE, markersize=10,
                capsize=6, linewidth=2, zorder=3)
    for xi, p in zip(xs, points):
        ax.annotate(f"{p['estimate']:.3f}\nn={p['n']:,}",
                    (xi, p["estimate"]), textcoords="offset points",
                    xytext=(14, 0), va="center", color=SECONDARY, fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{p['tercile']} cap" for p in points])
    ax.set_xlim(-0.5, len(points) - 0.2)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Spearman ρ  (sentiment vs abn. 1-day return)")
    ax.set_title("Sentiment–return correlation by market-cap tercile")
    ax.grid(axis="y", zorder=0)

    note = ("Monotone point estimates, but all pairwise 95% CIs overlap — "
            "the size ordering is not statistically distinguishable."
            if all_overlap else
            "Pairwise CI-overlap flags vary — see subgroup results.")
    ax.text(0.5, -0.13, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color=SECONDARY, style="italic")

    fig.suptitle("ES-14 · Market-cap subgroup analysis", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 4 — sentiment by realized return sign (local-only)
# --------------------------------------------------------------------------- #
def build_return_sign_frame(
    finetuned_path: Path = FINETUNED_SCORES_PATH,
    index_path: Path = INDEX_PATH,
) -> pd.DataFrame | None:
    """Call-level (sentiment_score, abn_return_1d), or None if inputs are absent.

    Rebuilds the frame exactly as the correlation analysis does — abnormal returns
    then aggregate-to-call — so the split mirrors the primary endpoint. The scored
    parquet and index are gitignored, so this returns None (a clean skip) when they
    are missing rather than raising.
    """
    if not finetuned_path.exists() or not index_path.exists():
        return None
    scores = pd.read_parquet(finetuned_path)
    index_close = pd.read_parquet(index_path)
    market_returns = compute_market_returns(index_close, scores["return_start_date"].unique())
    calls = aggregate_to_call(add_abnormal_returns(scores, market_returns))
    return calls.dropna(subset=["sentiment_score", PRIMARY_RETURN_COL])


def plot_sentiment_by_return_sign(calls: pd.DataFrame,
                                  out_path: Path = RETURN_SIGN_PNG) -> Path:
    """Overlaid sentiment-score densities for up-days vs down-days (by abn 1d sign)."""
    up = calls[calls[PRIMARY_RETURN_COL] > 0]["sentiment_score"].to_numpy()
    down = calls[calls[PRIMARY_RETURN_COL] < 0]["sentiment_score"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(up.min(), down.min()), max(up.max(), down.max()), 40)
    for data, color, label in ((up, BLUE, "up days"), (down, RED, "down days")):
        ax.hist(data, bins=bins, density=True, alpha=0.5, color=color,
                label=f"{label}  (n={len(data):,})")
        ax.axvline(np.median(data), color=color, linewidth=2, linestyle="--")

    ax.set_xlabel("call-level sentiment score")
    ax.set_ylabel("density")
    ax.set_title("Sentiment by realized next-day return sign")
    ax.legend(frameon=False)
    ax.grid(axis="y", zorder=0)
    ax.text(0.5, -0.16,
            "Earnings calls are unlabeled; this splits by realized outcome, so it "
            "visualizes the same correlation — not independent validation.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            color=SECONDARY, style="italic")

    fig.suptitle("ES-14 · Sentiment vs realized return sign (sanity check)",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _load_json(path: Path):
    return json.loads(path.read_text())


def main(argv=None) -> None:
    """Render figures 1–3 from committed JSON; attempt figure 4 from parquet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correlation-json", type=Path, default=CORRELATION_JSON)
    parser.add_argument("--metrics-json", type=Path, default=METRICS_JSON)
    parser.add_argument("--subgroup-json", type=Path, default=SUBGROUP_JSON)
    parser.add_argument("--finetuned", type=Path, default=FINETUNED_SCORES_PATH)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    plot_baseline_ladder(_load_json(args.correlation_json),
                         args.outdir / LADDER_PNG.name)
    plot_classification_quality(_load_json(args.metrics_json),
                                out_path=args.outdir / CLASSIFICATION_PNG.name)
    plot_market_cap_subgroup(_load_json(args.subgroup_json),
                             args.outdir / SUBGROUP_PNG.name)

    calls = build_return_sign_frame(args.finetuned, args.index)
    if calls is None:
        print(f"Skipping {RETURN_SIGN_PNG.name}: scored parquet / index not found "
              "(gitignored — run the pipeline locally to render it).")
    else:
        plot_sentiment_by_return_sign(calls, args.outdir / RETURN_SIGN_PNG.name)


if __name__ == "__main__":
    main()

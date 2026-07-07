"""Fine-tune FinBERT on Financial PhraseBank (ES-09/10).

Two-tier "baseline ladder" fine-tune of ``yiyanghkust/finbert-pretrain`` on the
75%-agreement PhraseBank splits built by ``src/data/load_phrasebank.py``:

* **Tier 1 - linear probe:** freeze the BERT backbone (and pooler), train only
  the classification head with a high learning rate. A cheap floor that measures
  how much signal the frozen [CLS] representation already carries.
* **Tier 2 - full fine-tune:** all layers trainable, small learning rate. The
  model actually shipped downstream for transcript scoring.

Both tiers share one weighted ``Trainer`` loop. The class imbalance (train split
is ~62% neutral / 26% positive / 12% negative) is handled with
sqrt-inverse-frequency class weights in the cross-entropy loss; model selection
and reporting use macro-F1 so the rare negative class counts equally.

Label ids follow the project convention in ``src/models/infer_baseline.py``:
``LABELS = ("neutral", "positive", "negative")`` -> 0=neutral, 1=positive,
2=negative. The index IS the label id the head learns, so the order is
load-bearing and must stay in sync with the baseline pipeline.

Runs flat: drop this file and the three ``phrasebank_75agree_*.parquet`` splits
in one directory (e.g. a Colab working dir) and run ``python
finetune_phrasebank.py`` — all outputs (models, metrics JSON, report) are written
back into that same directory to move off manually. From the repo you can also
run ``python -m src.models.finetune_phrasebank``. Add ``--smoke`` for a 1-epoch
CPU dry run on a small slice that exercises the whole loop.
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")               # headless: save PNGs, never open a window (Colab/CI-safe)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Paths resolve next to this script so it runs flat on Colab: drop the three
# phrasebank_75agree_*.parquet splits beside this file, run, and the trained
# models, metrics JSON, and report are written back into the same folder to move
# off Colab manually.
HERE = Path(__file__).resolve().parent

SPLIT_PATHS = {
    split: HERE / f"phrasebank_75agree_{split}.parquet"
    for split in ("train", "valid", "test")
}

MODEL_NAME = "yiyanghkust/finbert-pretrain"
NUM_LABELS = 3
MAX_LENGTH = 96                    # lossless: longest PhraseBank sentence is ~81 words

# Project label convention (matches infer_baseline.LABELS). Order is load-bearing:
# the index IS the label id the fine-tuned head learns.
LABELS = ("neutral", "positive", "negative")   # 0=neutral, 1=positive, 2=negative
ID2LABEL = {i: name for i, name in enumerate(LABELS)}
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

RANDOM_SEED = 42
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
WEIGHT_DECAY = 0.01                 # default; the full_finetune grid overrides it
WARMUP_RATIO = 0.1
REPORT_TO = "mlflow"               # ES-11 experiment tracking (mlflow is pinned)
NEGATIVE_ID = LABEL2ID["negative"]  # rare-class guardrail: watch its F1, not just macro

# Per-tier fixed hyperparameters. The probe gets a high LR, a taller epoch ceiling
# (low-capacity head, early stopping ends it), and coarse epoch-level eval — it is a
# deliberate low-capacity floor and is left unchanged. The full fine-tune uses the
# standard BERT recipe with step-level eval so the regularization grid's train/eval
# loss gap is measurable. ``early_stop_patience`` counts EVAL EVENTS, not epochs: at
# ~7 evals/epoch (eval_steps 25 on ~173 steps/epoch), patience 12 is ~1.7 epochs of
# no macro-F1 improvement.
TIERS = {
    "linear_probe": {
        "freeze_backbone": True, "learning_rate": 2e-3, "epochs": 20,
        "eval_steps": None, "early_stop_patience": 2,
    },
    "full_finetune": {
        "freeze_backbone": False, "learning_rate": 2e-5, "epochs": 4,
        "eval_steps": 25, "early_stop_patience": 6,
    },
}

# ES-09/10 overfitting RFC: a regularization grid applied to the full_finetune tier
# only (LR held at the working 2e-5). Each entry overrides regularization strength;
# runs are compared on valid macro-F1, the train/eval loss gap, and negative-class
# F1. See docs/rfc-reduce-overfitting.md.
FULL_FINETUNE_GRID = [
    {"label_smoothing": ls, "weight_decay": wd, "hidden_dropout_prob": do}
    for ls in (0.0, 0.1)
    for wd in (0.01, 0.1)
    for do in (0.1, 0.3)
]


def get_device() -> torch.device:
    """Return the best available device: CUDA (Colab GPU), then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TokenizedDataset(torch.utils.data.Dataset):
    """A PhraseBank split tokenized once, padded per-batch by the collator.

    Encodings are computed up front with ``truncation`` to :data:`MAX_LENGTH` but
    **no** padding — ``DataCollatorWithPadding`` pads each batch to its own longest
    sequence, which is cheaper than padding every row to 96. ``__getitem__``
    returns the variable-length ``input_ids``/``attention_mask`` plus the int
    ``labels`` the collator and ``Trainer`` expect.
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        self.encodings = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: values[idx] for key, values in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_splits(tokenizer, smoke: bool = False) -> dict[str, TokenizedDataset]:
    """Read the three parquet splits and tokenize each into a dataset.

    With ``smoke=True`` each split is truncated to a handful of rows so the loop
    can be exercised on CPU in seconds. Class-stratification is not preserved in
    smoke mode — it exists only to prove the wiring, not to learn anything.
    """
    caps = {"train": 64, "valid": 32, "test": 32}
    datasets: dict[str, TokenizedDataset] = {}
    for split, path in SPLIT_PATHS.items():
        frame = pd.read_parquet(path, columns=["text", "label"])
        if smoke:
            frame = frame.head(caps[split])
        datasets[split] = TokenizedDataset(
            frame["text"].tolist(), frame["label"].tolist(), tokenizer
        )
    return datasets


def compute_class_weights(labels) -> np.ndarray:
    """Return sqrt-inverse-frequency class weights, normalized to mean 1.

    ``w_c = sqrt(N_total / N_c)`` upweights rare classes without the aggression
    of plain inverse frequency; dividing by the mean keeps the loss scale
    comparable to the unweighted case. Indexed by label id (:data:`LABELS` order).
    """
    counts = np.bincount(np.asarray(labels), minlength=NUM_LABELS).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"every class must be present to weight; got counts {counts}")
    weights = np.sqrt(counts.sum() / counts)
    return weights / weights.mean()


def compute_metrics(eval_pred) -> dict:
    """Macro-F1 (model-selection metric) and accuracy from logits + labels."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
    }


class WeightedTrainer(Trainer):
    """``Trainer`` with a class-weighted, optionally label-smoothed cross-entropy.

    The weight tensor is moved onto the logits' device on each step so the same
    trainer works on CPU, MPS, or CUDA without pre-placing the weights.

    Because this overrides ``compute_loss`` and calls ``cross_entropy`` directly,
    ``TrainingArguments.label_smoothing_factor`` is bypassed — smoothing MUST be
    passed here via ``label_smoothing`` or it silently does nothing.
    """

    def __init__(
        self,
        *args,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights
        self._label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = (
            self._class_weights.to(outputs.logits.device)
            if self._class_weights is not None
            else None
        )
        loss = nn.functional.cross_entropy(
            outputs.logits, labels, weight=weight, label_smoothing=self._label_smoothing
        )
        return (loss, outputs) if return_outputs else loss


def build_model(
    hidden_dropout_prob: float | None = None,
) -> BertForSequenceClassification:
    """Load finbert-pretrain with a fresh 3-class head and the label mapping.

    ``hidden_dropout_prob`` (when set) overrides the model's dropout on both the
    encoder and — since BERT's ``classifier_dropout`` falls back to it — the head,
    the anti-overfit lever swept by :data:`FULL_FINETUNE_GRID`. ``None`` keeps the
    pretrained default (0.1).
    """
    extra = {} if hidden_dropout_prob is None else {"hidden_dropout_prob": hidden_dropout_prob}
    return BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID, **extra
    )


def freeze_backbone(model: BertForSequenceClassification) -> BertForSequenceClassification:
    """Freeze every ``bert.*`` parameter, leaving only ``classifier`` trainable.

    The pooler dense lives under ``bert`` and stays frozen too, so the probe fits
    a linear layer on the tanh-pooled [CLS] representation.
    """
    for name, param in model.named_parameters():
        if name.startswith("bert."):
            param.requires_grad = False
    return model


def build_training_args(
    output_dir: Path,
    learning_rate: float,
    epochs: int,
    *,
    weight_decay: float,
    eval_steps: int | None,
    smoke: bool,
) -> TrainingArguments:
    """Shared TrainingArguments; ``smoke`` forces CPU and disables reporting/fp16.

    When ``eval_steps`` is set (and not a smoke run) eval/save/logging switch to
    step-based cadence so a run yields ~7 curve points per epoch instead of one —
    enough resolution to compare the grid's train/eval loss gaps. ``smoke`` and the
    probe (``eval_steps=None``) keep the original epoch cadence.
    """
    step_based = eval_steps is not None and not smoke
    strategy = "steps" if step_based else "epoch"
    args = dict(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=1 if smoke else epochs,
        weight_decay=weight_decay,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy=strategy,
        save_strategy=strategy,
        logging_strategy=strategy,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=RANDOM_SEED,
        report_to="none" if smoke else REPORT_TO,
        fp16=torch.cuda.is_available() and not smoke,
        use_cpu=smoke,
    )
    if step_based:
        args.update(eval_steps=eval_steps, save_steps=eval_steps, logging_steps=eval_steps)
    return TrainingArguments(**args)


def build_run(tier_name: str, reg: dict, output_dir: Path) -> dict:
    """Merge a tier's fixed settings with one regularization config + output dir."""
    base = TIERS[tier_name]
    return {
        "tier": tier_name,
        "freeze_backbone": base["freeze_backbone"],
        "learning_rate": base["learning_rate"],
        "epochs": base["epochs"],
        "eval_steps": base["eval_steps"],
        "early_stop_patience": base["early_stop_patience"],
        "label_smoothing": reg["label_smoothing"],
        "weight_decay": reg["weight_decay"],
        "hidden_dropout_prob": reg["hidden_dropout_prob"],
        "output_dir": output_dir,
    }


def train_tier(
    run: dict,
    datasets: dict[str, TokenizedDataset],
    class_weights: torch.Tensor,
    tokenizer,
    *,
    smoke: bool,
) -> Trainer:
    """Build a fresh model and train one run to its output dir.

    ``run`` comes from :func:`build_run`. The seed is reset here so identical
    configs reproduce (the selection pass and the winner retrain must match), and
    the output dir is cleared first so stale checkpoints from a prior grid run
    can't be picked up.
    """
    set_seed(RANDOM_SEED)
    output_dir = run["output_dir"]
    if output_dir.exists():
        shutil.rmtree(output_dir)

    model = build_model(hidden_dropout_prob=run["hidden_dropout_prob"])
    if run["freeze_backbone"]:
        freeze_backbone(model)

    args = build_training_args(
        output_dir,
        run["learning_rate"],
        run["epochs"],
        weight_decay=run["weight_decay"],
        eval_steps=run["eval_steps"],
        smoke=smoke,
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        label_smoothing=run["label_smoothing"],
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=run["early_stop_patience"])
        ],
    )
    print(
        f"\n=== Training {run['tier']} "
        f"(label_smoothing={run['label_smoothing']}, weight_decay={run['weight_decay']}, "
        f"dropout={run['hidden_dropout_prob']}) ==="
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return trainer


def evaluate_tier(trainer: Trainer, test_dataset: TokenizedDataset) -> dict:
    """Score the (best) model on the test split; return a metrics dict.

    Reports macro-F1, accuracy, per-class precision/recall/F1, and the confusion
    matrix. The negative class has only ~42 test examples, so its recall carries a
    wide CI (~+/-14pp) — read the per-class numbers with that in mind.
    """
    prediction = trainer.predict(test_dataset)
    y_true = prediction.label_ids
    y_pred = np.argmax(prediction.predictions, axis=-1)
    report = classification_report(
        y_true, y_pred, target_names=list(LABELS), output_dict=True, zero_division=0
    )
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "per_class": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(NUM_LABELS))
        ).tolist(),
    }


def valid_selection_metrics(trainer: Trainer, valid_dataset: TokenizedDataset) -> tuple[float, float]:
    """Return (macro-F1, negative-class F1) on the valid split for the best model.

    ``load_best_model_at_end`` means ``trainer`` already holds the best checkpoint,
    so this scores the model that would ship. Grid selection is on valid only — the
    test split is never consulted for choosing a config.
    """
    prediction = trainer.predict(valid_dataset)
    y_true = prediction.label_ids
    y_pred = np.argmax(prediction.predictions, axis=-1)
    macro = f1_score(y_true, y_pred, average="macro")
    per_class = f1_score(y_true, y_pred, labels=list(range(NUM_LABELS)), average=None)
    return float(macro), float(per_class[NEGATIVE_ID])


def loss_gap_at_best(trainer: Trainer) -> dict:
    """Train/eval loss gap at the selected checkpoint — the overfitting diagnostic.

    Reads ``trainer.state.log_history`` for the eval entry at the best step and the
    training-loss entry logged at or just before it; the gap ``eval_loss -
    train_loss`` is how much worse held-out loss is than train loss where the model
    was selected. A smaller gap is the goal of the regularization grid. Returns
    zeros if the history is too sparse to pair up (e.g. a 1-epoch smoke run).
    """
    log = trainer.state.log_history
    best_step = trainer.state.best_global_step
    evals = [e for e in log if "eval_loss" in e]
    trains = [e for e in log if "loss" in e and "eval_loss" not in e]
    if not evals or not trains:
        return {"train_loss": 0.0, "eval_loss": 0.0, "loss_gap": 0.0}
    best_eval = next((e for e in evals if e.get("step") == best_step), evals[-1])
    prior = [e for e in trains if e.get("step", 0) <= best_eval.get("step", 0)]
    best_train = prior[-1] if prior else trains[0]
    return {
        "train_loss": float(best_train["loss"]),
        "eval_loss": float(best_eval["eval_loss"]),
        "loss_gap": float(best_eval["eval_loss"] - best_train["loss"]),
    }


def plot_learning_curves(trainer: Trainer, out_path: Path, title: str) -> Path | None:
    """Save a run's learning curves (train/eval loss + eval metrics vs step) as PNG.

    Reads ``trainer.state.log_history`` — dense for the step-eval full_finetune runs
    (~7 points/epoch), coarse for the epoch-eval probe. Left panel is the
    overfitting view (train vs eval loss); right panel is eval accuracy + macro-F1;
    a dashed line marks the selected checkpoint. Returns the path, or ``None`` if
    the history has no logged points (nothing to draw).
    """
    log = trainer.state.log_history
    train = [(e["step"], e["loss"]) for e in log if "loss" in e and "eval_loss" not in e]
    evals = [
        (e["step"], e["eval_loss"], e.get("eval_f1"), e.get("eval_accuracy"))
        for e in log
        if "eval_loss" in e
    ]
    if not train or not evals:
        return None

    ts, tl = zip(*train)
    es, el, ef, ea = zip(*evals)
    best = trainer.state.best_global_step

    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(ts, tl, marker="o", label="train loss")
    ax_loss.plot(es, el, marker="o", label="eval loss")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss (train vs eval)")

    ax_metric.plot(es, ea, marker="o", label="eval accuracy")
    ax_metric.plot(es, ef, marker="o", label="eval macro-F1")
    ax_metric.set_xlabel("step")
    ax_metric.set_ylabel("score")
    ax_metric.set_title("Eval metrics")

    for ax in (ax_loss, ax_metric):
        if best is not None:
            ax.axvline(best, color="gray", linestyle="--", linewidth=1, label="selected ckpt")
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def write_report(
    metrics: dict[str, dict], out_base: Path, grid_rows: list[dict] | None = None
) -> None:
    """Write per-tier metrics JSON and a combined markdown analysis report."""
    for name, tier_metrics in metrics.items():
        path = out_base / f"finetune_metrics_{name}.json"
        path.write_text(json.dumps(tier_metrics, indent=2))
        print(f"Wrote {path}")

    lines = [
        "# Fine-tuned FinBERT on Financial PhraseBank (ES-09/10)",
        "",
        "Test-split results for the two-tier baseline ladder. Label ids: "
        "0=neutral, 1=positive, 2=negative. The negative class has ~42 test "
        "examples, so its recall carries a wide CI (~+/-14pp).",
        "",
        "| Tier | Macro-F1 | Accuracy | Neg recall | Neg F1 |",
        "|---|---|---|---|---|",
    ]
    for name, m in metrics.items():
        neg = m["per_class"]["negative"]
        lines.append(
            f"| {name} | {m['macro_f1']:.3f} | {m['accuracy']:.3f} | "
            f"{neg['recall']:.3f} | {neg['f1-score']:.3f} |"
        )
    lines += [
        "",
        "Compare against the chance-floor baseline in "
        "[baseline_analysis.md](baseline_analysis.md) (macro-F1 ~= random).",
        "Per-class metrics and confusion matrices are in "
        "`finetune_metrics_*.json` (written beside this report).",
        "",
        "Learning curves (train/eval loss + eval metrics per step) are in "
        "`learning_curves_linear_probe.png` and `learning_curves_full_finetune.png`.",
        "",
    ]
    for name, m in metrics.items():
        lines += [f"## {name} confusion matrix (rows=true, cols=pred)", ""]
        header = "| true \\ pred | " + " | ".join(LABELS) + " |"
        lines += [header, "|" + "---|" * (NUM_LABELS + 1)]
        for i, row in enumerate(m["confusion_matrix"]):
            lines.append(f"| {LABELS[i]} | " + " | ".join(str(c) for c in row) + " |")
        lines.append("")

    if grid_rows:
        selected = metrics.get("full_finetune", {}).get("selected_config")
        lines += [
            "## full_finetune regularization grid (ES-09/10 overfitting RFC)",
            "",
            "Valid-split selection: max macro-F1, tie-broken by the smaller train/eval "
            "loss gap. Negative-class F1 is a guardrail — the rare class must not be "
            "sacrificed. `*` marks the selected config (retrained and scored on test above).",
            "",
            "| Selected | Label smoothing | Weight decay | Dropout | Valid macro-F1 | Valid neg F1 | Loss gap |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in grid_rows:
            cfg = {k: r[k] for k in ("label_smoothing", "weight_decay", "hidden_dropout_prob")}
            mark = "*" if cfg == selected else ""
            lines.append(
                f"| {mark} | {r['label_smoothing']} | {r['weight_decay']} | "
                f"{r['hidden_dropout_prob']} | {r['valid_macro_f1']:.3f} | "
                f"{r['valid_neg_f1']:.3f} | {r['loss_gap']:.3f} |"
            )
        lines.append("")

    report_path = out_base / "finetune_analysis.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path}")


DEFAULT_REG = {"label_smoothing": 0.0, "weight_decay": WEIGHT_DECAY, "hidden_dropout_prob": None}


def run_full_finetune_grid(
    datasets: dict[str, TokenizedDataset],
    class_weights: torch.Tensor,
    tokenizer,
    out_base: Path,
    *,
    smoke: bool,
) -> tuple[dict, list[dict]]:
    """Sweep the regularization grid, select the best config, retrain it, score test.

    Selection is on the valid split (max macro-F1, tie-broken by the smaller loss
    gap); the winner is retrained into ``out_base/phrasebank_full_finetune`` —
    identical seed/config, so it reproduces the selected run — and only then scored
    on test. Grid exploration writes to a throwaway scratch dir so at most one extra
    model sits on disk at a time. Under ``smoke`` ``out_base`` is an isolated dir
    (never the shipped model location) and the grid is the single all-knobs-on
    config, enough to exercise the wiring.
    """
    grid = FULL_FINETUNE_GRID[-1:] if smoke else FULL_FINETUNE_GRID
    scratch = out_base / "phrasebank_full_finetune_grid"
    rows: list[dict] = []
    best_key = None
    best_cfg = None
    for i, reg in enumerate(grid):
        run = build_run("full_finetune", reg, scratch)
        trainer = train_tier(run, datasets, class_weights, tokenizer, smoke=smoke)
        v_macro, v_neg = valid_selection_metrics(trainer, datasets["valid"])
        gap = loss_gap_at_best(trainer)
        rows.append({**reg, "valid_macro_f1": v_macro, "valid_neg_f1": v_neg, **gap})
        print(
            f"grid {i + 1}/{len(grid)}: valid macro-F1={v_macro:.4f} "
            f"neg-F1={v_neg:.4f} loss_gap={gap['loss_gap']:.4f}"
        )
        key = (v_macro, -gap["loss_gap"])
        if best_key is None or key > best_key:
            best_key, best_cfg = key, reg
    if scratch.exists():
        shutil.rmtree(scratch)

    print(f"\nSelected full_finetune config: {best_cfg}; retraining to ship + test.")
    winner = build_run("full_finetune", best_cfg, out_base / "phrasebank_full_finetune")
    trainer = train_tier(winner, datasets, class_weights, tokenizer, smoke=smoke)
    tier_metrics = evaluate_tier(trainer, datasets["test"])
    tier_metrics["selected_config"] = best_cfg
    plot_learning_curves(
        trainer,
        out_base / "learning_curves_full_finetune.png",
        "full_finetune (selected config)",
    )
    return tier_metrics, rows


def main(smoke: bool = False) -> None:
    """Train both tiers, evaluate on test, and write reports.

    ``linear_probe`` is a single unchanged run (the low-capacity floor);
    ``full_finetune`` sweeps the ES-09/10 regularization grid before shipping the
    best config.
    """
    set_seed(RANDOM_SEED)
    print(f"Device: {get_device()}")

    # Smoke routes ALL artifacts to a throwaway dir so a dry run never clobbers the
    # shipped models/reports (train_tier clears each output dir before training).
    out_base = HERE / "_smoke_output" if smoke else HERE
    out_base.mkdir(exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    datasets = load_splits(tokenizer, smoke=smoke)
    class_weights = torch.tensor(
        compute_class_weights(datasets["train"].labels), dtype=torch.float
    )
    print(f"Class weights (neutral/positive/negative): {class_weights.tolist()}")

    metrics: dict[str, dict] = {}

    probe_run = build_run("linear_probe", DEFAULT_REG, out_base / "phrasebank_linear_probe")
    probe_trainer = train_tier(probe_run, datasets, class_weights, tokenizer, smoke=smoke)
    metrics["linear_probe"] = evaluate_tier(probe_trainer, datasets["test"])
    print(f"linear_probe test macro-F1: {metrics['linear_probe']['macro_f1']:.4f}")
    plot_learning_curves(
        probe_trainer, out_base / "learning_curves_linear_probe.png", "linear_probe"
    )

    metrics["full_finetune"], grid_rows = run_full_finetune_grid(
        datasets, class_weights, tokenizer, out_base, smoke=smoke
    )
    print(f"full_finetune test macro-F1: {metrics['full_finetune']['macro_f1']:.4f}")

    write_report(metrics, out_base, grid_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1-epoch CPU dry run on a small slice to exercise the loop.",
    )
    args = parser.parse_args()
    main(smoke=args.smoke)

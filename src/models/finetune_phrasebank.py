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
from pathlib import Path

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
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOP_PATIENCE = 2
REPORT_TO = "mlflow"               # ES-11 experiment tracking (mlflow is pinned)

# Per-tier hyperparameters. The probe gets a high LR and a taller epoch ceiling
# (low-capacity head, early stopping ends it); the full fine-tune uses the
# standard BERT recipe.
TIERS = {
    "linear_probe": {"freeze_backbone": True, "learning_rate": 2e-3, "epochs": 20},
    "full_finetune": {"freeze_backbone": False, "learning_rate": 2e-5, "epochs": 10},
}


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
    """``Trainer`` with a class-weighted cross-entropy loss.

    The weight tensor is moved onto the logits' device on each step so the same
    trainer works on CPU, MPS, or CUDA without pre-placing the weights.
    """

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = (
            self._class_weights.to(outputs.logits.device)
            if self._class_weights is not None
            else None
        )
        loss = nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def build_model() -> BertForSequenceClassification:
    """Load finbert-pretrain with a fresh 3-class head and the label mapping."""
    return BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
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
    output_dir: Path, learning_rate: float, epochs: int, *, smoke: bool
) -> TrainingArguments:
    """Shared TrainingArguments; ``smoke`` forces CPU and disables reporting/fp16."""
    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=1 if smoke else epochs,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=RANDOM_SEED,
        report_to="none" if smoke else REPORT_TO,
        fp16=torch.cuda.is_available() and not smoke,
        use_cpu=smoke,
    )


def train_tier(
    name: str,
    datasets: dict[str, TokenizedDataset],
    class_weights: torch.Tensor,
    tokenizer,
    *,
    smoke: bool,
) -> Trainer:
    """Build a fresh model, train one tier, and save the best model to disk."""
    config = TIERS[name]
    model = build_model()
    if config["freeze_backbone"]:
        freeze_backbone(model)

    output_dir = HERE / f"phrasebank_{name}"
    args = build_training_args(
        output_dir, config["learning_rate"], config["epochs"], smoke=smoke
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )
    print(f"\n=== Training tier: {name} ===")
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


def write_report(metrics: dict[str, dict]) -> None:
    """Write per-tier metrics JSON and a combined markdown analysis report."""
    for name, tier_metrics in metrics.items():
        path = HERE / f"finetune_metrics_{name}.json"
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
    ]
    for name, m in metrics.items():
        lines += [f"## {name} confusion matrix (rows=true, cols=pred)", ""]
        header = "| true \\ pred | " + " | ".join(LABELS) + " |"
        lines += [header, "|" + "---|" * (NUM_LABELS + 1)]
        for i, row in enumerate(m["confusion_matrix"]):
            lines.append(f"| {LABELS[i]} | " + " | ".join(str(c) for c in row) + " |")
        lines.append("")

    report_path = HERE / "finetune_analysis.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path}")


def main(smoke: bool = False) -> None:
    """Train both tiers, evaluate on test, and write reports."""
    set_seed(RANDOM_SEED)
    print(f"Device: {get_device()}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    datasets = load_splits(tokenizer, smoke=smoke)
    class_weights = torch.tensor(
        compute_class_weights(datasets["train"].labels), dtype=torch.float
    )
    print(f"Class weights (neutral/positive/negative): {class_weights.tolist()}")

    metrics: dict[str, dict] = {}
    for name in TIERS:
        trainer = train_tier(name, datasets, class_weights, tokenizer, smoke=smoke)
        metrics[name] = evaluate_tier(trainer, datasets["test"])
        print(f"{name} test macro-F1: {metrics[name]['macro_f1']:.4f}")

    write_report(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1-epoch CPU dry run on a small slice to exercise the loop.",
    )
    args = parser.parse_args()
    main(smoke=args.smoke)

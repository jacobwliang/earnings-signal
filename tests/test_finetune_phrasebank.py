"""Unit tests for ES-09/10 FinBERT fine-tuning helpers.

Covers the pure logic that does not need a downloaded model or the real
(gitignored) dataset: sqrt-inverse-frequency class weights, the macro-F1
metric function, and the tokenized-dataset wiring (a whitespace tokenizer
stands in for BertTokenizer so no network/model download is required).
"""

import numpy as np

from src.models.finetune_phrasebank import (
    LABELS,
    NUM_LABELS,
    TokenizedDataset,
    compute_class_weights,
    compute_metrics,
)


class WhitespaceTokenizer:
    """Deterministic stand-in for BertTokenizer.

    Returns the batch-encoding shape ``TokenizedDataset`` relies on: a dict of
    per-example lists for ``input_ids`` and ``attention_mask``. One space-
    separated word == one token id (its length), truncated to ``max_length``.
    """

    def __call__(self, texts, truncation=True, max_length=96):
        ids = [[len(w) for w in t.split()][:max_length] for t in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(seq) for seq in ids]}


def test_class_weights_upweight_rare_classes_and_average_to_one():
    # Train-split counts: neutral=1713, positive=709, negative=336.
    labels = [0] * 1713 + [1] * 709 + [2] * 336
    weights = compute_class_weights(labels)

    assert weights.shape == (NUM_LABELS,)
    # Rarer class -> larger weight; ordering is neutral < positive < negative.
    assert weights[0] < weights[1] < weights[2]
    # Normalized to mean 1 so the loss scale stays comparable to unweighted.
    assert np.isclose(weights.mean(), 1.0)
    # sqrt(N_total / N_c) upweights negative less than plain inverse frequency
    # would: neg/neu weight ratio is sqrt(1713/336) ~= 2.26, not 1713/336.
    assert np.isclose(weights[2] / weights[0], np.sqrt(1713 / 336))


def test_class_weights_reject_missing_class():
    import pytest

    with pytest.raises(ValueError):
        compute_class_weights([0, 0, 1, 1])  # no negative examples


def test_compute_metrics_macro_f1_and_accuracy():
    # 3 of 4 correct; one neutral predicted as positive.
    logits = np.array(
        [
            [2.0, 0.0, 0.0],  # true 0 -> pred 0
            [0.0, 2.0, 0.0],  # true 1 -> pred 1
            [0.0, 0.0, 2.0],  # true 2 -> pred 2
            [0.0, 2.0, 0.0],  # true 0 -> pred 1 (wrong)
        ]
    )
    labels = np.array([0, 1, 2, 0])
    out = compute_metrics((logits, labels))

    assert np.isclose(out["accuracy"], 0.75)
    assert 0.0 < out["f1"] <= 1.0


def test_tokenized_dataset_shapes_and_labels():
    tokenizer = WhitespaceTokenizer()
    texts = ["revenue rose sharply", "guidance cut"]
    labels = [1, 2]
    dataset = TokenizedDataset(texts, labels, tokenizer)

    assert len(dataset) == 2
    first = dataset[0]
    assert set(first) == {"input_ids", "attention_mask", "labels"}
    assert first["labels"] == 1
    assert len(first["input_ids"]) == 3          # three words in the first text
    assert len(first["attention_mask"]) == len(first["input_ids"])


def test_label_contract_matches_project_convention():
    # The index IS the label id; downstream scoring depends on this exact order.
    assert LABELS == ("neutral", "positive", "negative")

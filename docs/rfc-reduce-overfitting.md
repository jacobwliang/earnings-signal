# RFC: Regularize the full fine-tune to reduce overfitting (ES-09/10)

**Status:** proposed
**Scope:** `src/models/finetune_phrasebank.py`, `full_finetune` tier only

## Summary

Add a small regularization grid (label smoothing, weight decay, dropout) to the `full_finetune`
tier so the model's train/eval loss gap shrinks without losing macro-F1.

## Motivation

The shipped `full_finetune` model is strong (0.935 test macro-F1, 0.954 accuracy) but the
epoch-level training curves show textbook overfitting: **train loss falls to ~0.04 while eval loss
bottoms at epoch 2 (~0.227) and drifts back up (~0.249) even as eval accuracy/F1 keep rising.**
That divergence is *overconfidence* — the model sharpens its training-set probabilities faster than
it generalizes. It does not currently cost accuracy (we select the best-F1 checkpoint), but it
signals a poorly calibrated model with headroom to regularize.

## Proposed change

Hold LR at the working `2e-5` and sweep an 8-run grid over regularization strength:

- `label_smoothing_factor ∈ {0.0, 0.1}` — the most on-target lever; soft targets directly counter
  overconfidence.
  - **Load-bearing detail:** `WeightedTrainer.compute_loss` calls `nn.functional.cross_entropy`
    directly, so it *ignores* `TrainingArguments.label_smoothing_factor`. Smoothing must be threaded
    into the custom loss (`cross_entropy(..., label_smoothing=...)`).
- `weight_decay ∈ {0.01, 0.1}`
- `hidden_dropout_prob ∈ {0.1, 0.3}` (passed into `build_model`)

Layer freezing (`freeze_lower_layers`) is held as an optional extra axis only if the grid above
doesn't shrink the gap. The `linear_probe` tier is a deliberate low-capacity floor and is left
unchanged.

**Enabler:** switch `full_finetune` to step-based eval (`eval_steps=25`, ~7 evals/epoch) so the
loss gap is actually measurable per config. `EarlyStoppingCallback` patience counts eval *events*,
not epochs, so it is rescaled (~10–14) to keep "no improvement" ≈ 1.5–2 epochs.

## Success criteria & guardrails

Select the config that **holds or improves test macro-F1 (≥ ~0.935) while shrinking the train/eval
loss gap** and **does not drop negative-class F1 below the current ~0.894**. Per-run we report valid
macro-F1, the loss gap, and negative-class F1 — macro-F1 alone can hide a sacrificed rare class.

## Risks & non-goals

- **Risk:** over-regularizing the rare negative class (~65 train / 42 test examples). Guardrail
  above is designed to catch this.
- **Non-goal:** no architecture or data-pipeline changes.
- **Out of scope:** single-seed run-to-run variance and test-metric confidence intervals remain
  unmeasured; a smaller gap with equal metrics is still a valid win (better calibration). If nothing
  beats the tradeoff, the honest conclusion is that the current recipe is already near-optimal.

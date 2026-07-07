# Fine-tuned FinBERT on Financial PhraseBank (ES-09/10)

Test-split results for the two-tier baseline ladder. Label ids: 0=neutral, 1=positive, 2=negative. The negative class has ~42 test examples, so its recall carries a wide CI (~+/-14pp).

| Tier | Macro-F1 | Accuracy | Neg recall | Neg F1 |
|---|---|---|---|---|
| full_finetune | 0.938 | 0.954 | 0.881 | 0.914 |

Compare against the chance-floor baseline in [baseline_analysis.md](baseline_analysis.md) (macro-F1 ~= random).
Per-class metrics and confusion matrices are in `finetune_metrics_*.json` (written beside this report).

Learning curves (train/eval loss + eval metrics per step) are in `learning_curves_linear_probe.png` and `learning_curves_full_finetune.png`.

## full_finetune confusion matrix (rows=true, cols=pred)

| true \ pred | neutral | positive | negative |
|---|---|---|---|
| neutral | 207 | 6 | 1 |
| positive | 3 | 85 | 1 |
| negative | 2 | 3 | 37 |

## full_finetune regularization grid (ES-09/10 overfitting RFC)

Valid-split selection: max macro-F1, tie-broken by the smaller train/eval loss gap. Negative-class F1 is a guardrail — the rare class must not be sacrificed. `*` marks the selected config (retrained and scored on test above).

| Selected | Label smoothing | Weight decay | Dropout | Valid macro-F1 | Valid neg F1 | Loss gap |
|---|---|---|---|---|---|---|
|  | 0.0 | 0.01 | 0.1 | 0.953 | 0.952 | 0.189 |
|  | 0.0 | 0.01 | 0.3 | 0.933 | 0.907 | -0.106 |
|  | 0.0 | 0.1 | 0.1 | 0.953 | 0.952 | 0.188 |
|  | 0.0 | 0.1 | 0.3 | 0.933 | 0.907 | -0.106 |
|  | 0.1 | 0.01 | 0.1 | 0.955 | 0.941 | 0.081 |
|  | 0.1 | 0.01 | 0.3 | 0.936 | 0.916 | 0.016 |
| * | 0.1 | 0.1 | 0.1 | 0.955 | 0.941 | 0.081 |
|  | 0.1 | 0.1 | 0.3 | 0.936 | 0.916 | 0.016 |

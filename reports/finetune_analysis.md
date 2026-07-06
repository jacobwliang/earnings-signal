# Fine-tuned FinBERT on Financial PhraseBank (ES-09/10)

Test-split results for the two-tier baseline ladder. Label ids: 0=neutral, 1=positive, 2=negative. The negative class has ~42 test examples, so its recall carries a wide CI (~+/-14pp).

| Tier | Macro-F1 | Accuracy | Neg recall | Neg F1 |
|---|---|---|---|---|
| linear_probe | 0.804 | 0.855 | 0.619 | 0.712 |
| full_finetune | 0.935 | 0.954 | 0.905 | 0.894 |

Compare against the chance-floor baseline in [baseline_analysis.md](baseline_analysis.md) (macro-F1 ~= random).
Per-class metrics and confusion matrices are in `finetune_metrics_*.json` (written beside this report).

## linear_probe confusion matrix (rows=true, cols=pred)

| true \ pred | neutral | positive | negative |
|---|---|---|---|
| neutral | 188 | 24 | 2 |
| positive | 5 | 81 | 3 |
| negative | 3 | 13 | 26 |

## full_finetune confusion matrix (rows=true, cols=pred)

| true \ pred | neutral | positive | negative |
|---|---|---|---|
| neutral | 207 | 4 | 3 |
| positive | 3 | 84 | 2 |
| negative | 2 | 2 | 38 |

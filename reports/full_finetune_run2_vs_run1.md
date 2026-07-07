# Full Fine-Tune: Run 2 vs Run 1

Run 2 is the better model. The test scores are basically a tie (macro-F1 0.935 → 0.938, accuracy 0.954 both times), but it overfits less and trains faster.

I added light regularization (label smoothing, weight decay, dropout) and cut the number of epochs. In Run 1 the model kept memorizing the training data past the point where it helped. In Run 2 the eval loss levels off and stays flat instead of drifting apart from the train loss, so the extra epochs weren't doing anything useful anyway.

Same accuracy, less overfitting, shorter training.

.PHONY: data baseline finetune analysis pipeline

data:
	python src/data/scraper.py
	python src/data/prices.py
	python src/data/preprocess.py

baseline:
	python src/model/inference.py --mode baseline

finetune:
	python src/model/train.py
	python src/model/inference.py --mode finetuned

# Correlation analysis loads both the baseline and fine-tuned scores together
# (the paired bootstrap needs both), so it runs once — not per model.
analysis:
	python -m src.data.download_index
	python -m src.analysis.correlate_returns

pipeline: data baseline finetune analysis

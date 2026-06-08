.PHONY: data baseline finetune analysis pipeline

data:
	python src/data/scraper.py
	python src/data/prices.py
	python src/data/preprocess.py

baseline:
	python src/model/inference.py --mode baseline
	python src/analysis/correlations.py --model baseline

finetune:
	python src/model/train.py
	python src/model/inference.py --mode finetuned
	python src/analysis/correlations.py --model finetuned

analysis:
	python src/analysis/correlations.py --subgroup market_cap

pipeline: data baseline finetune analysis

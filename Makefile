PY ?= python
CONFIG_DIR := configs

.PHONY: data semids sft grpo ablation eval slates sasrec test clean

data:
	$(PY) scripts/00_prepare_data.py --config $(CONFIG_DIR)/data.yaml

semids:
	$(PY) scripts/01_build_semantic_ids.py --config $(CONFIG_DIR)/rqvae.yaml

sft:
	$(PY) scripts/02_train_sft.py --config $(CONFIG_DIR)/sft.yaml

grpo:
	$(PY) scripts/03_train_grpo.py --config $(CONFIG_DIR)/grpo.yaml --sweep

ablation:
	$(PY) scripts/03_train_grpo.py --config $(CONFIG_DIR)/grpo.yaml --cap-ablation

eval:
	$(PY) scripts/04_evaluate.py --config $(CONFIG_DIR)/sft.yaml --all

slates:
	$(PY) scripts/05_slate_analysis.py

sasrec:
	$(PY) scripts/run_sasrec.py --config $(CONFIG_DIR)/sasrec.yaml

test:
	$(PY) -m pytest -q

clean:
	rm -rf checkpoints/ __pycache__/ .pytest_cache/

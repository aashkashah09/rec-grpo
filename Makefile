PY ?= python
CONFIG_DIR := configs

.PHONY: data semids sft grpo test clean

data:
	$(PY) scripts/00_prepare_data.py --config $(CONFIG_DIR)/data.yaml

semids:
	$(PY) scripts/01_build_semantic_ids.py --config $(CONFIG_DIR)/rqvae.yaml

sft:
	$(PY) scripts/02_train_sft.py --config $(CONFIG_DIR)/sft.yaml

grpo:
	$(PY) scripts/03_train_grpo.py --config $(CONFIG_DIR)/grpo.yaml --sweep

test:
	$(PY) -m pytest -q

clean:
	rm -rf checkpoints/ __pycache__/ .pytest_cache/

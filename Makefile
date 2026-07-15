PY ?= python
CONFIG_DIR := configs

.PHONY: data semids clean

data:
	$(PY) scripts/00_prepare_data.py --config $(CONFIG_DIR)/data.yaml

semids:
	$(PY) scripts/01_build_semantic_ids.py --config $(CONFIG_DIR)/rqvae.yaml

clean:
	rm -rf checkpoints/ __pycache__/ .pytest_cache/

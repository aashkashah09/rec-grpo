"""Unit tests for SFT warmup: cost estimate and the --confirm-spend gate (rider 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from specialist_router.config import EndpointConfig
from specialist_router.training.sft_warmup import estimate_demo_cost, main

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"

_SERVING_WITH_ENDPOINT = """\
schema_version: 1
seed: 0
backend: stub
stub:
  local:
    base_quality: 0.5
    difficulty_penalty: 0.1
    cost_usd_mean: 0.001
    cost_usd_std: 0.0
    latency_s_mean: 0.5
    latency_s_std: 0.0
  api:
    base_quality: 0.9
    difficulty_penalty: 0.05
    cost_usd_mean: 0.02
    cost_usd_std: 0.0
    latency_s_mean: 2.0
    latency_s_std: 0.0
local_endpoint: null
api_endpoint:
  base_url: "https://frontier.example/v1"
  model: "frontier-large"
  api_key_env: "FRONTIER_API_KEY"
  price_prompt_per_1k_usd: 0.003
  price_completion_per_1k_usd: 0.015
  timeout_s: 30.0
"""


def test_estimate_demo_cost_scales_with_attempts() -> None:
    endpoint = EndpointConfig(
        base_url="x",
        model="m",
        price_prompt_per_1k_usd=0.003,
        price_completion_per_1k_usd=0.015,
        timeout_s=30.0,
    )
    one = estimate_demo_cost(endpoint, 1)
    hundred = estimate_demo_cost(endpoint, 100)
    assert one > 0.0
    assert hundred == pytest.approx(one * 100)


def test_main_without_confirm_spend_makes_no_api_call(tmp_path: Path, capsys) -> None:
    serving = tmp_path / "serving.yaml"
    serving.write_text(_SERVING_WITH_ENDPOINT)

    # No --confirm-spend: must print the estimate and return 2 WITHOUT importing/calling any API.
    code = main(
        [
            "--config",
            str(_CONFIGS / "grpo.yaml"),
            "--serving-config",
            str(serving),
        ]
    )
    out = capsys.readouterr().out
    assert code == 2  # spend-gate exit code
    assert "Estimated API cost" in out
    assert "--confirm-spend" in out
    assert "No API calls made" in out


def test_main_errors_without_api_endpoint(tmp_path: Path) -> None:
    # Truncate the fixture before its api_endpoint block and set it to null.
    null_endpoint = _SERVING_WITH_ENDPOINT.split("api_endpoint:")[0] + "api_endpoint: null\n"
    serving = tmp_path / "serving.yaml"
    serving.write_text(null_endpoint)
    with pytest.raises(SystemExit):
        main(["--config", str(_CONFIGS / "grpo.yaml"), "--serving-config", str(serving)])

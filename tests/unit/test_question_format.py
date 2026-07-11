"""Every rendered question must explicitly state its expected answer format.

This enforces the requirement that a solver never has to guess units: each question contains
the format marker for its answer type (USD / decimal fraction / percentage points / integer /
ordered list). Checked across all templates on the frozen fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from specialist_router.config import Config
from specialist_router.env.tasks import FORMAT_MARKERS, TEMPLATES, EnvIndex


@pytest.mark.parametrize("template_id", sorted(TEMPLATES))
def test_rendered_question_states_answer_format(
    template_id: str, mini_index: EnvIndex, env_config: Config
) -> None:
    template = TEMPLATES[template_id]
    rng = np.random.default_rng(0)
    params = template.sample_params(rng, mini_index, env_config)
    question = template.render_question(params)
    marker = FORMAT_MARKERS[template.answer_type]
    assert marker in question, f"{template_id} question omits format marker {marker!r}: {question}"


def test_every_answer_type_has_a_format_marker() -> None:
    """No template may use an answer type without a declared format marker."""
    for template in TEMPLATES.values():
        assert template.answer_type in FORMAT_MARKERS

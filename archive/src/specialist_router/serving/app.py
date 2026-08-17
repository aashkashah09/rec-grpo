"""FastAPI wrapper exposing the routing service as ``POST /solve`` (Phase 4 demo; not used in CI).

The heavy lifting is in :class:`~specialist_router.serving.service.RouterService`; this module only
adapts it to HTTP. ``fastapi`` is imported lazily inside :func:`create_app`, so importing this
module (e.g. for ``mypy``) never requires the optional ``serving`` extra. Because verification
needs programmatic ground truth, ``/solve`` operates over a pre-generated task set keyed by
``task_id`` rather than free-form questions.
"""

from __future__ import annotations

import itertools

import numpy as np

from specialist_router.env.records import RouterDecision, Task
from specialist_router.serving.service import RouterService, synthetic_timestamp


def create_app(tasks: list[Task], service: RouterService, seed: int) -> object:
    """Build a FastAPI app serving ``POST /solve`` over the given task set.

    Args:
        tasks: The pre-generated tasks (with ground truth) the service can route.
        service: The configured routing service.
        seed: Seed for the action-sampling RNG.

    Returns:
        A FastAPI application instance (typed ``object`` so this module imports without fastapi).
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    tasks_by_id = {task.task_id: task for task in tasks}
    rng = np.random.default_rng(seed)
    counter = itertools.count()

    app = FastAPI(title="Specialist + Router — /solve")

    class SolveRequest(BaseModel):
        task_id: str

    @app.post("/solve")  # type: ignore[untyped-decorator]
    def solve(request: SolveRequest) -> RouterDecision:
        task = tasks_by_id.get(request.task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"unknown task_id: {request.task_id}")
        index = next(counter)
        return service.decide(task, rng, index, synthetic_timestamp(index))

    return app

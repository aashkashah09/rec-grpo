"""CLI: launch the FastAPI ``/solve`` service over a stub-backed router (needs the serving extra).

Builds a task set and a stub RouterService, then serves it with uvicorn. Requires
``pip install '.[serving]'`` (fastapi + uvicorn); not part of CI.
"""

from __future__ import annotations

import argparse

from specialist_router.analysis.pipeline import _featurizer, _tasks_for
from specialist_router.config import (
    load_config,
    load_router_config,
    load_serving_config,
)
from specialist_router.router.policies import UniformPolicy
from specialist_router.serving.app import create_app
from specialist_router.serving.service import RouterService, build_stub_runners


def main() -> None:
    """Parse arguments, build the service, and run uvicorn."""
    parser = argparse.ArgumentParser(
        description="Serve the /solve routing endpoint (stub backend)."
    )
    parser.add_argument("--env-config", default="configs/env.mini.yaml")
    parser.add_argument("--router-config", default="configs/router.yaml")
    parser.add_argument("--serving-config", default="configs/serving.yaml")
    parser.add_argument("--n", type=int, default=200, help="Number of tasks to serve.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    env = load_config(args.env_config)
    router = load_router_config(args.router_config)
    serving = load_serving_config(args.serving_config)

    tasks, used = _tasks_for(env, args.n, env.seed)
    runners = build_stub_runners(serving.seed, serving.stub.local, serving.stub.api, used.verifier)
    service = RouterService(
        _featurizer(env, router), UniformPolicy(), runners, router.reward, router.seed
    )
    app = create_app(tasks, service, router.seed)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

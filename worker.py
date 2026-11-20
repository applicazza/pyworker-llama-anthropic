"""llama.cpp PyWorker serving both OpenAI and Anthropic Messages routes.

start_server.sh runs this top-level worker.py (python3 -m worker) before falling back to
workers/<BACKEND>/worker.py. It reuses the request parsing of workers/openai/core.py and
adds Anthropic's /v1/messages and /v1/messages/count_tokens, which llama-server serves
natively. The startup benchmark is one short request: the default (30 x 500 tokens) costs
minutes of billed GPU time on every cold start of a single-slot server.
"""

import os

from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

from workers.openai.core import MODEL_SERVER_PORT, MODEL_SERVER_URL, _env_lines, request_parser

BENCH_TOKENS = 16


def _max_tokens(data: dict) -> float:
    """Workload in output tokens; Anthropic requires max_tokens, OpenAI may omit it."""
    value = data.get("max_tokens") or data.get("max_completion_tokens") or 512
    try:
        return max(float(value), 1.0)
    except (TypeError, ValueError):
        return 512.0


def _bench() -> dict:
    return {"prompt": "The capital of France is", "max_tokens": BENCH_TOKENS, "temperature": 0}


def _route(path: str, workload, benchmark: bool = False) -> HandlerConfig:
    return HandlerConfig(
        route=path,
        allow_parallel_requests=True,  # llama-server queues on its own single slot
        max_queue_time=1800.0,  # a near-1M-token prompt can take tens of minutes
        workload_calculator=workload,
        request_parser=request_parser,
        benchmark_config=BenchmarkConfig(generator=_bench, runs=1, concurrency=1)
        if benchmark else None,
    )


handlers = [
    _route("/v1/completions", _max_tokens, benchmark=True),
    _route("/v1/chat/completions", _max_tokens),
    _route("/v1/messages", _max_tokens),
    _route("/v1/messages/count_tokens", lambda data: 1.0),
]

print("serving: " + ", ".join(h.route for h in handlers), flush=True)

Worker(WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=os.environ.get("MODEL_LOG", "/var/log/portal/llama.log"),
    model_healthcheck_url=os.environ.get("MODEL_HEALTH_ENDPOINT", "/health"),
    handlers=handlers,
    log_action_config=LogActionConfig(
        on_load=_env_lines("MODEL_LOAD_LOG_MSG", ["model loaded"]),
        on_error=_env_lines("MODEL_ERROR_LOG_MSGS",
                            ["INFO exited: llama", "error loading model", "failed to load model"]),
        on_info=_env_lines("MODEL_INFO_LOG_MSGS", ['"message":"Download']),
    ),
)).run()

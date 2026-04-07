"""
Baseline inference for SQL Lab.

Contract:
- Required env: HF_TOKEN
- Defaults allowed: API_BASE_URL, MODEL_NAME
- Optional env:
  - OPENENV_BASE_URL (connect to a running server)
  - LOCAL_IMAGE_NAME (run environment via local Docker image)

Stdout is structured as exactly three record types:
  START {json}
  STEP {json}
  END {json}
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from openenv.core.containers.runtime import LocalDockerProvider

from sql_lab.client import SQLLabClient
from sql_lab.models import SQLAction

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-api-base-url>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model-name>")
HF_TOKEN = os.getenv("HF_TOKEN")

# URL mode (connect to an already-running OpenEnv server)
OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:8000")

# Docker-image mode (optional): run env from local image via LocalDockerProvider
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASK_ORDER = ("easy", "medium", "hard")
MAX_STEPS_PER_TASK = 5
MAX_QUERY_CHARS = 600

TASK_DESCRIPTIONS = {
    "easy": "List all names of employees in the Engineering department (dept_id = 1).",
    "medium": "Find the names of all departments along with the average salary of employees in each department.",
    "hard": "List the names of employees who have made total sales exceeding 8000.",
}


def _log(event: str, payload: Dict[str, Any]) -> None:
    # Deterministic JSON for machine parsing.
    sys.stdout.write(f"{event} {json.dumps(payload, sort_keys=True)}\n")
    sys.stdout.flush()


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def run_task(task_id: str, env: Any, client: OpenAI) -> float:
    reset_result = env.reset()
    obs = reset_result.observation
    desc = TASK_DESCRIPTIONS[task_id]

    for step in range(MAX_STEPS_PER_TASK):
        prompt = f"""You are an expert SQL developer.
Task: {desc}
Schema: {obs.sql_schema}
Last result: {obs.result}
Last error: {obs.error}

Reply ONLY with a valid SQL query inside a ```sql ... ``` block. No explanation."""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        if "```sql" in content:
            query = content.split("```sql", 1)[1].split("```", 1)[0].strip()
        else:
            query = content.strip()

        result = env.step(SQLAction(task_id=task_id, query=query))
        obs = result.observation
        _log(
            "STEP",
            {
                "task_id": task_id,
                "step_index": step,
                "query": _truncate(query, MAX_QUERY_CHARS),
                "reward": result.reward,
                "done": bool(result.done),
                "has_error": bool(getattr(obs, "error", None)),
            },
        )
        if result.done:
            break

    st = env.state()
    final_score = float(st.scores_by_task.get(task_id, 0.0))
    return final_score


def main() -> None:
    if not HF_TOKEN:
        print("Missing required environment variable: HF_TOKEN", file=sys.stderr)
        sys.exit(1)

    start_ts = time.time()
    _log(
        "START",
        {
            "api_base_url": API_BASE_URL,
            "model_name": MODEL_NAME,
            "openenv_base_url": None if LOCAL_IMAGE_NAME else OPENENV_BASE_URL,
            "local_image_name": LOCAL_IMAGE_NAME,
            "max_steps_per_task": MAX_STEPS_PER_TASK,
            "started_at_unix_s": start_ts,
        },
    )

    llm = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    provider: Optional[LocalDockerProvider] = None
    base_url: str
    if LOCAL_IMAGE_NAME:
        provider = LocalDockerProvider()
        base_url = provider.start_container(LOCAL_IMAGE_NAME)
        provider.wait_for_ready(base_url)
    else:
        base_url = OPENENV_BASE_URL

    scores: List[float] = []
    async_client = SQLLabClient(base_url=base_url, provider=provider)
    sync_env = async_client.sync()
    try:
        with sync_env:
            for task_id in TASK_ORDER:
                scores.append(run_task(task_id, sync_env, llm))
    finally:
        # Ensure docker container is stopped if we started one.
        try:
            sync_env.close()
        except Exception:
            pass

    mean = sum(scores) / len(scores)
    _log(
        "END",
        {
            "mean_score": mean,
            "scores_by_task": dict(zip(TASK_ORDER, scores)),
            "elapsed_s": round(time.time() - start_ts, 6),
        },
    )


if __name__ == "__main__":
    main()

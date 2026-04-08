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
import traceback
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from openenv.core.containers.runtime import LocalDockerProvider

from sql_lab.client import SQLLabClient
from sql_lab.models import SQLAction

load_dotenv()

_DEFAULT_API_BASE_URL = "<your-active-api-base-url>"
_DEFAULT_MODEL_NAME = "<your-active-model-name>"

API_BASE_URL = os.getenv("API_BASE_URL", _DEFAULT_API_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", _DEFAULT_MODEL_NAME)
HF_TOKEN = os.getenv("HF_TOKEN")

# URL mode (connect to an already-running OpenEnv server)
OPENENV_BASE_URL = (
    os.getenv("OPENENV_BASE_URL")
    or os.getenv("OPENENV_URL")
    or os.getenv("SPACE_URL")
    or "http://127.0.0.1:8000"
)

# Docker-image mode (optional): run env from local image via LocalDockerProvider
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# LLM resilience (OpenAI-compatible providers / HF router)
_LLM_MAX_ATTEMPTS = max(1, int(os.getenv("OPENAI_MAX_ATTEMPTS", "6")))
_LLM_BACKOFF_S = float(os.getenv("OPENAI_RETRY_BACKOFF_S", "2.0"))
_LLM_TIMEOUT_S = float(os.getenv("OPENAI_TIMEOUT_S", "120.0"))
_LLM_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))

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


def _is_placeholder_config() -> bool:
    """True when required non-default URLs/model were not provided (common CI/harness footgun)."""
    if not HF_TOKEN:
        return True
    if not API_BASE_URL or API_BASE_URL == _DEFAULT_API_BASE_URL:
        return True
    if not MODEL_NAME or MODEL_NAME == _DEFAULT_MODEL_NAME:
        return True
    if "your-active" in API_BASE_URL.lower() or "your-active" in MODEL_NAME.lower():
        return True
    return False


def _check_openenv_http_ready(base_url: str, timeout_s: float = 15.0) -> None:
    """Lightweight readiness probe (avoids failing later inside the agent loop)."""
    root = base_url.rstrip("/")
    health = f"{root}/health"
    try:
        req = urllib.request.Request(health, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if getattr(resp, "status", 200) != 200:
                raise RuntimeError(f"GET {health} returned HTTP {getattr(resp, 'status', 'unknown')}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenEnv health check failed: GET {health} -> HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenEnv health check failed: cannot reach {health} ({e})") from e


def _is_retryable_openai_error(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        try:
            c = int(code)
        except (TypeError, ValueError):
            return False
        return c in (408, 409, 425, 429, 500, 502, 503, 504)
    return False


def _chat_completion_text(client: OpenAI, prompt: str) -> Tuple[str, Optional[str]]:
    """
    Returns (assistant_text, error_string_if_any).

    Never raises for ordinary provider/network failures; returns a synthetic message so the env step can continue.
    """
    last_err: Optional[str] = None
    for attempt in range(_LLM_MAX_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=_LLM_MAX_TOKENS,
            )
            choices = getattr(response, "choices", None) or []
            if not choices:
                return "", "LLM returned no choices"
            msg = choices[0].message
            content = (getattr(msg, "content", None) or "") if msg is not None else ""
            return str(content), None
        except AuthenticationError as e:
            last_err = f"{type(e).__name__}: {e}"
            break
        except BadRequestError as e:
            last_err = f"{type(e).__name__}: {e}"
            break
        except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError) as e:
            last_err = f"{type(e).__name__}: {e}"
            if _is_retryable_openai_error(e) and attempt < _LLM_MAX_ATTEMPTS - 1:
                sleep_for = _LLM_BACKOFF_S * (2**attempt)
                time.sleep(sleep_for)
                continue
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            break

    # Deterministic fallback: keep the run alive for harnesses / partial grading.
    return "SELECT 1;", last_err or "LLM call failed"


def run_task(task_id: str, env: Any, client: OpenAI) -> float:
    try:
        reset_result = env.reset()
        obs = reset_result.observation
    except Exception as e:
        print(f"Env reset exception (task={task_id}): {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 0.0

    desc = TASK_DESCRIPTIONS[task_id]

    for step in range(MAX_STEPS_PER_TASK):
        prompt = f"""You are an expert SQL developer.
Task: {desc}
Schema: {obs.sql_schema}
Last result: {obs.result}
Last error: {obs.error}

Reply ONLY with a valid SQL query inside a ```sql ... ``` block. No explanation."""

        content, llm_err = _chat_completion_text(client, prompt)
        if llm_err:
            print(f"LLM warning (task={task_id}, step={step}): {llm_err}", file=sys.stderr)

        try:
            if "```sql" in content:
                query = content.split("```sql", 1)[1].split("```", 1)[0].strip()
            else:
                query = content.strip()
        except Exception as e:
            print(f"Parse warning (task={task_id}, step={step}): {type(e).__name__}: {e}", file=sys.stderr)
            query = "SELECT 1;"

        if not query:
            query = "SELECT 1;"

        try:
            result = env.step(SQLAction(task_id=task_id, query=query))
        except Exception as e:
            print(f"Env step exception (task={task_id}, step={step}): {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            query = "SELECT 1;"
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
                "llm_error": llm_err,
            },
        )
        if result.done:
            break

    try:
        st = env.state()
        final_score = float(st.scores_by_task.get(task_id, 0.0))
    except Exception as e:
        print(f"Env state exception (task={task_id}): {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        final_score = 0.0
    return final_score


def main() -> None:
    if _is_placeholder_config():
        print(
            "Invalid configuration: require HF_TOKEN and non-placeholder API_BASE_URL/MODEL_NAME.\n"
            "In Hugging Face Spaces, set secrets/variables: HF_TOKEN, API_BASE_URL, MODEL_NAME.\n"
            "Optional: OPENENV_BASE_URL should point at the running OpenEnv server (often this Space URL).",
            file=sys.stderr,
        )
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
            "llm_max_attempts": _LLM_MAX_ATTEMPTS,
            "llm_timeout_s": _LLM_TIMEOUT_S,
        },
    )

    llm = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
        timeout=_LLM_TIMEOUT_S,
        max_retries=0,  # explicit retries are implemented in _chat_completion_text
    )

    provider: Optional[LocalDockerProvider] = None
    base_url: str
    if LOCAL_IMAGE_NAME:
        provider = LocalDockerProvider()
        base_url = provider.start_container(LOCAL_IMAGE_NAME)
        provider.wait_for_ready(base_url)
    else:
        base_url = OPENENV_BASE_URL

    scores: List[float] = []
    try:
        _check_openenv_http_ready(base_url)
    except Exception as e:
        print(f"OpenEnv not reachable at {base_url!r}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        _log(
            "END",
            {
                "ok": False,
                "error": f"openenv_unreachable: {e}",
                "mean_score": None,
                "scores_by_task": {},
                "elapsed_s": round(time.time() - start_ts, 6),
            },
        )
        sys.exit(1)

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

    mean = sum(scores) / len(scores) if scores else 0.0
    end_payload: Dict[str, Any] = {
        "mean_score": mean,
        "scores_by_task": dict(zip(TASK_ORDER, scores)),
        "elapsed_s": round(time.time() - start_ts, 6),
        "ok": True,
    }
    _log("END", end_payload)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        # Last-resort guardrail for harnesses that require a non-crashing entrypoint.
        print(f"Unhandled fatal error: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        _log(
            "END",
            {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "mean_score": None,
                "scores_by_task": {},
                "elapsed_s": None,
            },
        )
        sys.exit(1)

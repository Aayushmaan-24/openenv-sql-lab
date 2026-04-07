"""
Baseline inference for SQL Lab (OpenAI-compatible API via API_BASE_URL).

Required env: API_BASE_URL, MODEL_NAME, HF_TOKEN (or OPENAI_API_KEY as fallback).
Optional: OPENENV_BASE_URL (default http://127.0.0.1:8000) for the OpenEnv server.
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI

from sql_lab.client import SQLLabClient
from sql_lab.models import SQLAction

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:8000")

TASK_ORDER = ("easy", "medium", "hard")

TASK_DESCRIPTIONS = {
    "easy": "List all names of employees in the Engineering department (dept_id = 1).",
    "medium": "Find the names of all departments along with the average salary of employees in each department.",
    "hard": "List the names of employees who have made total sales exceeding 8000.",
}


async def run_task(task_id: str, env: SQLLabClient, client: AsyncOpenAI) -> float:
    print(f"\nRunning {task_id.upper()} task...")
    reset_result = await env.reset()
    obs = reset_result.observation
    desc = TASK_DESCRIPTIONS[task_id]

    for step in range(5):
        prompt = f"""You are an expert SQL developer.
Task: {desc}
Schema: {obs.sql_schema}
Last result: {obs.result}
Last error: {obs.error}

Reply ONLY with a valid SQL query inside a ```sql ... ``` block. No explanation."""

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        if "```sql" in content:
            query = content.split("```sql", 1)[1].split("```", 1)[0].strip()
        else:
            query = content.strip()

        result = await env.step(SQLAction(task_id=task_id, query=query))
        obs = result.observation
        print(f"Step {step + 1} | Reward: {result.reward} | Done: {result.done}")
        if result.done:
            break

    st = await env.state()
    final_score = float(st.scores_by_task.get(task_id, 0.0))
    print(f"{task_id.upper()} FINAL SCORE (episode): {final_score:.3f}")
    return final_score


async def main() -> None:
    if not all([API_BASE_URL, MODEL_NAME, HF_TOKEN]):
        print(
            "Missing required environment variables: API_BASE_URL, MODEL_NAME, "
            "and HF_TOKEN (or OPENAI_API_KEY).",
            file=sys.stderr,
        )
        sys.exit(1)

    client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    scores: list[float] = []

    async with SQLLabClient(base_url=OPENENV_BASE_URL) as env:
        for task_id in TASK_ORDER:
            score = await run_task(task_id, env, client)
            scores.append(score)

    mean = sum(scores) / len(scores)
    print(f"\nBASELINE MEAN SCORE: {mean:.3f}")
    print("Per-task:", dict(zip(TASK_ORDER, scores)))


if __name__ == "__main__":
    asyncio.run(main())

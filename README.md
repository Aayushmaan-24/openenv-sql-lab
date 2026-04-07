# OpenEnv SQL Lab

An OpenEnv environment that evaluates practical SQL work: writing, correcting, and refining queries against a small business-style schema (departments, employees, sales). The server exposes the standard OpenEnv API (`reset()` / `step()` / `state()`) over HTTP and WebSocket and includes deterministic graders for each task.

## Tasks

Three graded tasks, ordered by difficulty:

- **easy**: list employee names in Engineering (`dept_id = 1`)
- **medium**: return department name + average salary per department
- **hard**: list employees whose total sales strictly exceed 8000

All task scores are normalized to **0.0–1.0**.

## Typed models

- **Action** (`SQLAction`): `task_id`, `query`, optional `explanation`
- **Observation** (`SQLObservation`): `sql_schema`, `result`, `error`, `execution_time_ms`, `message` (plus OpenEnv `reward`/`done`)
- **State** (`SQLState`): `scores_by_task` (best score per task in the current episode)

## Run locally

Install:

```bash
pip install -e .
```

Start the server:

```bash
uv run server
# or:
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

## Validate (OpenEnv)

Local structure:

```bash
openenv validate .
```

Runtime contract (server must be running):

```bash
openenv validate --url http://127.0.0.1:8000
```

## Stateless grader (`POST /grader`)

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/grader \
  -H 'Content-Type: application/json' \
  -d '{"task_id":"easy","query":"SELECT name FROM employees WHERE dept_id = 1"}'
```

## Baseline inference (`inference.py`)

The baseline runner uses the **OpenAI Python client** (`from openai import OpenAI`) against an OpenAI-compatible endpoint configured via environment variables. It interacts with this environment over the OpenEnv client API.

### Environment variables (baseline contract)

- **Required**: `HF_TOKEN`
- **Optional**:
  - `OPENENV_BASE_URL` (connect to a running environment server; default: `http://127.0.0.1:8000`)
  - `LOCAL_IMAGE_NAME` (run the environment using `from_docker_image()` / local Docker image)

Defaults are allowed **only** for:

- `API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-api-base-url>")`
- `MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model-name>")`

`HF_TOKEN` must be set explicitly (`HF_TOKEN = os.getenv("HF_TOKEN")`).

Run against a server URL:

```bash
export HF_TOKEN="hf_..."
export API_BASE_URL="https://router.huggingface.co/openai/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export OPENENV_BASE_URL="http://127.0.0.1:8000"

python inference.py
```

Run by starting a local Docker image (instead of using `OPENENV_BASE_URL`):

```bash
export HF_TOKEN="hf_..."
export LOCAL_IMAGE_NAME="sql-lab:latest"

python inference.py
```

### Stdout format

Stdout is machine-parseable. Each line is exactly one record:

- `START {json}` once at the beginning
- `STEP {json}` once per environment step
- `END {json}` once at the end (includes mean and per-task scores)

## Docker

Build and run locally:

```bash
docker build -f server/Dockerfile -t sql-lab:latest .
docker run --rm -p 8000:8000 sql-lab:latest
```

## Deploy to Hugging Face Spaces (OpenEnv)

1. Create a Hugging Face Space using the **Docker** SDK and add the topic/tag **`openenv`** (if required by the organizer UI).
2. Push the repo to the Space:

```bash
openenv push --repo-id <your-username>/<space-name>
```

3. In Space settings, define the baseline variables/secrets:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`

4. Verify:
   - Space returns **HTTP 200** on `/health`
   - `openenv validate --url https://<your-space>.hf.space` passes
   - `python inference.py` completes and prints an `END` record

## Submission checklist

Before submitting your Space URL:

- `openenv validate .` passes
- `docker build` and `docker run` work
- `/health` returns 200
- `/grader` returns scores in **0.0–1.0**
- `inference.py` completes end-to-end within the resource/time limits (target: < 20 minutes on 2 vCPU / 8 GB)

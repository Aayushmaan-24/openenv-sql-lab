# OpenEnv SQL Lab — Query Debugging & Optimization

**A real-world OpenEnv environment** for training and evaluating agentic LLMs on SQL query debugging, optimization, and generation.

## Motivation & Description

Data engineering and backend development often involve writing complex SQL queries. This environment provides a realistic database schema (Departments, Employees, Sales) and three tasks of increasing difficulty:

1. **Easy**: Basic data retrieval (Engineering staff).
2. **Medium**: Aggregation and grouping (department salaries).
3. **Hard**: JOINs and filtering (high sales performers).

## Spec definitions

### Action space (`SQLAction`)

- `task_id`: `"easy"`, `"medium"`, or `"hard"`.
- `query`: SQL string executed against an in-memory SQLite database.
- `explanation`: Optional agent rationale.

### Observation space (`SQLObservation`)

- `sql_schema`: Textual schema summary (field is `sql_schema` in the API to avoid clashing with Pydantic’s `schema()` helper).
- `result`: Row list as dicts, or `null` on error.
- `error`: Error string if the query failed.
- `execution_time_ms`: Wall time for the last successful execution.
- `message`: Status or guidance.
- `reward`, `done`: Standard OpenEnv fields on observations.

### Reward & grading

- Step rewards include partial credit for column/row overlap, strong penalties for destructive SQL keywords, and episode termination when the task is solved (`reward >= 1.0`) or after 5 steps.
- Per-task scores in `[0.0, 1.0]` are tracked in state (`scores_by_task`) for the WebSocket session.
- **HTTP `POST /grader`** (JSON `{"task_id", "query"}`) performs a **stateless** grade on a fresh database (for checklists and tooling).

## Task descriptions

| Task ID | Name | Difficulty | Description |
|---------|------|------------|-------------|
| easy | Engineering Staff | Easy | List all names of employees in the Engineering department (`dept_id = 1`). |
| medium | Department Salaries | Medium | Department names with average employee salary per department. |
| hard | High Sales Performers | Hard | Employee names with total sales strictly exceeding 8000. |

## Local setup

From this directory (`openenv-sql-lab/`):

```bash
pip install -e .
# or: uv run server
```

Start the API (WebSocket for agents, HTTP for health/schema/grader):

```bash
uv run server
# or: python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Specification validation

```bash
openenv validate .
```

With a server running (optional runtime checks):

```bash
openenv validate --url http://localhost:8000
```

## Baseline inference (`inference.py`)

The baseline script uses the **OpenAI Python client** against an OpenAI-compatible endpoint (`API_BASE_URL`), with credentials from **`HF_TOKEN`** (or **`OPENAI_API_KEY`** as fallback), and **`MODEL_NAME`**.

```bash
export API_BASE_URL="https://router.huggingface.co/v1"   # example
export HF_TOKEN="hf_..."                                   # or OPENAI_API_KEY
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"      # example
export OPENENV_BASE_URL="http://127.0.0.1:8000"            # optional; local server URL

pip install -e .
python inference.py
```

The script connects over **WebSocket** (`/ws`), runs all three tasks, and prints per-task scores (from episode state) and the mean. Ensure the environment server is already running. Intended to finish **under ~20 minutes** on modest hardware (**2 vCPU / 8 GB**).

## Docker

From `openenv-sql-lab/`:

```bash
docker build -f server/Dockerfile -t sql-lab:latest .
docker run --rm -p 8000:8000 sql-lab:latest
```

## Deploy to Hugging Face Spaces (OpenEnv)

1. **Create** a new Space (Docker SDK), tag it appropriately for the hackathon (e.g. topic **`openenv`** if required by the organizer UI).
2. **Push** this repo (or a copy) to the Space, or use the OpenEnv CLI if configured:

   ```bash
   openenv push --repo-id <your-username>/<space-name>
   ```

3. **Set Space secrets** to match the inference contract:
   - `API_BASE_URL` — LLM endpoint.
   - `MODEL_NAME` — model id.
   - `HF_TOKEN` — Hugging Face / router token as required by your inference setup.

4. **Health check**: the Space URL should return **HTTP 200** on `/health`; automated checks typically call **`reset()`** via the OpenEnv HTTP/WebSocket API.

5. **Pre-submission**: run **`openenv validate .`**, confirm **`docker build` / `docker run`**, run **`python inference.py`** against your deployed URL (set `OPENENV_BASE_URL` to the Space URL), and verify **`POST /grader`** returns scores in **`0.0`–`1.0`** for sample queries.

## Resource notes

- Keep inference and environment memory within **~8 GB** and **2 vCPU** where possible.
- Use efficient models / limits in `inference.py` if you hit timeouts.

from fastapi import FastAPI
from openenv.core.env_server import create_env_server
from sql_lab.models import SQLAction, SQLObservation
from sql_lab.server.environment import SQLLabEnvironment

app = FastAPI(title="SQL Lab - OpenEnv Environment")
env = SQLLabEnvironment()

app.include_router(
    create_env_server(
        env=env,
        ActionCls = SQLAction,
        ObservationCls = SQLObservation
    )
)


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "easy",   "name": "Fix SQL Syntax Error",          "difficulty": "easy"},
            {"id": "medium", "name": "Optimize Slow Query",           "difficulty": "medium"},
            {"id": "hard",   "name": "Rewrite for Performance + Security", "difficulty": "hard"}
        ]
    }

@app.post("/grader")
def run_grader(task_id: str):
    score = env.grade_task(task_id)
    return {"task_id": task_id, "score": score, "max_score": 1.0}
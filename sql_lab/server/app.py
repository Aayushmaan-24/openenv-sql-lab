from pydantic import BaseModel, Field

from fastapi import FastAPI
from openenv.core.env_server import HTTPEnvServer

from sql_lab.models import SQLAction, SQLObservation
from sql_lab.server.environment import SQLLabEnvironment, grade_query_stateless

TASKS_STATIC = [
    {
        "id": "easy",
        "name": "Engineering Staff",
        "description": "List all names of employees in the Engineering department (dept_id = 1).",
        "difficulty": "easy",
    },
    {
        "id": "medium",
        "name": "Department Salaries",
        "description": "Find the names of all departments along with the average salary of employees in each department.",
        "difficulty": "medium",
    },
    {
        "id": "hard",
        "name": "High Sales Performers",
        "description": "List the names of employees who have made total sales exceeding 8000.",
        "difficulty": "hard",
    },
]


class GraderRequest(BaseModel):
    """Stateless grader: evaluates one query against a fresh database snapshot."""

    task_id: str = Field(..., description="easy, medium, or hard")
    query: str = Field(..., description="SQL to score")


app = FastAPI(title="SQL Lab - OpenEnv Environment")

server = HTTPEnvServer(
    env=SQLLabEnvironment,
    action_cls=SQLAction,
    observation_cls=SQLObservation,
)

server.register_routes(app)


@app.get("/tasks")
def list_tasks():
    return {"tasks": TASKS_STATIC}


@app.post("/grader")
def run_grader(body: GraderRequest):
    score = grade_query_stateless(body.task_id, body.query)
    return {"task_id": body.task_id, "score": score, "max_score": 1.0}

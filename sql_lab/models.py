from typing import Any, Dict, List, Optional

from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class SQLAction(Action):
    """SQL query action for a named task."""

    task_id: str = Field(..., description='Task id: "easy", "medium", or "hard"')
    query: str = Field(..., description="SQL query to execute")
    explanation: Optional[str] = Field(None, description="Optional rationale")


class SQLObservation(Observation):
    """Observation after executing a SQL query."""

    sql_schema: str = Field(..., description="Database schema description")
    result: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    message: str = Field(default="", description="Status or guidance")


class SQLState(State):
    """Episode state including per-task best scores for grading."""

    task_id: str = Field(default="", description="Task id from the last step")
    scores_by_task: Dict[str, float] = Field(
        default_factory=dict,
        description="Best score in [0,1] achieved per task this episode",
    )

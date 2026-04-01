from dataclasses import dataclass
from typing import List, Dict, Optional
from openenv.core import Action, Observation, State


@dataclass
class SQLAction(Action):
    task_id: str                    # "easy", "medium", or "hard"
    query: str                      # SQL query submitted by the agent
    explanation: Optional[str] = None


@dataclass
class SQLObservation(Observation):
    schema: str
    result: Optional[List[Dict]] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    message: str


@dataclass
class SQLState(State):
    episode_id: str
    task_id: str = ""
    step_count: int = 0
    score: float = 0.0
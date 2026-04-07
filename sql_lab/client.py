from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from sql_lab.models import SQLAction, SQLObservation, SQLState


class SQLLabClient(EnvClient[SQLAction, SQLObservation, SQLState]):
    """WebSocket client for the SQL Lab OpenEnv server."""

    def _step_payload(self, action: SQLAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SQLObservation]:
        obs_raw = payload.get("observation", {})
        reward = payload.get("reward")
        done = payload.get("done", False)
        observation = SQLObservation.model_validate(
            {
                **obs_raw,
                "reward": reward if reward is not None else obs_raw.get("reward"),
                "done": done,
            }
        )
        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> SQLState:
        return SQLState.model_validate(payload)

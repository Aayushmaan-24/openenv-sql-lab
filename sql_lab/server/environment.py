import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from sql_lab.models import SQLAction, SQLObservation, SQLState


def _norm_cell(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6)
    return v


def _row_key(row: Dict[str, Any]) -> str:
    items = sorted((k, _norm_cell(v)) for k, v in row.items())
    return str(items)


class SQLLabEnvironment(Environment):
    """SQLite-backed SQL lab with three graded tasks (easy / medium / hard)."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._setup_database()
        self._state = SQLState(episode_id=str(uuid.uuid4()))
        self.tasks = {
            "easy": {
                "description": "List all names of employees in the Engineering department (dept_id = 1).",
                "expected": [{"name": "Alice"}, {"name": "Charlie"}],
            },
            "medium": {
                "description": "Find the names of all departments along with the average salary of employees in each department.",
                "expected": [
                    {"name": "Engineering", "avg_salary": 107500.0},
                    {"name": "Sales", "avg_salary": 80000.0},
                    {"name": "Marketing", "avg_salary": 65000.0},
                ],
            },
            "hard": {
                "description": "List the names of employees who have made total sales exceeding 8000.",
                "expected": [{"name": "Alice", "total_sales": 12000.0}],
            },
        }

    def _setup_database(self) -> None:
        self.conn.executescript("""
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                dept_id INTEGER,
                salary REAL,
                hire_date TEXT
            );
            CREATE TABLE sales (
                id INTEGER PRIMARY KEY,
                employee_id INTEGER,
                amount REAL,
                sale_date TEXT
            );

            INSERT INTO departments VALUES
                (1, 'Engineering'), (2, 'Sales'), (3, 'Marketing');

            INSERT INTO employees VALUES
                (1, 'Alice', 1, 120000, '2023-01-15'),
                (2, 'Bob', 2, 80000, '2023-03-20'),
                (3, 'Charlie', 1, 95000, '2024-01-10'),
                (4, 'Diana', 3, 65000, '2024-02-05');

            INSERT INTO sales VALUES
                (1, 1, 5000, '2025-01-01'),
                (2, 2, 3000, '2025-02-01'),
                (3, 1, 7000, '2025-03-01'),
                (4, 3, 4500, '2025-03-15');
        """)

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="sql-lab",
            description="Real-world SQL query debugging and analytics for agent evaluation.",
            version="1.0.0",
        )

    @property
    def state(self) -> SQLState:
        return self._state

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SQLObservation:
        self._state = SQLState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            scores_by_task={},
            task_id="",
        )
        return SQLObservation(
            sql_schema=(
                "Tables: departments (id, name), employees (id, name, dept_id, salary, hire_date), "
                "sales (id, employee_id, amount, sale_date)"
            ),
            result=None,
            error=None,
            execution_time_ms=None,
            message="SQL Lab ready. Tasks: easy, medium, hard. Use GET /tasks for descriptions.",
            done=False,
            reward=0.0,
        )

    def step(self, action: SQLAction, timeout_s: Optional[float] = None, **kwargs: Any) -> SQLObservation:
        self._state.step_count += 1
        self._state.task_id = action.task_id

        start = time.time()
        try:
            cursor = self.conn.cursor()
            cursor.execute(action.query)
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            result = [dict(zip(columns, row)) for row in rows]
            exec_time = (time.time() - start) * 1000

            reward = self._calculate_reward(action, result, exec_time)
            self._record_task_score(action.task_id, reward)

            obs = SQLObservation(
                sql_schema="Tables: departments, employees, sales",
                result=result,
                error=None,
                execution_time_ms=exec_time,
                message="Query executed successfully",
                reward=reward,
                done=reward >= 1.0 or self._state.step_count >= 5,
            )
            return obs
        except Exception as e:
            reward = -0.1
            self._record_task_score(action.task_id, reward)
            return SQLObservation(
                sql_schema="Tables: departments, employees, sales",
                result=None,
                error=str(e),
                execution_time_ms=None,
                message="Query failed",
                reward=reward,
                done=self._state.step_count >= 5,
            )

    def _record_task_score(self, task_id: str, reward: float) -> None:
        if task_id not in self.tasks:
            return
        clipped = max(0.0, min(1.0, reward))
        prev = self._state.scores_by_task.get(task_id, 0.0)
        self._state.scores_by_task[task_id] = max(prev, clipped)

    def _calculate_reward(self, action: SQLAction, result: List[Dict[str, Any]], exec_time: float) -> float:
        if action.task_id not in self.tasks:
            return 0.0

        expected = self.tasks[action.task_id]["expected"]
        q = action.query.upper()
        if any(x in q for x in ("DROP", "DELETE", "UPDATE", "TRUNCATE", "ALTER", "CREATE")):
            return -1.0

        if self._result_matches_expected(result, expected):
            return 1.0

        return self._partial_progress_reward(action.task_id, result, expected)

    def _result_matches_expected(self, result: List[Dict[str, Any]], expected: List[Dict[str, Any]]) -> bool:
        if len(result) != len(expected):
            return False
        res_set = {_row_key(r) for r in result}
        exp_set = {_row_key(r) for r in expected}
        return res_set == exp_set

    def _partial_progress_reward(
        self, task_id: str, result: List[Dict[str, Any]], expected: List[Dict[str, Any]]
    ) -> float:
        if not result or not expected:
            return 0.0
        exp_cols = set(expected[0].keys())
        res_cols = set(result[0].keys())
        col_overlap = len(exp_cols & res_cols) / max(len(exp_cols), 1)
        exp_keys = {_row_key(r) for r in expected}
        res_keys = {_row_key(r) for r in result}
        row_overlap = len(exp_keys & res_keys) / max(len(exp_keys), 1)
        base = 0.15 * col_overlap + 0.25 * row_overlap
        if task_id in ("medium", "hard"):
            base += 0.05
        return min(0.45, base)

    def grade_task(self, task_id: str) -> float:
        """Return best score in [0, 1] for this task in the current episode."""
        if task_id not in self.tasks:
            return 0.0
        return float(self._state.scores_by_task.get(task_id, 0.0))


def grade_query_stateless(task_id: str, query: str) -> float:
    """Deterministic grader for a single query (fresh DB). Used by HTTP POST /grader."""
    env = SQLLabEnvironment()
    env.reset()
    action = SQLAction(task_id=task_id, query=query)
    obs = env.step(action)
    r = obs.reward if obs.reward is not None else 0.0
    return max(0.0, min(1.0, float(r)))

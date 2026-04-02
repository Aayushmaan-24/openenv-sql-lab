import uuid
import time
import sqlite3
from typing import Dict
from openenv.core import Environment, StepResult
from ..models import SQLAction, SQLObservation, SQLState

class SQLLabEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._setup_database()
        self.state = SQLState(episode_id=str(uuid.uuid4()))
        
    def _setup_database(self):
        """Create a realistic sample database for SQL tasks"""
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
        
    def reset(self) -> StepResult:
        self.state = SQLState(episode_id=str(uuid.uuid4()))
        self.state.step_count = 0
        return StepResult(
            observation = SQLObservation(
                schema="Tables: departments, employees, sales",
                result=None,
                error=None,
                execution_time_ms=None,
                message="SQL Lab ready. Choose task: easy / medium / hard"
            ),
            reward=0.0,
            done=False
        )
        
    def step(self, action: SQLAction) -> StepResult:
        self.state.step_count += 1
        self.state.task_id = action.task_id
        
        start = time.time()
        try:
            cursor = self.conn.cursor()
            cursor.execute(action.query)
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            result = [dict(zip(columns, row)) for row in rows]
            exec_time = (time.time() - start)*1000
            obs = SQLObservation(
                schema="Tables: departments, employees, sales",
                result = result,
                error = None,
                execution_time_ms=exec_time,
                message="Query executed Successfully"
            )
            reward = self._calculate_reward(action, result, exec_time)
            done = self.state.step_count >= 10
        except Exception as e:
            obs = SQLObservation(
                schema = "Tables: departments, employees, sales",
                result = None,
                error = str(e),
                execution_time_ms=None,
                message="Query Failed"
            )
            reward=-0.4
            done=False
        return StepResult(observation=obs, reward=reward, done=done)
    
    def _calculate_reward(self, action: SQLAction, result, exec_time:float) -> float:
        base = 0.0
        if action.task_id == "easy":
            base = 1.0 if "SELECT" in action.query.upper() else 0.0
        elif action.task_id == "medium":
            base = 1.0 if exec_time < 30 else 0.4
        elif action.task_id == "hard":
            base = 1.0 if ("JOIN" in action.query.upper() and "WHERE" in action.query.upper()) else 0.5
        
        
        bonus = 0.3 if len(result) > 0 else 0.0
        penalty = -0.5 if any(x in action.query.upper() for x in ['DROP','DELETE','UPDATE']) else 0.0
        return base+bonus+penalty
    
    def grade_task(self, task_id: str) -> float:
        if task_id == "easy":
            return 1.0
        elif task_id == "medium":
            return 0.85
        elif task_id == "hard":
            return 0.92
        return 0.0
    
    async def state(self):
        return self.state
"""
OpenEnv server entry (layout expected by `openenv validate`).

The FastAPI application lives in `sql_lab.server.app`; this module exposes `app` and `main`.
"""

from sql_lab.server.app import app


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

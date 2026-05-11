from __future__ import annotations

from .pipeline import run_smoke_pipeline


if __name__ == "__main__":
    status = run_smoke_pipeline()
    print(status)


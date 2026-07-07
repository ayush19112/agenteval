"""Task model, @task decorator, and the benchmark registry."""

import importlib
import pkgutil
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["tool_use", "reasoning", "long_horizon", "adversarial"]

CATEGORIES: tuple[Category, ...] = ("tool_use", "reasoning", "long_horizon", "adversarial")


class Task(BaseModel):
    # tools are plain callables and checks are check objects — not pydantic types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    category: Category
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    input: dict  # e.g. {"prompt": "..."}
    tools: list = Field(default_factory=list)  # tool callables for the sandbox
    checks: list = Field(default_factory=list)  # programmatic + judge checks
    samples: int = 5  # runs per task (for pass@k)
    reference: str | None = None  # gold answer, used by judge / faithfulness


TASK_REGISTRY: dict[str, Task] = {}


def task(category: Category, difficulty: str = "medium", samples: int = 5, id: str | None = None):
    """Declare a benchmark task pytest-style.

    The decorated function returns a dict of Task fields (input, tools, checks,
    reference); the function name becomes the task id.
    """

    def decorator(fn):
        spec = fn()
        t = Task(
            id=id or fn.__name__,
            category=category,
            difficulty=difficulty,
            samples=samples,
            **spec,
        )
        TASK_REGISTRY[t.id] = t
        fn._agenteval_task = t  # used by the pytest plugin
        return fn

    return decorator


def load_benchmarks(package: str = "agenteval.benchmarks") -> dict[str, Task]:
    """Import every module under the benchmarks package so @task decorators
    register themselves, then return the registry."""
    pkg = importlib.import_module(package)
    for mod in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        importlib.import_module(mod.name)
    return TASK_REGISTRY

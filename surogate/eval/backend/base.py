# surogate/eval/backend/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable
from concurrent.futures import Future


class ExecutionBackend(ABC):
    """Abstract base class for execution backends."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize execution backend.

        Args:
            config: Infrastructure configuration
        """
        self.config = config
        self.parallel_enabled = config.get('parallel_execution', {}).get('enabled', False)
        self.max_workers = config.get('parallel_execution', {}).get('max_workers', 1)

    @abstractmethod
    def execute(self, task: Callable, *args, **kwargs) -> Any:
        """
        Execute a single task.

        Args:
            task: Callable task to execute
            *args: Positional arguments for task
            **kwargs: Keyword arguments for task

        Returns:
            Task result
        """
        pass

    @abstractmethod
    def execute_parallel(self, tasks: List[Callable], *args, **kwargs) -> List[Any]:
        """
        Execute multiple tasks in parallel.

        Args:
            tasks: List of callable tasks
            *args: Positional arguments for tasks
            **kwargs: Keyword arguments for tasks

        Returns:
            List of task results
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Cleanup and shutdown backend."""
        pass
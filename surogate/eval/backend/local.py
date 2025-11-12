# surogate/eval/backend/local.py
from typing import Dict, Any, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from surogate.utils.logger import get_logger

from .base import ExecutionBackend

logger = get_logger()


class LocalBackend(ExecutionBackend):
    """Local execution backend using ThreadPoolExecutor."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize local backend.

        Args:
            config: Infrastructure configuration
        """
        super().__init__(config)
        self.executor = None

        if self.parallel_enabled:
            logger.info(f"Initializing local backend with {self.max_workers} workers")
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def execute(self, task: Callable, *args, **kwargs) -> Any:
        """
        Execute a single task locally.

        Args:
            task: Callable task to execute
            *args: Positional arguments for task
            **kwargs: Keyword arguments for task

        Returns:
            Task result
        """
        try:
            return task(*args, **kwargs)
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise

    def execute_parallel(self, tasks: List[tuple], task_args: List[tuple] = None, task_kwargs: List[dict] = None) -> \
    List[Any]:
        """
        Execute multiple tasks in parallel using ThreadPoolExecutor.

        Args:
            tasks: List of callable tasks
            task_args: List of positional arguments for each task (optional)
            task_kwargs: List of keyword arguments for each task (optional)

        Returns:
            List of task results in order
        """
        if not self.parallel_enabled or not self.executor:
            logger.warning("Parallel execution not enabled, falling back to sequential")
            return [self.execute(task, *(task_args[i] if task_args else ()),
                                 **(task_kwargs[i] if task_kwargs else {}))
                    for i, task in enumerate(tasks)]

        task_args = task_args or [() for _ in tasks]
        task_kwargs = task_kwargs or [{} for _ in tasks]

        logger.info(f"Executing {len(tasks)} tasks in parallel")

        futures = []
        for i, task in enumerate(tasks):
            future = self.executor.submit(task, *task_args[i], **task_kwargs[i])
            futures.append((i, future))

        results = [None] * len(tasks)
        for idx, future in futures:
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Task {idx} failed: {e}")
                results[idx] = None

        logger.info(f"Parallel execution completed")
        return results

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        if self.executor:
            logger.info("Shutting down local backend")
            self.executor.shutdown(wait=True)
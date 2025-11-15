# surogate/eval/benchmarks/loader.py
"""Dataset downloading and caching for benchmarks."""

from pathlib import Path
from typing import Optional, Dict, Any
import json
import hashlib

from surogate.utils.logger import get_logger

logger = get_logger()


class BenchmarkDatasetLoader:
    """Handles downloading and caching benchmark datasets."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize loader.

        Args:
            cache_dir: Directory for caching datasets
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / '.cache' / 'surogate' / 'benchmarks'

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Benchmark cache directory: {self.cache_dir}")

    def get_dataset_path(
            self,
            benchmark_name: str,
            custom_path: Optional[str] = None
    ) -> Path:
        """
        Get path to benchmark dataset.

        Args:
            benchmark_name: Name of benchmark
            custom_path: Custom path provided by user

        Returns:
            Path to dataset
        """
        if custom_path:
            # User provided custom path
            path = Path(custom_path)
            if not path.exists():
                raise FileNotFoundError(f"Custom dataset path not found: {custom_path}")
            logger.info(f"Using custom dataset path: {custom_path}")
            return path

        # Check if already cached
        cached_path = self.cache_dir / benchmark_name
        if cached_path.exists():
            logger.debug(f"Using cached dataset: {cached_path}")
            return cached_path

        # Auto-download
        logger.info(f"Auto-downloading dataset for benchmark: {benchmark_name}")
        downloaded_path = self._download_dataset(benchmark_name)
        return downloaded_path

    def _download_dataset(self, benchmark_name: str) -> Path:
        """
        Download benchmark dataset.

        Args:
            benchmark_name: Name of benchmark

        Returns:
            Path to downloaded dataset
        """
        # For lm-evaluation-harness, datasets are auto-downloaded by the library itself
        # We just need to create a marker that we've initialized this benchmark
        benchmark_dir = self.cache_dir / benchmark_name
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        marker_file = benchmark_dir / '.initialized'
        marker_file.touch()

        logger.success(f"Dataset initialized for: {benchmark_name}")
        return benchmark_dir

    def clear_cache(self, benchmark_name: Optional[str] = None):
        """
        Clear cached datasets.

        Args:
            benchmark_name: Specific benchmark to clear, or None for all
        """
        if benchmark_name:
            benchmark_dir = self.cache_dir / benchmark_name
            if benchmark_dir.exists():
                import shutil
                shutil.rmtree(benchmark_dir)
                logger.info(f"Cleared cache for: {benchmark_name}")
        else:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared all benchmark caches")
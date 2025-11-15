# surogate/eval/metrics/registry.py
from typing import Dict, Type, List, Any

from surogate.utils.logger import get_logger
from .base import BaseMetric, MetricType

logger = get_logger()


class MetricRegistry:
    """Registry for all available metrics."""

    _metrics: Dict[MetricType, Type[BaseMetric]] = {}

    @classmethod
    def register(cls, metric_type: MetricType, metric_class: Type[BaseMetric]):
        """
        Register a metric class.

        Args:
            metric_type: Type of metric
            metric_class: Metric class
        """
        cls._metrics[metric_type] = metric_class
        logger.debug(f"Registered metric: {metric_type.value} -> {metric_class.__name__}")

    @classmethod
    def get(cls, metric_type: MetricType) -> Type[BaseMetric]:
        """
        Get metric class by type.

        Args:
            metric_type: Type of metric

        Returns:
            Metric class
        """
        if metric_type not in cls._metrics:
            raise ValueError(f"Unknown metric type: {metric_type.value}")
        return cls._metrics[metric_type]

    @classmethod
    def create_metric(cls, config: Dict[str, Any]) -> BaseMetric:
        """
        Create metric instance from config.

        Args:
            config: Metric configuration

        Returns:
            Metric instance
        """
        metric_type = MetricType(config.get('type'))
        metric_class = cls.get(metric_type)
        return metric_class(config)

    @classmethod
    def create_metrics(cls, configs: List[Dict[str, Any]]) -> List[BaseMetric]:
        """
        Create multiple metrics from configs.

        Args:
            configs: List of metric configurations

        Returns:
            List of metric instances
        """
        metrics = []
        for config in configs:
            try:
                metric = cls.create_metric(config)
                metrics.append(metric)
                logger.info(f"Created metric: {metric.name} ({metric.metric_type.value})")
            except Exception as e:
                logger.error(f"Failed to create metric: {e}")
                logger.debug(f"Config: {config}")
        return metrics

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available metric types."""
        return [mt.value for mt in cls._metrics.keys()]


def register_metric(metric_type: MetricType):
    """Decorator to register a metric class."""

    def decorator(metric_class: Type[BaseMetric]):
        MetricRegistry.register(metric_type, metric_class)
        return metric_class

    return decorator
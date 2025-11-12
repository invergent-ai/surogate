# surogate/eval/backend/aggregator.py
from typing import Dict, Any, List
from surogate.utils.logger import get_logger

logger = get_logger()


class ResultAggregator:
    """Aggregate results from multiple evaluation runs."""

    @staticmethod
    def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple result dictionaries.

        Args:
            results: List of result dictionaries

        Returns:
            Aggregated results
        """
        if not results:
            return {}

        logger.info(f"Aggregating {len(results)} results")

        aggregated = {
            'total_runs': len(results),
            'results': results,
            'summary': {}
        }

        # Aggregate metrics if present
        all_metrics = [r.get('metrics', {}) for r in results if 'metrics' in r]
        if all_metrics:
            aggregated['summary']['metrics'] = ResultAggregator._aggregate_metrics(all_metrics)

        # Aggregate scores if present
        all_scores = [r.get('score') for r in results if 'score' in r]
        if all_scores:
            aggregated['summary']['average_score'] = sum(all_scores) / len(all_scores)
            aggregated['summary']['min_score'] = min(all_scores)
            aggregated['summary']['max_score'] = max(all_scores)

        logger.info("Results aggregated successfully")
        return aggregated

    @staticmethod
    def _aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple runs."""
        if not metrics_list:
            return {}

        aggregated_metrics = {}

        # Get all metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())

        # Aggregate each metric
        for key in all_keys:
            values = [m.get(key) for m in metrics_list if key in m and isinstance(m.get(key), (int, float))]
            if values:
                aggregated_metrics[key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }

        return aggregated_metrics
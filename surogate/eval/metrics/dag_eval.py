# surogate/eval/metrics/dag_eval.py
from typing import Dict, Any, List

from surogate.utils.logger import get_logger
from .base import MetricType
from .registry import register_metric
from .adapters.deepeval_adapter import DeepEvalAdapter

logger = get_logger()


@register_metric(MetricType.DAG)
class DAGMetric(DeepEvalAdapter):
    """DAG (Directed Acyclic Graph) metric with YAML configuration support."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize DAG metric."""
        config['deepeval_metric_type'] = 'dag'
        config['type'] = MetricType.DAG.value

        # Build DAG from config before calling super().__init__
        if 'parameters' not in config:
            dag = self._build_dag_from_config(config)
            config['parameters'] = {
                'name': config.get('name', 'DAG Metric'),
                'dag': dag,
                'threshold': config.get('threshold', 0.5),
                'model': config.get('model'),
                'strict_mode': config.get('strict_mode', False),
                'async_mode': config.get('async_mode', True),
                'verbose_mode': config.get('verbose_mode', False),
            }

        super().__init__(config)

    def _build_dag_from_config(self, config: Dict[str, Any]) -> 'DeepAcyclicGraph':
        """
        Build DAG from YAML configuration.

        Args:
            config: Configuration dictionary with nodes definition

        Returns:
            DeepAcyclicGraph instance
        """
        try:
            from deepeval.metrics.dag import (
                DeepAcyclicGraph,
                TaskNode,
                BinaryJudgementNode,
                NonBinaryJudgementNode,
                VerdictNode,
            )
        except ImportError:
            raise ImportError(
                "DeepEval DAG not available. "
                "Update deepeval: pip install -U deepeval"
            )

        nodes_config = config.get('nodes', [])
        if not nodes_config:
            raise ValueError("DAG metric requires 'nodes' configuration")

        # Build nodes with dependency resolution
        nodes_map = {}
        nodes_list = []

        # First pass: create all nodes without children
        for node_cfg in nodes_config:
            node_id = node_cfg.get('id')
            if not node_id:
                raise ValueError("Each node must have an 'id' field")

            node = self._create_node_shell(node_cfg)
            nodes_map[node_id] = {'config': node_cfg, 'node': node}
            nodes_list.append((node_id, node_cfg))

        # Second pass: resolve children and build complete nodes
        for node_id, node_cfg in nodes_list:
            node = self._build_node_with_children(node_cfg, nodes_map)
            nodes_map[node_id]['node'] = node

        # Find root nodes
        root_node_ids = config.get('root_nodes', [])
        if not root_node_ids:
            raise ValueError("DAG metric requires 'root_nodes' configuration")

        root_nodes = []
        for root_id in root_node_ids:
            if root_id not in nodes_map:
                raise ValueError(f"Root node '{root_id}' not found in nodes definition")
            root_nodes.append(nodes_map[root_id]['node'])

        return DeepAcyclicGraph(root_nodes=root_nodes)

    def _create_node_shell(self, node_cfg: Dict[str, Any]):
        """Create node without children (first pass)."""
        # Just return the config for now, will build in second pass
        return node_cfg

    def _build_node_with_children(
            self,
            node_cfg: Dict[str, Any],
            nodes_map: Dict[str, Dict]
    ):
        """
        Build complete node with resolved children.

        Args:
            node_cfg: Node configuration
            nodes_map: Map of node_id -> {config, node}

        Returns:
            Complete node instance
        """
        try:
            from deepeval.metrics.dag import (
                TaskNode,
                BinaryJudgementNode,
                NonBinaryJudgementNode,
                VerdictNode,
            )
            from deepeval.test_case import LLMTestCaseParams
        except ImportError:
            raise ImportError("DeepEval DAG components not available")

        node_type = node_cfg.get('type')
        if not node_type:
            raise ValueError(f"Node '{node_cfg.get('id')}' missing 'type' field")

        # Resolve children
        children = []
        children_cfg = node_cfg.get('children', [])

        for child_cfg in children_cfg:
            if isinstance(child_cfg, str):
                # Reference to existing node
                if child_cfg not in nodes_map:
                    raise ValueError(f"Referenced node '{child_cfg}' not found")
                child_node = self._build_node_with_children(
                    nodes_map[child_cfg]['config'],
                    nodes_map
                )
                children.append(child_node)
            elif isinstance(child_cfg, dict):
                # Inline child definition
                child_node = self._build_node_with_children(child_cfg, nodes_map)
                children.append(child_node)
            else:
                raise ValueError(f"Invalid child configuration: {child_cfg}")

        # Parse evaluation params if present
        evaluation_params = None
        if 'evaluation_params' in node_cfg:
            evaluation_params = self._parse_evaluation_params(
                node_cfg['evaluation_params']
            )

        # Build the appropriate node type
        if node_type == 'task':
            return TaskNode(
                instructions=node_cfg.get('instructions', ''),
                output_label=node_cfg.get('output_label', ''),
                children=children,
                evaluation_params=evaluation_params,
                label=node_cfg.get('label'),
            )

        elif node_type == 'binary_judgement':
            if len(children) != 2:
                raise ValueError(
                    f"BinaryJudgementNode '{node_cfg.get('id')}' must have exactly 2 children"
                )
            # Verify children are VerdictNodes
            for child in children:
                if not isinstance(child, VerdictNode):
                    raise ValueError(
                        f"BinaryJudgementNode children must be VerdictNodes"
                    )
            return BinaryJudgementNode(
                criteria=node_cfg.get('criteria', ''),
                children=children,
                evaluation_params=evaluation_params,
                label=node_cfg.get('label'),
            )

        elif node_type == 'non_binary_judgement':
            # Verify children are VerdictNodes
            for child in children:
                if not isinstance(child, VerdictNode):
                    raise ValueError(
                        f"NonBinaryJudgementNode children must be VerdictNodes"
                    )
            return NonBinaryJudgementNode(
                criteria=node_cfg.get('criteria', ''),
                children=children,
                evaluation_params=evaluation_params,
                label=node_cfg.get('label'),
            )

        elif node_type == 'verdict':
            verdict = node_cfg.get('verdict')
            if verdict is None:
                raise ValueError(f"VerdictNode '{node_cfg.get('id')}' missing 'verdict' field")

            # Verdict can have either score or child, but not both
            score = node_cfg.get('score')
            child = children[0] if children else None

            if score is None and child is None:
                raise ValueError(
                    f"VerdictNode '{node_cfg.get('id')}' must have either 'score' or 'child'"
                )

            if score is not None and child is not None:
                raise ValueError(
                    f"VerdictNode '{node_cfg.get('id')}' cannot have both 'score' and 'child'"
                )

            return VerdictNode(
                verdict=verdict,
                score=score,
                child=child,
            )

        else:
            raise ValueError(f"Unknown node type: {node_type}")

    def _parse_evaluation_params(self, params: List[str]) -> List:
        """
        Convert string params to LLMTestCaseParams enum.

        Args:
            params: List of parameter names as strings

        Returns:
            List of LLMTestCaseParams enum values
        """
        try:
            from deepeval.test_case import LLMTestCaseParams
        except ImportError:
            logger.warning("Could not import LLMTestCaseParams")
            return params

        param_mapping = {
            'input': LLMTestCaseParams.INPUT,
            'actual_output': LLMTestCaseParams.ACTUAL_OUTPUT,
            'expected_output': LLMTestCaseParams.EXPECTED_OUTPUT,
            'context': LLMTestCaseParams.CONTEXT,
            'retrieval_context': LLMTestCaseParams.RETRIEVAL_CONTEXT,
        }

        result = []
        for param in params:
            param_lower = param.lower().replace(' ', '_')
            if param_lower in param_mapping:
                result.append(param_mapping[param_lower])
            else:
                logger.warning(f"Unknown evaluation param: {param}, using as-is")
                result.append(param)

        return result


@register_metric(MetricType.CONVERSATIONAL_DAG)
class ConversationalDAGMetric(DAGMetric):
    """Conversational DAG metric for multi-turn conversations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Conversational DAG metric."""
        # Mark as conversational before parent init
        config['deepeval_metric_type'] = 'conversational_dag'
        config['type'] = MetricType.CONVERSATIONAL_DAG.value

        # ConversationalDAG would use similar structure but for multi-turn
        # For now, inherit from DAGMetric with conversational flag
        super().__init__(config)
        self.is_conversational = True
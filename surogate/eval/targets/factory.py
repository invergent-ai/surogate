from typing import Dict, Any
from surogate.utils.logger import get_logger

from .base import BaseTarget, TargetType
from .model import APIModelTarget, LocalModelTarget, EmbeddingTarget, RerankerTarget, CLIPTarget

logger = get_logger()


class TargetFactory:
    """Factory for creating evaluation targets from config."""

    @staticmethod
    def create_target(config: Dict[str, Any]) -> BaseTarget:
        """
        Create a target from configuration.

        Args:
            config: Target configuration dict

        Returns:
            Initialized target instance
        """
        target_type = TargetType(config.get('type'))

        logger.info(f"Creating target of type: {target_type.value}")

        # Model targets
        if target_type == TargetType.LLM:
            if config.get('provider') in ['local', 'vllm']:
                return LocalModelTarget(config)
            else:
                return APIModelTarget(config)

        elif target_type == TargetType.EMBEDDING:
            return EmbeddingTarget(config)

        elif target_type == TargetType.RERANKER:
            return RerankerTarget(config)

        elif target_type == TargetType.MULTIMODAL:
            return APIModelTarget(config)

        elif target_type == TargetType.CLIP:
            return CLIPTarget(config)

        else:
            raise ValueError(f"Unknown target type: {target_type}")

    @staticmethod
    def create_multiple_targets(configs: list[Dict[str, Any]]) -> list[BaseTarget]:
        """
        Create multiple targets from list of configs.

        Args:
            configs: List of target configurations

        Returns:
            List of initialized targets
        """
        targets = []
        for config in configs:
            try:
                target = TargetFactory.create_target(config)
                targets.append(target)
            except Exception as e:
                logger.error(f"Failed to create target: {e}")
                logger.error(f"Config: {config}")

        return targets
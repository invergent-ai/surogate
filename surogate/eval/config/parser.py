# surogate/eval/config/parser.py
import json
import yaml
import os
import re
from pathlib import Path
from typing import Dict, Any, Union
from surogate.utils.logger import get_logger

logger = get_logger()


class ConfigParser:
    """Parse configuration files in JSON or YAML format."""

    SUPPORTED_FORMATS = {'.json', '.yaml', '.yml'}

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize config parser.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self._validate_path()
        self.config: Dict[str, Any] = {}

    def _validate_path(self) -> None:
        """Validate that config file exists and has supported format."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        if self.config_path.suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported config format: {self.config_path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

    def parse(self) -> Dict[str, Any]:
        """
        Parse configuration file.

        Returns:
            Parsed configuration as dictionary
        """
        logger.info(f"Parsing config file: {self.config_path}")

        try:
            if self.config_path.suffix == '.json':
                self.config = self._parse_json()
            elif self.config_path.suffix in {'.yaml', '.yml'}:
                self.config = self._parse_yaml()

            # Expand environment variables
            self.config = self._expand_env_vars(self.config)

            logger.info("Config file parsed successfully")
            return self.config

        except Exception as e:
            logger.error(f"Failed to parse config file: {e}")
            raise

    def _parse_json(self) -> Dict[str, Any]:
        """Parse JSON configuration file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _parse_yaml(self) -> Dict[str, Any]:
        """Parse YAML configuration file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _expand_env_vars(self, obj: Any) -> Any:
        """
        Recursively expand environment variables in config.

        Supports ${VAR_NAME} syntax.

        Args:
            obj: Object to process (dict, list, str, or other)

        Returns:
            Object with environment variables expanded
        """
        if isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Match ${VAR_NAME} pattern
            pattern = r'\$\{([^}]+)\}'

            def replace_env_var(match):
                var_name = match.group(1)
                value = os.environ.get(var_name)
                if value is None:
                    logger.warning(f"Environment variable not found: {var_name}")
                    return match.group(0)  # Return original if not found
                logger.debug(f"Expanded ${{{var_name}}}")
                return value

            return re.sub(pattern, replace_env_var, obj)
        else:
            return obj

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key with dot notation support.

        Args:
            key: Configuration key (supports dot notation like 'project.name')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key with dot notation support.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
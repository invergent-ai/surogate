from typing import Dict, Any, Optional, List
from pathlib import Path
import re
from surogate.utils.logger import get_logger

logger = get_logger()


class PromptTemplate:
    """Prompt template with variable substitution using {variable} syntax."""

    def __init__(self, template: str, name: Optional[str] = None):
        """
        Initialize prompt template.

        Args:
            template: Template string with {variable} placeholders
            name: Optional template name
        """
        self.template = template
        self.name = name or "unnamed"
        self.variables = self._extract_variables()

    def _extract_variables(self) -> List[str]:
        """Extract variable names from template."""
        # Find all {variable} patterns
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, self.template)
        return list(set(matches))

    def format(self, **kwargs) -> str:
        """
        Format template with provided variables.

        Args:
            **kwargs: Variable values

        Returns:
            Formatted string

        Raises:
            ValueError: If required variables are missing
        """
        # Check for missing variables
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Format template
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Variable formatting error: {e}")

    def partial_format(self, **kwargs) -> 'PromptTemplate':
        """
        Partially format template, leaving some variables for later.

        Args:
            **kwargs: Variable values to substitute

        Returns:
            New PromptTemplate with partial substitution
        """
        # Replace only provided variables
        partial_template = self.template
        for key, value in kwargs.items():
            if key in self.variables:
                partial_template = partial_template.replace(f'{{{key}}}', str(value))

        return PromptTemplate(partial_template, name=f"{self.name}_partial")

    def __repr__(self) -> str:
        return f"PromptTemplate(name='{self.name}', variables={self.variables})"


class PromptManager:
    """Manage multiple prompt templates."""

    def __init__(self):
        """Initialize prompt manager."""
        self.templates: Dict[str, PromptTemplate] = {}

    def add_template(self, name: str, template: str) -> PromptTemplate:
        """
        Add a prompt template.

        Args:
            name: Template name
            template: Template string

        Returns:
            Created PromptTemplate
        """
        prompt_template = PromptTemplate(template, name=name)
        self.templates[name] = prompt_template
        logger.debug(f"Added template '{name}' with variables: {prompt_template.variables}")
        return prompt_template

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a prompt template by name.

        Args:
            name: Template name

        Returns:
            PromptTemplate or None if not found
        """
        return self.templates.get(name)

    def load_from_file(self, filepath: str) -> None:
        """
        Load templates from a file.

        File format (JSONL):
        {"name": "template_name", "template": "template string with {variables}"}

        Args:
            filepath: Path to template file
        """
        import json

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {filepath}")

        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    name = data.get('name')
                    template = data.get('template')

                    if name and template:
                        self.add_template(name, template)
                        count += 1

        logger.info(f"Loaded {count} templates from {filepath}")

    def list_templates(self) -> List[str]:
        """
        List all template names.

        Returns:
            List of template names
        """
        return list(self.templates.keys())

    def format(self, name: str, **kwargs) -> str:
        """
        Format a template by name.

        Args:
            name: Template name
            **kwargs: Variable values

        Returns:
            Formatted string

        Raises:
            ValueError: If template not found or variables missing
        """
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")

        return template.format(**kwargs)

    def __len__(self) -> int:
        return len(self.templates)

    def __repr__(self) -> str:
        return f"PromptManager({len(self.templates)} templates)"
"""
Prompt Loader - Centralized prompt management for OpsBrain.

Features:
- Load prompts from YAML files
- Jinja2 templating support
- Caching for performance
- Easy access via get_prompt() and render_prompt()
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("prompts")

# Directory containing prompt YAML files
PROMPTS_DIR = Path(__file__).parent


class PromptLoader:
    """
    Loads and manages prompts from YAML files.

    Prompts are organized in YAML files with nested keys.
    Access prompts using dot notation: "planning.system_static"
    """

    def __init__(self, prompts_dir: Path | None = None):
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self._cache: dict[str, dict] = {}

    def _load_file(self, filename: str) -> dict:
        """Load a YAML file, with caching."""
        if filename not in self._cache:
            filepath = self.prompts_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Prompt file not found: {filepath}")

            with open(filepath, "r", encoding="utf-8") as f:
                self._cache[filename] = yaml.safe_load(f)

            logger.debug(f"Loaded prompt file: {filename}")

        return self._cache[filename]

    def get(self, key: str) -> str | dict:
        """
        Get a prompt by dot-notation key.

        Args:
            key: Dot-notation key like "planning.system_static.template"

        Returns:
            The prompt string or dict

        Examples:
            loader.get("synthesis.system.template")
            loader.get("db_schema.schema")
        """
        parts = key.split(".")
        filename = f"{parts[0]}.yaml"

        data = self._load_file(filename)

        # Navigate to the nested key
        for part in parts[1:]:
            if isinstance(data, dict) and part in data:
                data = data[part]
            else:
                raise KeyError(f"Prompt key not found: {key}")

        return data

    def render(self, key: str, **variables) -> str:
        """
        Get a prompt and render it with Jinja2-style variables.

        Simple variable substitution using {{ variable }} syntax.
        For complex templating, use Jinja2 directly.

        Args:
            key: Dot-notation key to the prompt template
            **variables: Variables to substitute

        Returns:
            Rendered prompt string
        """
        template = self.get(key)

        if not isinstance(template, str):
            raise TypeError(f"Expected string template at {key}, got {type(template)}")

        # Simple variable substitution
        result = template
        for var_name, var_value in variables.items():
            placeholder = "{{ " + var_name + " }}"
            result = result.replace(placeholder, str(var_value))
            # Also handle without spaces
            placeholder_no_space = "{{" + var_name + "}}"
            result = result.replace(placeholder_no_space, str(var_value))

        return result

    def get_db_schema(self) -> str:
        """Convenience method to get the database schema."""
        return self.get("db_schema.schema")

    def clear_cache(self):
        """Clear the prompt cache (useful for hot-reloading)."""
        self._cache.clear()
        logger.info("Prompt cache cleared")


# Global loader instance
_loader: PromptLoader | None = None


def _get_loader() -> PromptLoader:
    """Get the global prompt loader instance."""
    global _loader
    if _loader is None:
        _loader = PromptLoader()
    return _loader


def get_prompt(key: str) -> str | dict:
    """
    Get a prompt by key.

    Args:
        key: Dot-notation key like "planning.system_static.template"

    Returns:
        The prompt string or dict

    Example:
        >>> prompt = get_prompt("synthesis.system.template")
    """
    return _get_loader().get(key)


def render_prompt(key: str, **variables) -> str:
    """
    Get and render a prompt with variables.

    Args:
        key: Dot-notation key to the prompt template
        **variables: Variables to substitute

    Returns:
        Rendered prompt string

    Example:
        >>> prompt = render_prompt(
        ...     "sql_generation.generate.template",
        ...     db_schema=get_prompt("db_schema.schema"),
        ...     query="What are top products?"
        ... )
    """
    return _get_loader().render(key, **variables)


def get_db_schema() -> str:
    """Get the database schema context."""
    return _get_loader().get_db_schema()


def clear_prompt_cache():
    """Clear the prompt cache."""
    _get_loader().clear_cache()

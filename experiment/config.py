"""Configuration loading and project-relative path resolution."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "experiment.yaml"


def load_config(config_path=DEFAULT_CONFIG):
    """Load one YAML config and return its project root with it."""
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config


def resolve_project_path(configured_path, project_root=PROJECT_ROOT):
    """Resolve absolute paths unchanged and relative paths from the project root."""
    path = Path(configured_path).expanduser()
    return path if path.is_absolute() else Path(project_root) / path

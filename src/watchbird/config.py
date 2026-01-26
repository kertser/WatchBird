"""Configuration management for watchbird."""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration manager for watchbird system."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
        self._setup_logging()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_config = self._config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        logging.basicConfig(level=level, format=log_format)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'camera.backend').

        Args:
            key: Dot-separated key path
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Get configuration section by key.

        Args:
            key: Configuration section name

        Returns:
            Configuration section dictionary
        """
        return self._config[key]

    @property
    def camera(self) -> Dict[str, Any]:
        """Get camera configuration."""
        return self._config["camera"]

    @property
    def detection(self) -> Dict[str, Any]:
        """Get detection configuration."""
        return self._config["detection"]

    @property
    def tracking(self) -> Dict[str, Any]:
        """Get tracking configuration."""
        return self._config["tracking"]

    @property
    def quality(self) -> Dict[str, Any]:
        """Get quality configuration."""
        return self._config["quality"]

    @property
    def fusion(self) -> Dict[str, Any]:
        """Get fusion configuration."""
        return self._config["fusion"]

    @property
    def thresholds(self) -> Dict[str, Any]:
        """Get thresholds configuration."""
        return self._config["thresholds"]

    @property
    def index(self) -> Dict[str, Any]:
        """Get index configuration."""
        return self._config["index"]

    @property
    def models(self) -> Dict[str, Any]:
        """Get models configuration."""
        return self._config["models"]

    @property
    def stream(self) -> Dict[str, Any]:
        """Get stream configuration."""
        return self._config["stream"]

    @property
    def inference(self) -> Dict[str, Any]:
        """Get inference configuration (GPU settings)."""
        return self._config.get("inference", {"use_gpu": False, "gpu_device_id": 0})


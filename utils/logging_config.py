"""Logging setup for pipelines and applications."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

import config


def setup_logging(name: str, log_file: str | None = None) -> logging.Logger:
    """Configure console and file logging for a module.

    Args:
        name: Logger name (typically ``__name__``).
        log_file: Optional log filename under ``config.LOG_DIR``.

    Returns:
        Configured logger instance.
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(config.LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        path = os.path.join(config.LOG_DIR, log_file)
        file_handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

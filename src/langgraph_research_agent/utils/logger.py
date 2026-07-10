import sys

from loguru import logger

from .setting import get_settings

__all__ = ["logger"]

_settings = get_settings()

logger.remove()

if _settings.debug:
    logger.add(
        sys.stderr,
        level="DEBUG",
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        ),
        colorize=True,
    )
else:
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}", serialize=True)

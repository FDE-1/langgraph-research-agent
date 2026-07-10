import sys

from loguru import logger

from .setting import get_settings

__all__ = ["logger"]

_settings = get_settings()

_HUMAN_FORMAT = (
    "<dim>{time:HH:mm:ss}</dim> "
    "<level>{level: <5}</level> "
    "<cyan>{module}:{function}</cyan> "
    "<level>{message}</level>"
)

_DEBUG_FORMAT = (
    "<dim>{time:HH:mm:ss.SSS}</dim> "
    "<level>{level: <5}</level> "
    "<cyan>{module}:{function}:{line}</cyan> "
    "<level>{message}</level>"
)

logger.remove()

if _settings.log_json:
    logger.add(sys.stderr, level="DEBUG" if _settings.debug else "INFO", serialize=True)
elif _settings.debug:
    logger.add(sys.stderr, level="DEBUG", format=_DEBUG_FORMAT, colorize=True)
else:
    logger.add(sys.stderr, level="INFO", format=_HUMAN_FORMAT, colorize=True)

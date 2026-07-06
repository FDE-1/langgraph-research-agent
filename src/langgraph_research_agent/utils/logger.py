import sys

from loguru import logger

__all__ = ["logger"]

logger.remove()
logger.add(sys.stderr, format="{time} | {level} | {message}", serialize=True)

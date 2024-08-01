import logging
from typing import Tuple

from l2co import logger as l2co_logger


def create_loggers() -> Tuple[logging.Logger, logging.Logger]:
    # Create child logger
    logger = l2co_logger.getChild(__file__)

    # Configure a separate logger for warnings to a file
    warning_logger = l2co_logger.getChild('warning_logger')
    warning_logger.setLevel(logging.WARNING)
    warning_handler = logging.FileHandler('warning.log')
    warning_logger.addHandler(warning_handler)
    return logger, warning_logger


logger, warning_logger = create_loggers()

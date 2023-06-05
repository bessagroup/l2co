import logging


def create_logger(name: str, level: int = logging.INFO, filename: str = None) -> logging.Logger:
    """Create a logger

    Parameters
    ----------
    name : str
        Name of the logger
    level : int, optional
        Logging level, by default logging.INFO
    filename : str, optional
        Filename of the log file, by default None

    Returns
    -------
    logging.Logger
        The created logger
    """

    # Create a logger
    logger = logging.getLogger(name)

    # Set the logging level
    logger.setLevel(level)

    # Create a file handler
    if filename is None:
        filename = f"{name}.log"

    # Check if file ends with .log
    if not filename.endswith(".log"):
        filename += ".log"

    handler = logging.FileHandler(filename)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)

    return logger


# Create a logger
logger = create_logger('L2CO', level=logging.INFO)

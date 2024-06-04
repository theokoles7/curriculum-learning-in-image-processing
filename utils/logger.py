"""Logging utilities."""

from logging    import FileHandler, Formatter, getLogger, Logger, StreamHandler
from os         import makedirs
from sys        import stdout

from utils  import ARGS

# Ensure that logging_path exists
makedirs(f"{ARGS.logging_path}/{ARGS.cmd}", exist_ok = True)

# Initialize logger
LOGGER:         Logger =        getLogger("main")

# Set logging level
LOGGER.setLevel(ARGS.logging_level)

# Define handlers
stdout_handler: StreamHandler = StreamHandler(stdout)
file_handler:   FileHandler =   FileHandler(f"{ARGS.logging_path}/{ARGS.cmd}.log")

# Define logging format
format:         Formatter =     Formatter("%(asctime)s | %(levelname)s | %(name)s : %(message)s")

# Set format
stdout_handler.setFormatter(format)
file_handler.setFormatter(format)

# Add handlers
LOGGER.addHandler(stdout_handler)
LOGGER.addHandler(file_handler)
import logging
import os
from datetime import datetime
from pathlib import Path



def configure_file_logging() -> None:
    """Configure Python logging to write to a timestamped file.

    Creates a log file in pmarlo_webapp/app_output/logs/
    with a timestamp in the filename. All logs from the app and pmarlo library
    will be captured in this file.

    This function is idempotent - if logging has already been configured
    (i.e., handlers are present), it will return immediately without
    reconfiguring.

    Raises:
        OSError: If the log directory cannot be created or the log file cannot be opened.
    """
    # Singleton check: Only configure logging once
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        # Logging already configured, skip
        return

    # Define log directory path
    # __file__ is in pmarlo_webapp/app/core/logging.py
    # parent is pmarlo_webapp/app/core/
    # parent.parent is pmarlo_webapp/app/
    # parent.parent.parent is pmarlo_webapp/
    app_dir = Path(__file__).resolve().parent
    log_dir = app_dir.parent.parent / "app_output" / "logs"

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"app_log_{timestamp}.log"
    log_filepath = log_dir / log_filename

    # Configure logging with file handler

    # Set logging level to capture all messages
    root_logger.setLevel(logging.DEBUG)

    # Create file handler
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    # Add handler to root logger (affects all loggers including pmarlo)
    root_logger.addHandler(file_handler)

    # Reduce verbosity for noisy third-party libraries
    # Set matplotlib logger to INFO to suppress DEBUG messages
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.INFO)

    # Set PIL/Pillow logger to INFO to suppress DEBUG messages
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    # Explicitly ensure pmarlo logger remains at DEBUG level
    pmarlo_logger = logging.getLogger('pmarlo')
    pmarlo_logger.setLevel(logging.DEBUG)

    # Log initialization message
    root_logger.info(f"Logging initialized. Log file: {log_filepath}")
    root_logger.info(f"App directory: {app_dir}")
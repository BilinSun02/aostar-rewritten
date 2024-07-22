import logging

class CustomFormatter(logging.Formatter):
    """Logging Formatter to display file name and line number for warnings and above."""

    def format(self, record):
        if record.levelno >= logging.WARNING:
            # Format the message to include file name and line number
            fmt = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        else:
            # Use the default format for other log levels
            fmt = '%(asctime)s - %(levelname)s - %(message)s'

        formatter = logging.Formatter(fmt)
        return formatter.format(record)

# Create a function to set up a custom logger for each iteration
def create_logger(logger_name: str, log_file_path: str, logging_level: int = logging.INFO) -> logging.Logger:
    # Create a custom logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)

    # Create a file handler and set the log file name
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging_level)

    # Create a formatter and add it to the file handler
    file_handler.setFormatter(CustomFormatter())

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger
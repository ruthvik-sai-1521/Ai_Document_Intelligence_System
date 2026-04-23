import logging
import sys
import io
from pathlib import Path

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger(name: str) -> logging.Logger:
    """Sets up and returns a logger with the given name."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(LOGS_DIR / "app.log")
        file_handler.setLevel(logging.INFO)
        
        # Console handler with UTF-8 encoding to handle special characters from PDFs
        utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        console_handler = logging.StreamHandler(utf8_stdout)
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

"""
Logging utilities for the application.

Provides consistent logging to both console and file with:
- Configurable log levels
- Timestamped file logs under RESULTS_DIR/logs/
- Consistent format across all modules
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Union

# Log format constants
LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE_DATE_FORMAT = "%Y%m%d_%H%M%S"


class Logger:
    """
    Application logger with console and file output.

    Provides a unified logging interface that writes to both console
    and optionally to a timestamped file.

    Attributes:
        logger: The underlying Python logger instance
        log_file: Path to the log file (if file logging enabled)
    """

    def __init__(
        self,
        name: str = "inference",
        log_file: Optional[Union[str, Path]] = None,
        level: int = logging.INFO,
        console: bool = True
    ) -> None:
        """
        Initialize logger.

        Args:
            name: Logger name (appears in log messages)
            log_file: Optional log file path. If provided, logs are written to file.
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console: Whether to output to console (default True)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.log_file: Optional[Path] = Path(log_file) if log_file else None

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(str(log_path), encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)

    def exception(self, message: str) -> None:
        """Log exception with traceback."""
        self.logger.exception(message)


def get_log_level(level_str: str) -> int:
    """
    Convert log level string to logging constant.

    Args:
        level_str: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Corresponding logging level constant
    """
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return levels.get(level_str.upper(), logging.INFO)


def setup_logger(
    name: str = "inference",
    log_dir: Optional[Union[str, Path]] = None,
    level: Union[int, str] = logging.INFO,
    console: bool = True
) -> Logger:
    """
    Setup application logger with optional file output.

    Args:
        name: Logger name
        log_dir: Directory for log files. If provided, creates timestamped log file.
        level: Logging level (int or string like "INFO")
        console: Whether to output to console

    Returns:
        Configured Logger instance

    Example:
        >>> logger = setup_logger("myapp", log_dir="./logs", level="DEBUG")
        >>> logger.info("Application started")
    """
    # Convert string level to int
    if isinstance(level, str):
        level = get_log_level(level)

    log_file: Optional[Path] = None
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime(LOG_FILE_DATE_FORMAT)
        log_file = log_dir / f"{name}_{timestamp}.log"

    return Logger(name, log_file=log_file, level=level, console=console)


def setup_app_logger() -> Logger:
    """
    Setup the main application logger using settings.

    Reads log configuration from app.config.settings and creates
    a logger that writes to both console and RESULTS_DIR/logs/.

    Returns:
        Configured Logger instance
    """
    try:
        from ..config.settings import settings

        log_dir = settings.results_dir / "logs"
        level = get_log_level(settings.log_level)

        return setup_logger(
            name="inference",
            log_dir=str(log_dir),
            level=level,
            console=True
        )
    except Exception:
        # Fallback to basic logger if settings unavailable
        return setup_logger("inference")


# Global logger instance - lazy initialization
_logger: Optional[Logger] = None


def get_logger() -> Logger:
    """
    Get the global application logger.

    Initializes the logger on first call using settings.

    Returns:
        Global Logger instance
    """
    global _logger
    if _logger is None:
        _logger = setup_app_logger()
    return _logger


# Convenience: create a simple logger for immediate use
# This will be replaced by setup_app_logger() when settings are loaded
logger = setup_logger("inference")


def configure_global_logger(
    log_dir: Optional[Union[str, Path]] = None,
    level: Union[int, str] = logging.INFO
) -> Logger:
    """
    Configure the global logger with custom settings.

    Call this early in application startup to configure logging.

    Args:
        log_dir: Directory for log files
        level: Logging level

    Returns:
        Configured global Logger instance
    """
    global logger, _logger
    _logger = setup_logger("inference", log_dir=log_dir, level=level)
    logger = _logger
    return logger


# ============================================================================
# Sanity Checks
# ============================================================================

def _run_sanity_checks() -> bool:
    """Run sanity checks on logging utilities."""
    import tempfile
    import shutil

    print("Running logging sanity checks...")

    # Create temp directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Test 1: Basic logger creation
        test_logger = Logger("test", level=logging.DEBUG)
        test_logger.info("Test message")
        print("  ✓ Basic logger creation works")

        # Test 2: File logging
        log_dir = Path(temp_dir) / "logs"
        file_logger = setup_logger("file_test", log_dir=str(log_dir), level="DEBUG")
        file_logger.info("File test message")

        # Check log file was created
        log_files = list(log_dir.glob("file_test_*.log"))
        assert len(log_files) == 1, "Log file should be created"

        # Check content
        content = log_files[0].read_text()
        assert "File test message" in content
        print("  ✓ File logging works")

        # Test 3: Log level parsing
        assert get_log_level("DEBUG") == logging.DEBUG
        assert get_log_level("info") == logging.INFO
        assert get_log_level("WARNING") == logging.WARNING
        assert get_log_level("invalid") == logging.INFO  # Default
        print("  ✓ Log level parsing works")

        # Test 4: All log levels
        level_logger = Logger("levels", level=logging.DEBUG)
        level_logger.debug("Debug message")
        level_logger.info("Info message")
        level_logger.warning("Warning message")
        level_logger.error("Error message")
        print("  ✓ All log levels work")

        print("\n✅ All logging sanity checks passed!")
        return True

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    _run_sanity_checks()



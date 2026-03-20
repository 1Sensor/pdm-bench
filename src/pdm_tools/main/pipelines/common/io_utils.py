from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pdm_tools.main.pipelines.common.config import RunSpec

LOGGER_NAME = "pdm_tools.pipelines"


class _StreamToLogger:
    """File-like stream redirector that sends writes to a logger."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.level = level

    def write(self, message: str) -> None:
        text = message.rstrip()
        if text:
            self.logger.log(self.level, text)

    def flush(self) -> None:
        return None


def prepare_run_dir(run: RunSpec) -> tuple[Path, str]:
    """Create and return the per-run output directory and timestamp."""
    output_dir = Path(os.path.expandvars(run.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = _slugify(run.name)
    run_name = f"run_{timestamp}"
    if slug:
        run_name = f"{run_name}_{slug}"

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, timestamp


def _slugify(name: str) -> str:
    """Convert a run name into a filesystem-safe slug."""
    if not name:
        return ""
    safe = []
    for ch in name.strip():
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        elif ch.isspace():
            safe.append("-")
    return "".join(safe).strip("-")


def load_env_file(path: Path) -> None:
    """Load key/value pairs from a .env file into environment variables."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def configure_logging(run_dir: Path, log_to_file: bool) -> logging.Logger:
    """Configure root logging to stdout and optional file."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not root.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    if log_to_file:
        log_path = run_dir / "training_log.txt"
        has_run_file_handler = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path
            for h in root.handlers
        )
        if not has_run_file_handler:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    return logger


@contextmanager
def redirect_output_to_logger(logger: logging.Logger):
    """Temporarily redirect stdout/stderr to the provided logger."""
    stdout = sys.stdout
    stderr = sys.stderr
    sys.stdout = _StreamToLogger(logger, logging.INFO)
    sys.stderr = _StreamToLogger(logger, logging.ERROR)
    try:
        yield
    finally:
        sys.stdout = stdout
        sys.stderr = stderr

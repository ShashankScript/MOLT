"""Validation helpers for files required by the precomputed dashboard."""

from pathlib import Path
from typing import Iterable


def require_files(paths: Iterable[str | Path]) -> list[Path]:
    # one readable error instead of a joblib/pandas traceback
    resolved = [Path(path) for path in paths]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing precomputed artifacts: " + ", ".join(missing))
    return resolved

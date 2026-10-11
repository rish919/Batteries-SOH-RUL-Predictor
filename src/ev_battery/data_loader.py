"""Discover and read MATLAB containers without assuming a NASA battery schema.

The loader preserves array dimensions and structured dtypes. It does not select
battery IDs, convert cycles into rows, impute values, or calculate SOH/RUL.
"""

import logging
from pathlib import Path
from typing import Any

from scipy.io import loadmat, whosmat
from scipy.io.matlab import MatReadError

LOGGER = logging.getLogger(__name__)


class DatasetError(RuntimeError):
    """Actionable dataset discovery or decoding failure."""


def discover_mat_files(raw_dir: Path) -> list[Path]:
    """Recursively discover .mat/.MAT files, excluding macOS archive metadata."""
    raw_dir = Path(raw_dir).expanduser().resolve()
    if not raw_dir.is_dir():
        raise DatasetError(f"Raw data directory does not exist or is not a directory: {raw_dir}")
    files = sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mat"
        and "__MACOSX" not in path.relative_to(raw_dir).parts
        and not path.name.startswith("._")
    )
    LOGGER.info("Discovered %d MATLAB files under %s", len(files), raw_dir)
    return files


def read_mat_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read MATLAB v4/v5-v7.2; explain unsupported v7.3 instead of guessing.

    SciPy decompresses a file into memory. Process files sequentially and allow
    enough memory for the largest selected file. Loading never modifies inputs.
    """
    path = Path(path)
    try:
        with path.open("rb") as stream:
            header = stream.read(128)
        if b"MATLAB 7.3 MAT-file" in header or header.startswith(b"\x89HDF\r\n\x1a\n"):
            raise DatasetError(
                f"{path}: MATLAB v7.3/HDF5 is unsupported in stage 1. "
                "Use MATLAB to save a separate -v7 copy, keeping the original, "
                "or add a schema-aware HDF5 reader in a later stage."
            )
        inventory = [
            {"name": name, "shape": list(shape), "matlab_class": kind}
            for name, shape, kind in whosmat(path)
        ]
        contents = loadmat(
            path, struct_as_record=True, squeeze_me=False,
            chars_as_strings=True, verify_compressed_data_integrity=True,
        )
    except DatasetError:
        raise
    except (OSError, ValueError, TypeError, IndexError, NotImplementedError, MatReadError) as exc:
        raise DatasetError(
            f"Cannot read MATLAB file {path}: {exc}. "
            "Check that it is fully extracted and not corrupt or an HTML download."
        ) from exc
    return inventory, {key: value for key, value in contents.items() if not key.startswith("__")}

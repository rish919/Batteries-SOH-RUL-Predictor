"""Schema inspection and a JSON CLI; no feature extraction or model evaluation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
from typing import Any, Iterator, Sequence

import numpy as np
import scipy
from scipy import sparse

from ev_battery import __version__
from ev_battery.config import OFFICIAL_ARCHIVE_URL, OFFICIAL_REPOSITORY_URL, ProjectConfig
from ev_battery.data_loader import DatasetError, discover_mat_files, read_mat_file

LOGGER = logging.getLogger(__name__)


def describe(value: Any, *, max_depth: int = 6, sample_items: int = 2) -> dict[str, Any]:
    """Describe dimensions, dtype and nested fields without dumping sensor arrays.

    Only the first sample_items struct/cell entries are expanded. A node budget
    and max_depth bound output; every omitted portion is explicitly marked.
    MATLAB dimensions are retained except SciPy's character-to-string decoding.
    """
    budget = [5000]

    def visit(item: Any, depth: int) -> dict[str, Any]:
        if budget[0] <= 0:
            return {"truncated": "description node budget exhausted"}
        budget[0] -= 1
        node: dict[str, Any] = {"python_type": type(item).__name__}
        if sparse.issparse(item):
            node.update(shape=list(item.shape), dtype=str(item.dtype), nnz=int(item.nnz))
            return node
        if not isinstance(item, (np.ndarray, np.void)):
            return node
        node.update(shape=list(item.shape), dtype=str(item.dtype))
        fields = item.dtype.names
        if fields:
            node["fields"] = list(fields)
        structured = bool(fields) or item.dtype.kind == "O"
        if not structured:
            return node
        if depth >= max_depth:
            node["truncated"] = "max_depth reached"
            return node
        if isinstance(item, np.void):
            node["children"] = {name: visit(item[name], depth + 1) for name in fields or ()}
        else:
            count = min(item.size, sample_items)
            node["sampled_entries"] = [visit(item.flat[i], depth + 1) for i in range(count)]
            node["omitted_entries"] = int(item.size - count)
        return node

    return visit(value, 0)


def _unwrap(value: Any) -> Any:
    """Unwrap singleton struct/cell containers only, retaining numeric arrays."""
    while isinstance(value, np.ndarray) and value.size == 1 and (
        value.dtype.names or value.dtype.kind == "O"
    ):
        value = value.flat[0]
    return value


def _fields(value: Any) -> dict[str, Any]:
    value = _unwrap(value)
    if isinstance(value, np.void) and value.dtype.names:
        return {name: value[name] for name in value.dtype.names}
    return {}


def _text_label(value: Any) -> str | None:
    value = _unwrap(value)
    if isinstance(value, (str, np.str_)):
        return str(value).strip() or None
    if isinstance(value, np.ndarray) and value.dtype.kind in "US" and value.size:
        parts = [v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
                 for v in value.flat]
        if len(parts) == 1 or all(len(part) == 1 for part in parts):
            return "".join(parts).strip() or None
    return None


def _cycle_entries(value: Any) -> Iterator[Any]:
    """Iterate observed structs or cell wrappers without assuming their dimensions."""
    value = _unwrap(value)
    if isinstance(value, np.ndarray) and (value.dtype.names or value.dtype.kind == "O"):
        for entry in value.flat:
            yield from _cycle_entries(entry)
    elif isinstance(value, np.ndarray) and value.size == 0:
        return
    else:
        yield value


def summarize_cycles(value: Any, *, max_depth: int, sample_items: int) -> dict[str, Any]:
    """Count all observed entries and operation labels in a cycle-named container.

    This is structural evidence, not validation of a battery or a discharge-cycle
    definition. The optional 'type' and 'data' fields are read only if present.
    Data-field shapes/dtypes include every entry, even outside displayed samples.
    """
    groups: dict[str, dict[str, Any]] = {}
    total = 0
    non_struct = 0
    for entry in _cycle_entries(value):
        total += 1
        fields = _fields(entry)
        if not fields:
            non_struct += 1
        label = _text_label(fields.get("type")) or "<missing-or-nontext-type>"
        if label not in groups:
            groups[label] = {
                "count": 0, "field_presence": Counter(), "data_fields": {},
                "first_entry_structure": describe(entry, max_depth=max_depth, sample_items=sample_items),
            }
        group = groups[label]
        group["count"] += 1
        group["field_presence"].update(fields.keys())
        for name, item in _fields(fields.get("data")).items():
            stats = group["data_fields"].setdefault(name, {
                "present_entries": 0, "dtypes": Counter(), "shapes": Counter(),
            })
            stats["present_entries"] += 1
            stats["dtypes"][str(getattr(item, "dtype", type(item).__name__))] += 1
            stats["shapes"][tuple(getattr(item, "shape", ()))] += 1
    for group in groups.values():
        for stats in group["data_fields"].values():
            shapes = stats.pop("shapes")
            stats["distinct_shape_count"] = len(shapes)
            stats["shape_examples"] = [
                {"shape": list(shape), "entry_count": count}
                for shape, count in sorted(shapes.items())[:10]
            ]
            stats["omitted_shape_count"] = max(0, len(shapes) - 10)
    return {
        "observed_entry_count": total, "non_struct_entry_count": non_struct,
        "by_observed_type": groups,
        "note": "Entries are observed records, not verified charge/discharge pairs or RUL cycles.",
    }


def find_cycle_containers(contents: dict[str, Any], *, max_depth: int, sample_items: int) -> dict[str, Any]:
    """Search all nested struct/cell entries for fields named cycle/cycles.

    Discovery has an independent 32-level/100000-node bound, reported explicitly.
    Display sampling never limits cycle counts inside a discovered container.
    """
    containers: list[dict[str, Any]] = []
    budget = [100000]
    truncated = [False]

    def walk(value: Any, path: str, depth: int) -> None:
        if budget[0] <= 0 or depth > 32:
            truncated[0] = True
            return
        budget[0] -= 1
        value = _unwrap(value)
        fields = _fields(value)
        if fields:
            for name, child in fields.items():
                child_path = f"{path}.{name}"
                if name.lower() in {"cycle", "cycles"}:
                    containers.append({
                        "path": child_path,
                        "container_shape": list(getattr(child, "shape", ())),
                        **summarize_cycles(child, max_depth=max_depth, sample_items=sample_items),
                    })
                else:
                    walk(child, child_path, depth + 1)
        elif isinstance(value, np.ndarray) and (value.dtype.names or value.dtype.kind == "O"):
            for i, child in enumerate(value.flat):
                if budget[0] <= 0:
                    truncated[0] = True
                    break
                walk(child, f"{path}[flat={i}]", depth + 1)

    for name, value in contents.items():
        if name.lower() in {"cycle", "cycles"}:
            containers.append({
                "path": name, "container_shape": list(getattr(value, "shape", ())),
                **summarize_cycles(value, max_depth=max_depth, sample_items=sample_items),
            })
        else:
            walk(value, name, 0)
    return {"containers": containers, "search_truncated": truncated[0]}


def inspect_file(path: Path, *, max_depth: int = 6, sample_items: int = 2) -> dict[str, Any]:
    """Inventory one real input with a SHA-256 fingerprint for reproducibility."""
    LOGGER.info("Inspecting %s", path)
    inventory, contents = read_mat_file(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    cycles = find_cycle_containers(contents, max_depth=max_depth, sample_items=sample_items)
    warnings = []
    if not cycles["containers"]:
        warnings.append("No fields named cycle/cycles found; inspect variables before defining a loader.")
    if cycles["search_truncated"]:
        warnings.append("Cycle discovery hit a traversal limit; the container list may be incomplete.")
    return {
        "path": str(path.resolve()), "size_bytes": path.stat().st_size,
        "sha256": digest, "status": "inspected", "variables": inventory,
        "structure": {name: describe(value, max_depth=max_depth, sample_items=sample_items)
                      for name, value in contents.items()},
        "cycle_discovery": cycles, "warnings": warnings,
    }


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect local files; exit 0 on success, 1 on failures, 2 on missing inputs."""
    config = ProjectConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=config.paths.raw,
                        help="Directory searched recursively (default: project data/raw)")
    parser.add_argument("--output", type=Path, help="Save JSON to a new file instead of stdout; never overwrites")
    parser.add_argument("--max-depth", type=_positive_int, default=6, help="Structure display depth (default: 6)")
    parser.add_argument("--sample-items", type=_positive_int, default=2, help="Expanded struct/cell entries (default: 2)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    raw_dir = args.raw_dir.expanduser().resolve()
    if args.output and args.output.suffix.lower() != ".json":
        parser.error("--output must have a .json suffix")
    report: dict[str, Any] = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_kind": "dataset_structure_inspection; not experimental results",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "project": __version__},
        "expected_source": {"repository": OFFICIAL_REPOSITORY_URL, "archive": OFFICIAL_ARCHIVE_URL,
                            "note": "Source authenticity cannot be established from local filenames alone."},
        "raw_dir": str(raw_dir), "configuration": config.to_dict(),
        "inspection_options": {"max_depth": args.max_depth, "sample_items": args.sample_items},
        "files": [], "errors": [],
    }
    try:
        files = discover_mat_files(raw_dir)
    except (DatasetError, OSError) as exc:
        files = []
        report["errors"].append(str(exc))
    exit_code = 0
    if not files:
        exit_code = 2
        message = (
            f"No usable .mat files found under {raw_dir}. Download the official NASA archive "
            f"manually from {OFFICIAL_ARCHIVE_URL}, extract it (including nested archives), "
            "and place the .mat files under data/raw/ or pass --raw-dir. See data/raw/README.md. "
            "No experiment has run."
        )
        report["errors"].append(message)
        LOGGER.error(message)
    for path in files:
        try:
            report["files"].append(inspect_file(path, max_depth=args.max_depth, sample_items=args.sample_items))
        except Exception as exc:
            # Continue across independent files; retain the failure in the saved report.
            LOGGER.error("Failed to inspect %s: %s", path, exc, exc_info=args.verbose)
            report["files"].append({"path": str(path), "status": "error", "error": str(exc)})
            exit_code = 1
    report["status"] = {0: "inspected", 1: "inspection_failed", 2: "no_data"}[exit_code]
    report["file_count"] = len(files)
    report["inspected_count"] = sum(item["status"] == "inspected" for item in report["files"])
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        except OSError as exc:
            LOGGER.error("Cannot save report to %s: %s. Choose a new output filename.", output, exc)
            return 1
        LOGGER.info("Saved %s report to %s", report["status"], output)
    else:
        print(payload, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

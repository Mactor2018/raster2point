from __future__ import annotations

from pathlib import Path
import numpy as np


def load_points(path: str | Path) -> tuple[np.ndarray, dict]:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path), {}
    if path.suffix.lower() == ".npz":
        archive = np.load(path)
        key = "points" if "points" in archive else archive.files[0]
        return archive[key], {k: archive[k].tolist() if archive[k].ndim == 0 else archive[k] for k in archive.files if k != key}
    if path.suffix.lower() == ".csv":
        return _load_csv(path), {}
    if path.suffix.lower() == ".ply":
        return _load_ply(path)
    raise ValueError(f"unsupported point format: {path.suffix} (use .npy, .npz, .csv, or ASCII .ply)")


def save_points(path: str | Path, points: np.ndarray, metadata: dict | None = None) -> None:
    path = Path(path)
    metadata = metadata or {}
    if path.suffix.lower() == ".npy":
        np.save(path, points)
    elif path.suffix.lower() == ".npz":
        np.savez(path, points=points, **{k: v for k, v in metadata.items() if isinstance(v, (np.ndarray, list, tuple, int, float, str))})
    elif path.suffix.lower() == ".csv":
        np.savetxt(path, points, delimiter=",")
    elif path.suffix.lower() == ".ply":
        _save_ply(path, points)
    else:
        raise ValueError(f"unsupported output format: {path.suffix}")


def _load_csv(path: Path) -> np.ndarray:
    """Read a numeric CSV, skipping a header row only if one is actually present.

    save_points writes bare numeric rows, so skipping unconditionally would
    silently drop the first point on every save/load round trip.
    """
    with path.open() as handle:
        first = handle.readline()
    return np.loadtxt(path, delimiter=",", ndmin=2, skiprows=0 if _is_numeric_row(first) else 1)


def _is_numeric_row(line: str) -> bool:
    if not line.strip():
        return False
    try:
        [float(field) for field in line.split(",")]
    except ValueError:
        return False
    return True


def _load_ply(path: Path) -> tuple[np.ndarray, dict]:
    lines = path.read_text().splitlines()
    end = lines.index("end_header")
    vertex = next(line for line in lines[:end] if line.startswith("element vertex "))
    count = int(vertex.split()[-1])
    properties = [line.split()[-1] for line in lines[:end] if line.startswith("property")]
    return np.loadtxt(lines[end + 1:end + 1 + count], ndmin=2), {"properties": properties}


def _save_ply(path: Path, points: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("PLY output needs point rows of at least x, y, z")
    names = ["x", "y", "z"] + [f"value_{i}" for i in range(points.shape[1] - 3)]
    header = ["ply", "format ascii 1.0", f"element vertex {len(points)}"] + [f"property float {name}" for name in names] + ["end_header"]
    path.write_text("\n".join(header) + "\n" + "\n".join(" ".join(map(str, row)) for row in points) + "\n")

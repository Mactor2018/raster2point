from __future__ import annotations

from pathlib import Path
import json
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
        return np.loadtxt(path, delimiter=",", ndmin=2, skiprows=1), {}
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


def _load_ply(path: Path) -> tuple[np.ndarray, dict]:
    lines = path.read_text().splitlines()
    end = lines.index("end_header")
    vertex = next(line for line in lines[:end] if line.startswith("element vertex "))
    count = int(vertex.split()[-1])
    properties = [line.split()[-1] for line in lines[:end] if line.startswith("property")]
    return np.loadtxt(lines[end + 1:end + 1 + count], ndmin=2), {"properties": properties}


def _save_ply(path: Path, points: np.ndarray) -> None:
    names = ["x", "y", "z"] + [f"value_{i}" for i in range(points.shape[1] - 3)]
    header = ["ply", "format ascii 1.0", f"element vertex {len(points)}"] + [f"property float {name}" for name in names] + ["end_header"]
    path.write_text("\n".join(header) + "\n" + "\n".join(" ".join(map(str, row)) for row in points) + "\n")

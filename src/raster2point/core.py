from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from affine import Affine

@dataclass(frozen=True)
class Raster:
    data: np.ndarray
    transform: Affine
    crs: Any = None
    nodata: Any = None
    def __post_init__(self) -> None:
        array = np.asarray(self.data)
        if array.ndim == 2: array = array[None, ...]
        if array.ndim != 3: raise ValueError("raster data must have shape (bands, rows, cols) or (rows, cols)")
        object.__setattr__(self, "data", array)
    @property
    def height(self) -> int: return self.data.shape[1]
    @property
    def width(self) -> int: return self.data.shape[2]
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        corners = [self.transform * xy for xy in ((0, 0), (self.width, 0), (0, self.height), (self.width, self.height))]
        xs, ys = zip(*corners)
        return min(xs), min(ys), max(xs), max(ys)

@dataclass(frozen=True)
class AlignmentReport:
    crs_match: bool | None
    overlap: bool
    raster_bounds: tuple[float, float, float, float]
    point_bounds: tuple[float, float, float, float]
    points_inside: int
    points_outside: int
    message: str

@dataclass(frozen=True)
class TransferResult:
    values: np.ndarray
    valid: np.ndarray
    pixel: np.ndarray

def _bounds(points: np.ndarray) -> tuple[float, float, float, float]:
    if points.ndim != 2 or points.shape[1] < 2: raise ValueError("points must have shape (n, >=2)")
    xy = points[:, :2]
    return float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())

def _project_xy(raster: Raster, points: np.ndarray, point_crs: Any) -> np.ndarray:
    if point_crs is None or raster.crs is None or str(point_crs) == str(raster.crs): return points[:, :2].astype(float, copy=False)
    from rasterio.warp import transform
    x, y = transform(point_crs, raster.crs, points[:, 0].tolist(), points[:, 1].tolist())
    return np.column_stack((x, y))

def _pixel_coordinates(raster: Raster, points: np.ndarray, point_crs: Any = None) -> tuple[np.ndarray, np.ndarray]:
    cols, rows = (~raster.transform) * tuple(_project_xy(raster, points, point_crs).T)
    # Baseline convention: affine coordinates are pixel edges; subtract 0.5 for centers.
    return np.asarray(cols, dtype=float) - 0.5, np.asarray(rows, dtype=float) - 0.5

def alignment_report(raster: Raster, points: np.ndarray, point_crs: Any = None) -> AlignmentReport:
    points = np.asarray(points); projected = _project_xy(raster, points, point_crs)
    point_bounds = _bounds(np.column_stack((projected, points[:, 2:])))
    rb = raster.bounds
    overlap = not (point_bounds[2] < rb[0] or point_bounds[0] > rb[2] or point_bounds[3] < rb[1] or point_bounds[1] > rb[3])
    cols, rows = _pixel_coordinates(raster, points, point_crs)
    inside = (cols >= -0.5) & (cols < raster.width - 0.5) & (rows >= -0.5) & (rows < raster.height - 0.5)
    crs_match = None if raster.crs is None or point_crs is None else str(raster.crs) == str(point_crs)
    note = "CRS unavailable; spatial overlap is only a plausibility check." if crs_match is None else ("CRS matches; physical registration is not proven." if crs_match else "CRS mismatch.")
    return AlignmentReport(crs_match, overlap, rb, point_bounds, int(inside.sum()), int((~inside).sum()), note)

def transfer(points: np.ndarray, raster: Raster, *, fill_value: Any = np.nan, point_crs: Any = None, require_crs_match: bool = False, interpolation: str = "bilinear") -> TransferResult:
    """Use the HyperPointFormer YEG3D baseline's pixel-center sampler."""
    points = np.asarray(points); report = alignment_report(raster, points, point_crs)
    if require_crs_match and report.crs_match is not True: raise ValueError("raster and point-cloud CRS do not both exist and match")
    cols, rows = _pixel_coordinates(raster, points, point_crs)
    left, top = np.floor(cols).astype(np.int64), np.floor(rows).astype(np.int64)
    pixel = np.column_stack((np.floor(rows + 0.5).astype(np.int64), np.floor(cols + 0.5).astype(np.int64)))
    value_dtype = np.result_type(raster.data.dtype, type(fill_value), np.float64)
    values = np.zeros((len(points), raster.data.shape[0]), dtype=value_dtype) if interpolation == "bilinear" else np.full((len(points), raster.data.shape[0]), fill_value, dtype=value_dtype)
    if interpolation == "nearest":
        rr, cc = np.clip(pixel[:, 0], 0, raster.height - 1), np.clip(pixel[:, 1], 0, raster.width - 1)
        inside = (pixel[:, 0] >= 0) & (pixel[:, 0] < raster.height) & (pixel[:, 1] >= 0) & (pixel[:, 1] < raster.width)
        sampled = np.moveaxis(raster.data[:, rr, cc], 0, 1)
        good = inside & (~np.any(sampled == raster.nodata, axis=1) if raster.nodata is not None else True)
        values[good] = sampled[good]
        return TransferResult(values, good, pixel)
    if interpolation != "bilinear": raise ValueError("interpolation must be 'bilinear' or 'nearest'")
    dx, dy = cols - left, rows - top; weights = np.zeros(len(points), dtype=float)
    for dr, dc, weight in ((0, 0, (1 - dy) * (1 - dx)), (0, 1, (1 - dy) * dx), (1, 0, dy * (1 - dx)), (1, 1, dy * dx)):
        rr, cc = top + dr, left + dc
        usable = (rr >= 0) & (rr < raster.height) & (cc >= 0) & (cc < raster.width) & np.isfinite(weight)
        indices = np.flatnonzero(usable)
        if not len(indices): continue
        sample = raster.data[:, rr[indices], cc[indices]].T; good = np.isfinite(sample).all(axis=1)
        if raster.nodata is not None: good &= ~np.any(sample == raster.nodata, axis=1)
        indices = indices[good]; w = weight[indices]
        values[indices] += raster.data[:, rr[indices], cc[indices]].T * w[:, None]; weights[indices] += w
    valid = weights > 0; values[valid] /= weights[valid, None]; values[~valid] = fill_value
    return TransferResult(values, valid, pixel)

def transfer_mask(points: np.ndarray, mask: np.ndarray, transform: Affine, *, crs: Any = None, point_crs: Any = None, fill_value: int = -1) -> TransferResult:
    if not np.issubdtype(np.asarray(mask).dtype, np.integer) and not np.issubdtype(np.asarray(mask).dtype, np.bool_): raise ValueError("semantic masks must be integer or boolean arrays")
    return transfer(points, Raster(np.asarray(mask), transform, crs=crs), fill_value=fill_value, point_crs=point_crs, interpolation="nearest")

def load_raster(path: str | Path) -> Raster:
    import rasterio
    with rasterio.open(path) as source: return Raster(source.read(), source.transform, source.crs, source.nodata)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine


@dataclass(frozen=True)
class Raster:
    """An array in (bands, rows, columns) order and its world transform."""

    data: np.ndarray
    transform: Affine
    crs: Any = None
    nodata: Any = None

    def __post_init__(self) -> None:
        array = np.asarray(self.data)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3:
            raise ValueError("raster data must have shape (bands, rows, cols) or (rows, cols)")
        object.__setattr__(self, "data", array)

    @property
    def height(self) -> int:
        return self.data.shape[1]

    @property
    def width(self) -> int:
        return self.data.shape[2]

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
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points must have shape (n, >=2)")
    xy = points[:, :2]
    return float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())


def _pixels(raster: Raster, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inverse = ~raster.transform
    col_row = np.array([inverse * (float(x), float(y)) for x, y in points[:, :2]], dtype=float)
    # Pixel centers own [integer - .5, integer + .5); nearest pixel is floor(u + .5).
    cols = np.floor(col_row[:, 0] + 0.5).astype(np.int64)
    rows = np.floor(col_row[:, 1] + 0.5).astype(np.int64)
    # Test the continuous coordinate before rounding so an outside point just
    # beyond an edge cannot round back into the first/last pixel.
    valid = (col_row[:, 0] >= 0) & (col_row[:, 0] < raster.width) & (col_row[:, 1] >= 0) & (col_row[:, 1] < raster.height)
    cols = np.clip(cols, 0, raster.width - 1)
    rows = np.clip(rows, 0, raster.height - 1)
    return np.column_stack((rows, cols)), valid


def alignment_report(raster: Raster, points: np.ndarray, point_crs: Any = None) -> AlignmentReport:
    point_bounds = _bounds(np.asarray(points))
    rb = raster.bounds
    overlap = not (point_bounds[2] < rb[0] or point_bounds[0] > rb[2] or point_bounds[3] < rb[1] or point_bounds[1] > rb[3])
    _, inside = _pixels(raster, np.asarray(points))
    crs_match = None if raster.crs is None or point_crs is None else str(raster.crs) == str(point_crs)
    note = "CRS unavailable; spatial overlap is only a plausibility check." if crs_match is None else ("CRS matches; physical registration is not proven." if crs_match else "CRS mismatch.")
    return AlignmentReport(crs_match, overlap, rb, point_bounds, int(inside.sum()), int((~inside).sum()), note)


def transfer(points: np.ndarray, raster: Raster, *, fill_value: Any = np.nan, point_crs: Any = None, require_crs_match: bool = False) -> TransferResult:
    """Sample nearest raster pixels at point XY, preserving point order."""
    points = np.asarray(points)
    report = alignment_report(raster, points, point_crs)
    if require_crs_match and report.crs_match is not True:
        raise ValueError("raster and point-cloud CRS do not both exist and match")
    pixel, valid = _pixels(raster, points)
    values = np.full((len(points), raster.data.shape[0]), fill_value, dtype=np.result_type(raster.data.dtype, type(fill_value)))
    if valid.any():
        values[valid] = np.moveaxis(raster.data[:, pixel[valid, 0], pixel[valid, 1]], 0, 1)
    if raster.nodata is not None and valid.any():
        valid_indices = np.flatnonzero(valid)
        valid[valid_indices] &= ~np.any(values[valid_indices] == raster.nodata, axis=1)
        values[~valid] = fill_value
    return TransferResult(values, valid, pixel)


def transfer_mask(points: np.ndarray, mask: np.ndarray, transform: Affine, *, crs: Any = None, point_crs: Any = None, fill_value: int = -1) -> TransferResult:
    """Convenience API for an in-memory binary or integer mask."""
    if not np.issubdtype(np.asarray(mask).dtype, np.integer) and not np.issubdtype(np.asarray(mask).dtype, np.bool_):
        raise ValueError("semantic masks must be integer or boolean arrays")
    return transfer(points, Raster(np.asarray(mask), transform, crs=crs, nodata=None), fill_value=fill_value, point_crs=point_crs)


def load_raster(path: str | Path) -> Raster:
    import rasterio
    with rasterio.open(path) as source:
        data = source.read()
        return Raster(data, source.transform, source.crs, source.nodata)

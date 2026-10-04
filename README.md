# raster2point

`raster2point` transfers values from a georeferenced raster onto an existing
3D point cloud. It is a small, deterministic utility for combining 2D
orthophotos or semantic masks with 3D points, without depending on a machine-
learning framework.

The operation uses each point's `x` and `y` coordinates to locate the
corresponding raster pixel. The point's `z` coordinate and every existing
attribute are copied unchanged; sampled raster values are appended as new
columns.

## What it supports

- GeoTIFF rasters read through Rasterio.
- RGB, multispectral, continuous, and integer-valued rasters.
- Semantic masks with nearest-neighbor sampling.
- Point inputs in `.npy`, `.npz`, `.csv`, and ASCII `.ply` formats.
- Point outputs in `.npy`, `.npz`, `.csv`, and ASCII `.ply` formats.
- Optional reprojection of point coordinates when a point CRS is supplied
  through the Python API.
- Alignment inspection before transfer.

Point arrays must have at least three columns in this order:

```text
x, y, z, [existing attributes...]
```

For a raster with three bands, an output row is therefore:

```text
x, y, z, [existing attributes...], band_1, band_2, band_3
```

## Installation

```bash
python -m venv .venv
```

Activate the environment, then install the package:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
```

The package requires Python 3.10 or newer, NumPy, Rasterio, and Affine.

## Command-line usage

The installed command is `raster2point`. Each transfer command takes a raster,
an input point file, and an output path.

### Inspect alignment

Run this before transferring values:

```bash
raster2point inspect ortho.tif points.npy
```

The report includes raster bounds, point bounds, overlap, and the number of
points inside and outside the raster. It is a plausibility check, not proof of
survey-level registration.

### Transfer RGB or continuous values

`rgb` and `transfer` use bilinear interpolation. The four neighboring pixels
are weighted by distance. NoData neighbors are omitted and the remaining
weights are renormalized, so points in the half-pixel border inside the raster
edge are still sampled from the neighbors that exist. A NoData neighbor means
one equal to the raster's declared NoData value or one that is non-finite,
which covers the `NaN` NoData that float rasters commonly declare.

Sampling stops at the raster footprint. A point outside it receives the fill
value and is reported invalid, even when it lies within one pixel of the edge
and a single bilinear corner would still be in bounds. The `points_inside`
count from `inspect` therefore matches the number of points that can be
sampled.

```bash
raster2point rgb ortho.tif points.npy points_rgb.npy
raster2point transfer ortho.tif points.ply points_rgb.ply
```

### Transfer a semantic mask

`mask` uses nearest-neighbor sampling, so class IDs are never blended into
intermediate values. `transfer_mask` in the Python API also returns an integer
array when the mask and fill value are both integral, which keeps large class
IDs exact; CLI output is written alongside float `x/y/z` columns and is
therefore float-typed:

```bash
raster2point mask road_mask.tif points.npy points_road.npy --fill -1
```

The default fill value is `-1` for masks and `NaN` for RGB/continuous values.
It is used for points outside the raster or points whose sampled neighbors are
all NoData.

### Select the output format

By default, the output format is inferred from the output filename extension.
Use `--output-type` when the output path has no extension or when the desired
format should be explicit:

```bash
raster2point rgb ortho.tif points.npy result --output-type npy
raster2point rgb ortho.tif points.npy result --output-type csv
raster2point rgb ortho.tif points.npy result --output-type ply
```

These commands write `result.npy`, `result.csv`, and `result.ply`, respectively.
Supported values are `npy`, `npz`, `csv`, and `ply`.

CSV output is numeric, comma-delimited, and written without a header row. The
CSV reader accepts both forms: a leading row is skipped only when it is not
numeric, so a written CSV reads back with every point intact. PLY output is
ASCII and contains one `float` property for each output column. The PLY reader
currently accepts ASCII PLY only; binary PLY files should be converted to ASCII
or NumPy first.

### Process multiple pairs

Create a manifest with exactly these columns:

```csv
raster,points,output
ortho_a.tif,points_a.npy,points_a_rgb.npy
road_a.tif,points_a.npy,points_a_road.npy
```

Run the batch:

```bash
raster2point batch pairs.csv
raster2point batch pairs.csv --overwrite
```

Existing outputs are skipped unless `--overwrite` is supplied. Every row is
attempted; failures are reported and the command exits unsuccessfully if any
row fails. Batch output formats are inferred from each output filename.

The manifest carries no per-row sampler, so every batch row uses bilinear
interpolation with a `NaN` fill, as `transfer` does. Semantic masks must not be
run through `batch`: bilinear interpolation blends neighboring class IDs into
meaningless intermediate values. Use the `mask` command for masks.

## Python API

Use the API when the raster or point data is already in memory, or when CRS
information must be supplied explicitly:

```python
import numpy as np
from raster2point import Raster, transfer
from raster2point.core import transfer_mask

raster = Raster(
    data=rgb_array,
    transform=source_transform,
    crs="EPSG:26912",
    nodata=0,
)

sampled = transfer(points, raster, fill_value=np.nan, point_crs="EPSG:26912")
rgb_values = sampled.values
valid_points = sampled.valid

labels = transfer_mask(
    points,
    mask_array,
    mask_transform,
    crs="EPSG:26912",
    point_crs="EPSG:26912",
    fill_value=-1,
)
```

`Raster.data` may have shape `(bands, rows, columns)` or `(rows, columns)`.
The affine transform must describe the actual raster array. If a model uses a
cropped or resized image, derive and pass the corresponding crop/resampling
transform; do not infer geographic placement from array dimensions alone.

## Coordinate systems and alignment

The raster's CRS and affine transform are authoritative. If both the raster
and point CRS are supplied and differ, the Python API reprojects point `x/y`
coordinates into the raster CRS before sampling. The CLI point formats do not
carry CRS metadata, so CLI inputs are assumed to already use the raster CRS.

Matching CRS and overlapping bounds do not establish physical registration.
Check the source transform, origin, pixel size, and known control points before
using the output for quantitative analysis.

This is 2D-to-3D attribute transfer, not 3D scene reconstruction. Points with
different heights but the same projected `x/y` receive the same raster value.
The operation does not alter point order, coordinates, elevation, or existing
attributes.

## Development and tests

Install the package in editable mode and run the test suite:

```bash
pip install -e .
python -m pytest
```

The tests cover affine pixel lookup, bilinear RGB transfer, nearest-neighbor
mask transfer, NoData handling (sentinel and `NaN`), CRS and bounds reporting,
outside points and footprint edges, integer label preservation, CSV round
trips, empty and non-finite input, and point-order preservation. Tests use
synthetic data and do not modify source datasets.

## License and project status

This repository is a focused utility rather than a full point-cloud processing
pipeline. It does not create point clouds from a raster alone; a compatible
3D point file is required as input.

## Acknowledgment

The raster-to-point projection approach was adapted from the coordinate and
pixel-sampling strategy used in [HyperPointFormer](https://github.com/aldinorizaldy/hyperpointformer)
by Aldino Rizaldy and collaborators. We thank the authors for making their
research and implementation available. The related publication is:

> A. Rizaldy, R. Gloaguen, F. E. Fassnacht, and P. Ghamisi, “HyperPointFormer:
> Multimodal Fusion in 3-D Space With Dual-Branch Cross-Attention Transformers,”
> *IEEE Journal of Selected Topics in Applied Earth Observations and Remote
> Sensing*, vol. 18, pp. 21254–21274, 2025.

See the [published paper](https://doi.org/10.1109/JSTARS.2025.3595648) and the
[upstream repository](https://github.com/aldinorizaldy/hyperpointformer) for
the original work. The upstream repository did not include an explicit license
file when inspected on 2026-09-04; its code and other materials remain subject
to the rights of their original authors. This repository's MIT license applies
only to the original contents of `raster2point`.

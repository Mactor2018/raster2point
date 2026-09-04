# raster2point

Small, deterministic utilities for sampling georeferenced raster attributes onto corresponding 3D points. It is deliberately independent of machine-learning frameworks.

The core operation inverse-transforms each point's `(x, y)` into raster pixel coordinates, chooses the nearest pixel, and appends its band values without changing point order. `z` and all existing attributes are untouched. `rgb` and `mask` are CLI names for the same operation; masks use nearest-neighbor integer sampling. Out-of-bounds and NoData samples are marked invalid and receive the configured fill value.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
```

GeoTIFF input is read with Rasterio. Point clouds can be `.npy`, `.npz`, `.csv`, or ASCII `.ply`; arrays must have at least `x,y,z` in their first three columns. Outputs use the extension supplied.

## CLI

```bash
raster2point inspect ortho.tif points.npy
raster2point rgb ortho.tif points.npy points_rgb.npy
raster2point mask road_mask.tif points.npy points_labels.npy --fill -1
raster2point transfer ortho.tif points.ply points_rgb.ply --require-crs-match
raster2point batch pairs.csv
```

Batch CSV columns are `raster,points,output`; existing outputs are skipped unless `--overwrite` is used. Failures are reported per row and produce a nonzero exit status.

## Python API and in-memory masks

```python
import numpy as np
from affine import Affine
from raster2point import Raster, transfer
from raster2point.core import transfer_mask

raster = Raster(rgb_array, transform, crs="EPSG:26912", nodata=0)
result = transfer(points, raster, fill_value=np.nan)
labels = transfer_mask(points, mask_array, crop_transform, crs="EPSG:26912", fill_value=-1)
```

An in-memory array without its original affine transform and CRS is insufficient. For a cropped or resized network input, derive the crop/resampled transform from the source raster and pass that transform with the array; never map it using array shape alone.

## Alignment and limitations

`inspect` reports CRS compatibility, raster and point XY bounds, overlap, and inside/outside counts. Matching CRS and overlap only establish plausible metadata compatibility; they do not prove physical registration or remove a residual XY offset. The mapping is XY/image correspondence, not 3D scene understanding: points at different heights sharing an XY location receive the same pixel value.

The HyperPointFormer repository was inspected for the requested compatibility check. Its published data loader consumes point records already containing spectral features; it does not define a georeferenced orthophoto projection/KD-tree routine. This project therefore preserves the relevant nearest-pixel behavior with the affine transform as the authoritative geographic mapping, avoiding unnecessary resizing or a fragile KD-tree. The `transfer` API is suitable between a 2D model and a later 3D model.

## Development

```bash
pip install -e .
python -m pytest
```

Synthetic tests cover affine lookup, RGB/mask transfer, NoData, bounds, CRS, outside points, and order preservation. No source datasets are modified.

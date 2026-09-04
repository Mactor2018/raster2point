import numpy as np
from affine import Affine
from raster2point.core import Raster, alignment_report, transfer, transfer_mask


def raster():
    return Raster(np.array([[[10, 20], [30, 40]], [[1, 2], [3, 4]]]), Affine.translation(100, 200) * Affine.scale(2, -2), crs="EPSG:26912", nodata=40)


def test_rgb_order_and_nodata():
    result = transfer(np.array([[101, 199, 9], [103, 197, 8], [99, 200, 7]]), raster(), fill_value=-1)
    assert result.values.tolist() == [[10, 1], [-1, -1], [-1, -1]]
    assert result.valid.tolist() == [True, False, False]


def test_baseline_bilinear_pixel_center_sampling():
    result = transfer(np.array([[102, 198, 9]]), Raster(np.array([[0, 10], [20, 30]]), raster().transform))
    assert result.values.tolist() == [[15.0]]


def test_mask_and_alignment():
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    result = transfer_mask(np.array([[100, 200, 1], [102, 198, 2], [104, 200, 3]]), mask, raster().transform)
    assert result.values[:, 0].tolist() == [0, 3, -1]
    report = alignment_report(raster(), np.array([[100, 200, 0], [105, 197, 0]]), "EPSG:26912")
    assert report.crs_match and report.points_inside == 1 and report.overlap

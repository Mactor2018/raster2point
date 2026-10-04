import numpy as np
import pytest
from affine import Affine
from raster2point.core import Raster, alignment_report, transfer, transfer_mask
from raster2point.io import load_points, save_points


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


def test_nearest_rejects_nan_nodata():
    # Float rasters commonly declare NaN NoData; `== nodata` never matches it,
    # so the sample must be rejected on non-finiteness instead.
    nan_raster = Raster(np.array([[[1.0, np.nan], [3.0, 4.0]]]), raster().transform, nodata=np.nan)
    point = np.array([[103.0, 199.0, 0.0]])  # pixel-center of the NaN cell
    for interpolation in ("nearest", "bilinear"):
        result = transfer(point, nan_raster, fill_value=-999.0, interpolation=interpolation)
        assert result.valid.tolist() == [False], interpolation
        assert result.values.tolist() == [[-999.0]], interpolation


def test_bilinear_does_not_extrapolate_outside_footprint():
    # A point up to one pixel beyond the edge still has one in-bounds bilinear
    # corner; it must not be sampled off that corner alone.
    plain = Raster(np.array([[[10.0, 20.0], [30.0, 40.0]]]), raster().transform)
    outside = np.array([[99.4, 200.6, 0.0]])
    assert plain.bounds == (100.0, 196.0, 104.0, 200.0)
    report = alignment_report(plain, outside)
    result = transfer(outside, plain, fill_value=-999.0)
    assert report.points_inside == 0 and report.points_outside == 1
    assert result.valid.tolist() == [False]
    assert result.values.tolist() == [[-999.0]]


def test_bilinear_still_interpolates_inside_edge_pixels():
    # The half-pixel border inside the footprint keeps renormalized weights.
    plain = Raster(np.array([[[10.0, 20.0], [30.0, 40.0]]]), raster().transform)
    result = transfer(np.array([[100.5, 199.5, 0.0]]), plain, fill_value=-999.0)
    assert result.valid.tolist() == [True]
    assert result.values.tolist() == [[10.0]]


def test_mask_labels_stay_integers():
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    result = transfer_mask(np.array([[100.0, 200.0, 1.0], [104.0, 200.0, 3.0]]), mask, raster().transform)
    assert np.issubdtype(result.values.dtype, np.integer)
    assert result.values[:, 0].tolist() == [0, -1]
    # int64 labels beyond 2**53 survive only if the result never becomes a float.
    big = np.array([[2**53 + 1, 2**53 + 3]], dtype=np.int64)
    exact = transfer_mask(np.array([[100.0, 200.0, 1.0]]), big, raster().transform)
    assert exact.values[0, 0] == 2**53 + 1


def test_csv_round_trip_preserves_every_point(tmp_path):
    points = np.array([[100.0, 200.0, 1.0], [102.0, 198.0, 2.0], [104.0, 196.0, 3.0]])
    path = tmp_path / "rt.csv"
    save_points(path, points)
    back, _ = load_points(path)
    assert np.array_equal(back, points)


def test_csv_with_header_still_skips_it(tmp_path):
    path = tmp_path / "headered.csv"
    path.write_text("x,y,z\n100.0,200.0,1.0\n102.0,198.0,2.0\n")
    back, _ = load_points(path)
    assert np.array_equal(back, np.array([[100.0, 200.0, 1.0], [102.0, 198.0, 2.0]]))


def test_empty_point_cloud_is_handled():
    plain = Raster(np.array([[[10.0, 20.0], [30.0, 40.0]]]), raster().transform)
    empty = np.zeros((0, 3))
    report = alignment_report(plain, empty)
    assert report.points_inside == 0 and report.points_outside == 0 and report.overlap is False
    for interpolation in ("bilinear", "nearest"):
        result = transfer(empty, plain, interpolation=interpolation)
        assert result.values.shape == (0, 1) and result.valid.shape == (0,)


def test_non_finite_coordinates_are_invalid_not_undefined():
    plain = Raster(np.array([[[10.0, 20.0], [30.0, 40.0]]]), raster().transform)
    points = np.array([[np.nan, 199.0, 0.0], [101.0, 199.0, 0.0]])
    for interpolation in ("bilinear", "nearest"):
        result = transfer(points, plain, fill_value=-999.0, interpolation=interpolation)
        assert result.valid.tolist() == [False, True], interpolation
        assert result.values[0, 0] == -999.0, interpolation


def test_rejects_bad_interpolation_and_fill():
    plain = Raster(np.array([[[10.0, 20.0], [30.0, 40.0]]]), raster().transform)
    point = np.array([[101.0, 199.0, 0.0]])
    with pytest.raises(ValueError, match="interpolation"):
        transfer(point, plain, interpolation="cubic")
    with pytest.raises(ValueError, match="fill_value"):
        transfer(point, plain, fill_value=None)

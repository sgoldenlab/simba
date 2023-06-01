import os
import pytest
import numpy as np
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin


def test_euclidean_distance():
    x1, x2 = np.full((10, 1), 1), np.full((10, 1), 2)
    y1, y2 = np.full((10, 1), 2), np.full((10, 1), 1)
    results = FeatureExtractionMixin().euclidean_distance(bp_1_x=x1, bp_2_x=x2, bp_1_y=y1, bp_2_y=y2, px_per_mm=4.56)
    assert results.shape == x1.shape
    assert np.all(results == results[0])
    assert round(results[0][0], 2) == 0.31

def test_angle3pt():
    results = FeatureExtractionMixin.angle3pt(ax=122.0, ay=198.0, bx=237.0, by=138.0, cx=191.0, cy=109)
    assert round(results, 2) == 59.78

def test_angle3pt_serialized():
    x, y, z = np.full((6, 2), 1), np.full((6, 2), 2), np.full((6, 2), 3)
    coordinates = np.hstack((x, y, z))
    results = FeatureExtractionMixin.angle3pt_serialized(data=coordinates)
    assert np.all(results == results[0])
    assert results.shape[0] == x.shape[0]
    assert results[0] == 180

def test_convex_hull_calculator_mp():
    coordinates = np.array([[1, 2], [4, 7], [9, 1], [3, 6]])
    results = FeatureExtractionMixin.convex_hull_calculator_mp(arr=coordinates, px_per_mm=1)
    assert round(results, 2) == 21.76


def test_count_values_in_range():
    data = np.array([[0.01, 0.5, 0.9, 0.3, 0.6], [0.01, 0.5, 0.9, 0.3, 0.6]])
    results = FeatureExtractionMixin.count_values_in_range(data=data, ranges=np.array([[0.0, 0.25], [0.25, 0.5], [0.5, 1.0]]))
    assert results.shape == (2, 3)
    assert np.array_equal(results[0], [1, 2, 3])

def test_framewise_euclidean_distance_roi():
    loc_1 = np.array([[180, 1], [ 24, 35], [6, 129]]).astype('float32')
    loc_2 = np.array([[117, 134]]).astype('float32')
    results = FeatureExtractionMixin.framewise_euclidean_distance_roi(location_1=loc_1, location_2=loc_2, px_per_mm=4.56, centimeter=False).astype(int)
    assert np.array_equal(results, [32, 29, 24])

def test_framewise_inside_rectangle_roi():
    bp_loc = np.array([[5, 7], [4, 2]])
    roi_coords = np.array([[0, 0], [10, 9]])
    results = FeatureExtractionMixin.framewise_inside_rectangle_roi(bp_location=bp_loc, roi_coords=roi_coords)
    assert np.array_equal(results, [1, 1])
    roi_coords = np.array([[10, 0], [10, 9]])
    results = FeatureExtractionMixin.framewise_inside_rectangle_roi(bp_location=bp_loc, roi_coords=roi_coords)
    assert np.array_equal(results, [0, 0])

def test_framewise_inside_polygon_roi():
    bp_loc = np.array([[5, 7], [4, 2]])
    roi_coords = np.array([[5, 7], [6, 4],  [8, 5], [2, 7]]).astype(np.float32)
    results = FeatureExtractionMixin.framewise_inside_polygon_roi(bp_location=bp_loc, roi_coords=roi_coords)
    assert np.array_equal(results, [1, 0])

#test_euclidean_distance()
# test_angle3pt()
# test_angle3pt_serialized()
# test_convex_hull_calculator_mp()
# test_count_values_in_range()
# test_framewise_euclidean_distance_roi()
# test_framewise_inside_rectangle_roi()
# test_framewise_inside_polygon_roi()
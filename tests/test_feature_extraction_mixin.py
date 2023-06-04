import pytest
from copy import deepcopy
import numpy as np
import os
import pandas as pd
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
import simba
from simba.utils.enums import Paths
from scipy.signal import savgol_filter


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
    
def test_windowed_frequentist_distribution_tests():
    data = np.random.randint(1, 10, size=(10))
    results = FeatureExtractionMixin.windowed_frequentist_distribution_tests(data=data, fps=25, feature_name='Anima_1_velocity')
    assert len(results.columns) == 4; assert len(results) == data.shape[0]
    assert results._is_numeric_mixed_type

def test_cdist():
    array_1 = np.array([[3, 9], [5, 1], [2, 5]]).astype('float32')
    array_2 = np.array([[1, 5], [3, 6], [4, 9]]).astype('float32')
    results = FeatureExtractionMixin.cdist(array_1=array_1, array_2=array_2).astype('int')
    assert np.array_equal(results, np.array([[4, 3, 1], [5, 5, 8], [1, 1, 4]]))

def test_create_shifted_df():
    df = pd.DataFrame(data=[10, 95, 85], columns=['Feature_1'])
    results = FeatureExtractionMixin.create_shifted_df(df=df)
    assert list(results.columns) == ['Feature_1', 'Feature_1_shifted']
    assert len(results) == len(df)
    assert results._is_numeric_mixed_type

@pytest.mark.parametrize("pose", ['2 animals 16 body-parts',
                                  '2 animals 14 body-parts',
                                  '2 animals 8 body-parts',
                                  '1 animal 8 body-parts',
                                  '1 animal 7 body-parts',
                                  '1 animal 4 body-parts',
                                  '1 animal 9 body-parts'])
def test_get_feature_extraction_headers(pose):
    simba_dir = os.path.dirname(simba.__file__)
    feature_categories_csv_path = os.path.join(simba_dir, Paths.SIMBA_FEATURE_EXTRACTION_COL_NAMES_PATH.value)
    bps = list(pd.read_csv(feature_categories_csv_path)[pose])
    assert type(bps) == list

def test_minimum_bounding_rectangle():
    points = np.array([[7, 6], [9, 4], [4, 7], [5,1]])
    results = FeatureExtractionMixin.minimum_bounding_rectangle(points=points).astype(int)
    assert np.array_equal(results, np.array([[9, 1], [4, 1], [4, 7], [8, 7]]))

def test_framewise_euclidean_distance():
    loc_1 = np.array([[108, 162]]).astype(np.float32)
    loc_2 = np.array([[91, 11]]).astype(np.float32)
    results_mm = FeatureExtractionMixin.framewise_euclidean_distance_roi(location_1=loc_1, location_2=loc_2, px_per_mm=4.56, centimeter=False).astype(int)
    results_cm = FeatureExtractionMixin.framewise_euclidean_distance_roi(location_1=loc_1, location_2=loc_2, px_per_mm=4.56, centimeter=True).astype(int)
    assert int(results_mm / 10) == results_cm

@pytest.mark.parametrize("time_window", [1, 3])
def test_dataframe_gaussian_smoother(time_window):
    in_df = pd.DataFrame(data=[10, 95, 85], columns=['Feature_1'])
    out_df = deepcopy(in_df)
    for c in in_df.columns: out_df[c] = out_df[c].rolling(window=int(time_window), win_type='gaussian', center=True).mean(std=5).fillna(out_df[c]).abs().astype(int)
    if time_window == 1: assert out_df.equals(in_df)
    else: assert out_df.equals(pd.DataFrame(data=[10, 63, 85], columns=['Feature_1']))

@pytest.mark.parametrize("time_window", [5])
def test_dataframe_savgol_smoother(time_window):
    in_df = pd.DataFrame(data=[10, 95, 85, 109, 205], columns=['Feature_1'])
    out_df = deepcopy(in_df)
    for c in in_df.columns:
        out_df[c] = savgol_filter(x=in_df[c].to_numpy(), window_length=time_window, polyorder=3, mode='nearest').astype(int)
    assert out_df.equals(pd.DataFrame(data=[32, 68, 92, 126, 182], columns=['Feature_1']))



#test_euclidean_distance()
# test_angle3pt()
# test_angle3pt_serialized()
# test_convex_hull_calculator_mp()
# test_count_values_in_range()
# test_framewise_euclidean_distance_roi()
# test_framewise_inside_rectangle_roi()
# test_framewise_inside_polygon_roi()

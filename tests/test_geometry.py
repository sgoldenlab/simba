import numpy as np
import pytest
import math
from shapely.geometry import Polygon, Point, LineString
from simba.mixins.geometry_mixin import GeometryMixin

@pytest.mark.parametrize('cap_style, preserve_topology, parallel_offset', [('round', True, 10), ('square', False, 5), ('flat', True, 3)])
def test_bodyparts_to_polygon(cap_style, preserve_topology, parallel_offset):
    data = np.array([[364, 308], [383, 323], [403, 335], [423, 351]])
    result = GeometryMixin.bodyparts_to_polygon(data=data, cap_style='round')
    assert type(result) == Polygon
    assert result.is_valid

@pytest.mark.parametrize('data, buffer, px_per_mm,', [(np.random.randint(0, 100, (5, 2)), None, None), (np.random.randint(0, 100, (20, 2)), 1, 1)])
def test_bodyparts_to_points(data, buffer, px_per_mm):
    results = GeometryMixin().bodyparts_to_points(data=data, buffer=buffer, px_per_mm=px_per_mm)
    for i in results: assert type(i) in (Point, Polygon)

@pytest.mark.parametrize('parallel_offset, pixels_per_mm,', [(10, 1.5), (15.1, 3)])
def test_bodyparts_to_points(parallel_offset, pixels_per_mm):
    data = np.array([364, 308])
    results = GeometryMixin().bodyparts_to_circle(data=data, parallel_offset=parallel_offset, pixels_per_mm=pixels_per_mm)
    assert type(results) == Polygon
    assert results.is_valid

def test_compute_pct_shape_overlap():
    polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[0, 0], [0, 50], [50, 0], [50, 50]]))
    polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[25, 0], [25, 50], [75, 0], [75, 50]]))
    results = GeometryMixin().compute_pct_shape_overlap(shapes=[polygon_1, polygon_2])
    assert 0.0 <= results <= 100.0

def test_crosses():
    line_1 = GeometryMixin().bodyparts_to_line(np.array([[10, 10], [20, 10], [30, 10], [40, 10]]))
    line_2 = GeometryMixin().bodyparts_to_line(np.array([[25, 5], [25, 20], [25, 30], [25, 40]]))
    results = GeometryMixin().crosses(shapes=[line_1, line_2])
    assert results is True
    line_2 = GeometryMixin().bodyparts_to_line(np.array([[100, 50], [75, 10], [102, 60], [52, 59]]))
    results = GeometryMixin().crosses(shapes=[line_1, line_2])
    assert results is False
def test_is_shape_covered():
    polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[10, 10], [10, 100], [100, 10], [100, 100]]))
    polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[25, 25], [25, 75], [90, 25], [90, 75]]))
    results = GeometryMixin().is_shape_covered(shapes=[polygon_2, polygon_1])
    assert results is True
    polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[0, 0], [0, 75], [90, 25], [90, 75]]))
    results = GeometryMixin().is_shape_covered(shapes=[polygon_2, polygon_1])
    assert results is False

@pytest.mark.parametrize('data_size,', [(10, 2), (15, 2), (4, 2)])
def test_area(data_size):
    data = np.random.randint(0, 100, size=(data_size))
    polygon = GeometryMixin().bodyparts_to_polygon(data)
    results = GeometryMixin().area(shape=polygon, pixels_per_mm=4.9)
    assert results >= 0
def test_bodyparts_to_line():
    data = np.array([[364, 308], [383, 323], [403, 335], [423, 351]])
    results = GeometryMixin().bodyparts_to_line(data=data)
    assert type(results) == LineString
    results = GeometryMixin().bodyparts_to_line(data=data, buffer=10, px_per_mm=4)
    assert type(results) == Polygon

def test_minimum_rotated_rectangle():
    polygon = GeometryMixin().bodyparts_to_polygon(np.array([[364, 308], [383, 323], [403, 335], [423, 351]]))
    results = GeometryMixin().minimum_rotated_rectangle(shape=polygon)
    assert type(results) is Polygon
    assert results.is_valid

def test_length():
    line = GeometryMixin().bodyparts_to_line(np.array([[0, 0], [0, 10], [0, 25], [0, 50]]))
    results = GeometryMixin().length(shape=line, pixels_per_mm=1.0)
    assert results == 50.0

def test_point_lineside():
    lines = np.array([[[25, 25], [25, 20]], [[15, 25], [15, 20]], [[15, 25], [50, 20]]]).astype(np.float32)
    points = np.array([[20, 0], [15, 20], [90, 0]]).astype(np.float32)
    results = GeometryMixin().point_lineside(lines=lines, points=points)
    assert np.array_equal(results, np.array([1., 0., 1.]))







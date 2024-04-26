import pytest
import numpy as np
from simba.roi_tools.ROI_analyzer import ROIAnalyzer
from simba.roi_tools.ROI_clf_calculator import ROIClfCalculator
from simba.roi_tools.ROI_directing_analyzer import DirectingROIAnalyzer
from simba.roi_tools.ROI_feature_analyzer import ROIFeatureCreator
from simba.roi_tools.ROI_size_calculations import rectangle_size_calc, circle_size_calc, polygon_size_calc
from simba.roi_tools.ROI_time_bin_calculator import ROITimebinCalculator


@pytest.fixture(params=['tests/data/test_projects/two_c57/project_folder/project_config.ini'])
def config_path_args(request):
    return request

@pytest.fixture(params=['tests/data/test_projects/two_c57/project_folder/project_config.ini',
                        'tests/data/test_projects/mouse_open_field/project_folder/project_config.ini',
                        'tests/data/test_projects/zebrafish/project_folder/project_config.ini'])
def all_config_path_args(request):
    return request


# @pytest.mark.parametrize("calculate_distances", [True, False])
# def test_roi_analyser(all_config_path_args, calculate_distances):
#     roi_analyzer = ROIAnalyzer(config_path=all_config_path_args.param,
#                                data_path=None,
#                                calculate_distances=calculate_distances)
#     roi_analyzer.run()
#     roi_analyzer.save()

def test_roi_clf_calculator(config_path_args):
    clf_ROI_analyzer = ROIClfCalculator(config_ini=config_path_args.param)
    clf_ROI_analyzer.run(behavior_list=['Attack', 'Sniffing'], ROI_dict_lists={'Rectangle': ['Rectangle_1']}, body_part_list=['Nose_1'], measurements=['Total time by ROI (s)', 'Started bouts by ROI (count)', 'Ended bouts by ROI (count)'])

# @pytest.mark.parametrize("config_path, settings", [('tests/data/test_projects/two_c57/project_folder/project_config.ini', {'body_parts': {'Simon': 'Ear_left_1', 'JJ': 'Ear_left_2'}, 'threshold': 0.4})])
# def test_roi_directing_analyzer(config_path, settings):
#     directing_analyzer = DirectingROIAnalyzer(config_path=config_path, settings=settings, )
#     directing_analyzer.run()

# @pytest.mark.parametrize("config_path, settings", [('tests/data/test_projects/two_c57/project_folder/project_config.ini', {'body_parts': {'Simon': 'Ear_left_1', 'JJ': 'Ear_left_2'}, 'threshold': 0.4})])
# def test_roi_feature_creator(config_path, settings):
#     roi_featurizer = ROIFeatureCreator(config_path=config_path, settings=settings)
#     roi_featurizer.run()

# def test_roi_movement_analyzer(all_config_path_args):
#     _ = ROIMovementAnalyzer(config_path=all_config_path_args.param)

@pytest.mark.parametrize("rectangle_dict, px_mm, expected_area", [({'height': 500, 'width': 500}, 10, 25), ({'height': 500, 'width': 500}, 5, 100)])
def test_rectangle_size_calc(rectangle_dict, px_mm, expected_area):
    results = rectangle_size_calc(rectangle_dict=rectangle_dict, px_mm=px_mm)
    assert int(results['area_cm']) == int(expected_area)

@pytest.mark.parametrize("circle_dict, px_mm, expected_area", [({'radius': 100}, 5, 12.57)])
def test_circle_size_calc(circle_dict, px_mm, expected_area):
    results = circle_size_calc(circle_dict=circle_dict, px_mm=px_mm)
    assert results['area_cm'] == expected_area

@pytest.mark.parametrize("polygon_dict, px_mm, expected_area", [({'vertices': np.array([[0, 2], [200, 98], [100, 876], [10, 702]])}, 5, 45.29)])
def test_polygon_size_calc(polygon_dict, px_mm, expected_area):
    results = polygon_size_calc(polygon_dict=polygon_dict, px_mm=px_mm)
    assert results['area_cm'] == expected_area

#
# @pytest.mark.parametrize("bin_length, threshold", [(10, 0.00), (1, 0.50)])
# def test_roi_timbin_calculator(config_path_args, bin_length, threshold):
#     timebin_calculator = ROITimebinCalculator(config_path=config_path_args.param, bin_length=bin_length, body_parts=['Nose_1'], threshold=threshold)
#     timebin_calculator.run()
#     timebin_calculator.save()

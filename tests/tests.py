__author__ = "Simon Nilsson", "JJ Choong"

from simba.BORIS_appender import BorisAppender
from simba.solomon_importer import SolomonImporter
from simba.clf_validator import ClassifierValidationClips
from simba.create_clf_log import ClfLogCreator
from simba.data_plotter import DataPlotter
from simba.Directing_animals_analyzer import DirectingOtherAnimalsAnalyzer
from simba.Directing_animals_visualizer import DirectingOtherAnimalsVisualizer
from simba.distance_plotter import DistancePlotter
from simba.ethovision_import import ImportEthovision
from simba.gantt_creator_mp import GanttCreator
from simba.interpolate_pose import Interpolate
from simba.Kleinberg_calculator import KleinbergCalculator
from simba.merge_videos import FrameMergerer
from simba.heat_mapper_clf import HeatMapperClf
from simba.probability_plot_creator_mp import TresholdPlotCreator
import pytest
from simba.rw_dfs import read_df
import pandas as pd
from simba.project_config_creator import ProjectConfigCreator
from simba.read_config_unit_tests import (check_int,
                                          check_str,
                                          check_float,
                                          read_config_entry,
                                          read_simba_meta_files,
                                          read_meta_file,
                                          check_file_exist_and_readable,
                                          read_config_file,
                                          insert_default_headers_for_feature_extraction,
                                          check_that_column_exist)
from simba.misc_tools import (get_video_meta_data,
                              check_multi_animal_status)
from simba.ROI_analyzer import ROIAnalyzer
from simba.ROI_directing_analyzer import DirectingROIAnalyzer
from simba.ROI_feature_analyzer import ROIFeatureCreator
from simba.ROI_feature_visualizer import ROIfeatureVisualizer
from simba.ROI_movement_analyzer import ROIMovementAnalyzer
from simba.ROI_plot_new import ROIPlot
from simba.run_model_new import RunModel





#from simba.dlc_multi_animal_importer import MADLC_Importer

@pytest.mark.parametrize("config_path, boris_path", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', 'test_data/multi_animal_dlc_two_c57/boris_import')])
def test_boris_import_use_case(config_path, boris_path):
    boris_appender = BorisAppender(config_path=config_path,
                                   boris_folder=boris_path)
    boris_appender.create_boris_master_file()
    boris_appender.append_boris()

@pytest.mark.parametrize("config_path, solomon_path", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', 'test_data/multi_animal_dlc_two_c57/solomon_import')])
def test_solomon_import_use_case(config_path, solomon_path):
    solomon_appender = SolomonImporter(config_path='test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini',
                                       solomon_dir='test_data/multi_animal_dlc_two_c57/solomon_import')
    solomon_appender.import_solomon()


@pytest.mark.parametrize("config_path, window, clf_name", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', 1, 'Attack')])
def test_validation_clips_import_use_case(config_path, window, clf_name):
      clf_validator = ClassifierValidationClips(config_path=config_path, window=window, clf_name=clf_name)
      clf_validator.create_clips()

@pytest.mark.parametrize("config_path, data_measures", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', ['Bout count', 'Total event duration']),
                                                        ('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', [])])
def test_create_clf_log_use_case(config_path, data_measures):
    clf_log_creator = ClfLogCreator(config_path=config_path,
                                         data_measures=data_measures)
    clf_log_creator.analyze_data()
    clf_log_creator.save_results()

@pytest.mark.parametrize("config_path", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini')])
def test_data_plotter_use_case(config_path):
    data_plotter = DataPlotter(config_path=config_path)
    data_plotter.process_movement()
    data_plotter.create_data_plots()

@pytest.mark.parametrize("config_path", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini')])
def test_directing_animal_analyzer(config_path):
    directing_animal_analyzer = DirectingOtherAnimalsAnalyzer(config_path=config_path)
    directing_animal_analyzer.process_directionality()
    directing_animal_analyzer.create_directionality_dfs()
    directing_animal_analyzer.save_directionality_dfs()
    directing_animal_analyzer.summary_statistics()

@pytest.mark.parametrize("config_path", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini')])
def test_directing_animal_visualizer(config_path):
    directing_animal_visualizer = DirectingOtherAnimalsVisualizer(config_path=config_path)
    directing_animal_visualizer.visualize_results()


@pytest.mark.parametrize("config_path, frame_setting, video_setting", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', False, True)])
def test_distance_plotter_use_case(config_path, frame_setting, video_setting):
    distance_plotter = DistancePlotter(config_path=config_path, frame_setting=frame_setting, video_setting=video_setting)
    distance_plotter.create_distance_plot()

@pytest.mark.parametrize("config_path, folder_path", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', 'test_data/multi_animal_dlc_two_c57/ethovision_import')])
def test_ethovision_import_use_case(config_path, folder_path):
    _ = ImportEthovision(config_path=config_path, folder_path=folder_path)

@pytest.mark.parametrize("config_path, frame_setting, video_setting", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', False, True)])
def test_gantt_creator_use_case(config_path, frame_setting, video_setting):
    gantt_creator = GanttCreator(config_path=config_path, frame_setting=frame_setting, video_setting=video_setting)
    gantt_creator.create_gannt()

@pytest.mark.parametrize("config_file_path, file_path, interpolation_method", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', 'test_data/multi_animal_dlc_two_c57/project_folder/csv/input_csv/Together_1.csv', 'Body-parts: Nearest'),
                                                                               ('test_data/mouse_open_field/project_folder/project_config.ini', 'test_data/mouse_open_field/project_folder/csv/input_csv/SI_DAY3_308_CD1_PRESENT.csv', 'Body-parts: Quadratic')])
def test_interpolate_use_case(config_file_path, file_path, interpolation_method):
    in_file = read_df(file_path, 'csv')
    body_part_interpolator = Interpolate(config_file_path=config_file_path, in_file=in_file)
    body_part_interpolator.detect_headers()
    body_part_interpolator.fix_missing_values(method_str=interpolation_method)
    body_part_interpolator.reorganize_headers()

@pytest.mark.parametrize("config_path, classifier_names, sigma, gamma, hierarchy", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', ['Attack'], 2, 0.3, 2)])
def test_kleinberg_use_case(config_path, classifier_names, sigma, gamma, hierarchy):
    kleinberg_calculator = KleinbergCalculator(config_path=config_path, classifier_names=classifier_names, sigma=sigma, gamma=gamma, hierarchy=hierarchy)
    kleinberg_calculator.perform_kleinberg()

@pytest.mark.parametrize("config_path, frame_types, frame_setting, video_setting, output_height", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', ['Classifications', 'Gantt', 'Path'], False, True, 1280)])
def test_frame_merger_use_case(config_path, frame_types, frame_setting, video_setting, output_height):
    _ = FrameMergerer(config_path=config_path, frame_types=frame_types, frame_setting=frame_setting, video_setting=video_setting, output_height=output_height)

@pytest.mark.parametrize("config_path, final_img_setting, video_setting, frame_setting, bin_size, palette, bodypart, clf_name, max_scale", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', False, True, False, 50, 'jet', 'Nose_1', 'Attack', 20)])
def test_heat_mapper_clf_use_case(config_path, final_img_setting, video_setting, frame_setting, bin_size, palette, bodypart, clf_name, max_scale):
    heat_mapper_clf = HeatMapperClf(config_path=config_path, final_img_setting=final_img_setting, video_setting=video_setting, frame_setting=frame_setting, bin_size=bin_size, palette=palette, bodypart=bodypart, clf_name=clf_name, max_scale=max_scale)
    heat_mapper_clf.create_heatmaps()

@pytest.mark.parametrize("config_path, frame_setting, video_setting, clf_name", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', False, True, 'Attack')])
def test_probability_plotter_use_case(config_path, frame_setting, video_setting, clf_name):
    plot_creator = TresholdPlotCreator(config_path=config_path, frame_setting=frame_setting, video_setting=video_setting, clf_name=clf_name)
    plot_creator.create_plot()

@pytest.mark.parametrize("project_path, project_name, target_list, pose_estimation_bp_cnt, body_part_config_idx, animal_cnt, file_type", [('test_directory', 'test_name', ['test_clf'], '16', 9, 2, 'csv')])
def test_project_config_create_use_case(project_path, project_name, target_list, pose_estimation_bp_cnt, body_part_config_idx, animal_cnt, file_type):
    _ = ProjectConfigCreator(project_path=project_path, project_name=project_name, target_list=target_list, pose_estimation_bp_cnt=pose_estimation_bp_cnt, body_part_config_idx=body_part_config_idx, animal_cnt=animal_cnt, file_type=file_type)

@pytest.mark.parametrize("name, value, max_value, min_value", [('test_1', 1, 2, 0),
                                                               ('test_2', 2.0, 5.0, 1.0),
                                                               ('test_2', 2.0, None, None)])
def test_check_int_use_case(name, value, max_value, min_value):
    _ = check_int(name=value, value=value, max_value=max_value, min_value=min_value)

@pytest.mark.parametrize("name, value, options, allow_blank", [('test_1', 'test_str', ('test_str', 'test_2'), True),
                                                               ('test_1', 'test_str', (), True)])
def test_check_str_use_case(name, value, options, allow_blank):
    _ = check_str(name=name, value=value, options=options, allow_blank=allow_blank)

@pytest.mark.parametrize("name, value, max_value, min_value", [('test_1', 1, 2, 0),
                                                               ('test_2', 2.0, 5.0, 1.0),
                                                               ])
def test_check_float_use_case(name, value, max_value, min_value):
    _ = check_float(name=name, value=value, max_value=max_value, min_value=min_value)

@pytest.mark.parametrize("config, section, option, data_type, default_value, options", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', 'General settings', 'animal_no', 'int', 2, None)])
def test_read_config_entry_use_case(config, section, option, data_type, default_value, options):
    config = read_config_file(config)
    _ = read_config_entry(config=config, section=section, option=option, data_type=data_type, default_value=default_value)

@pytest.mark.parametrize("folder_path", [('test_data/multi_animal_dlc_two_c57/project_folder/configs')])
def test_read_simba_meta_files(folder_path):
    meta_file_lst = read_simba_meta_files(folder_path=folder_path)
    assert isinstance(meta_file_lst, list)
    for i in meta_file_lst:
        assert isinstance(i, str)
        d = read_meta_file(i)
        assert isinstance(d, dict)
        assert len(d.keys()) > 0
        check_file_exist_and_readable(file_path=i)

@pytest.mark.parametrize("df_path", [('test_data/multi_animal_dlc_two_c57/project_folder/csv/outlier_corrected_movement_location/Together_1.csv')])
def test_insert_default_headers_for_feature_extraction(df_path):
    df = read_df(df_path, 'csv')
    header_lst = [str(x) for x in range(0, 48)]
    df = insert_default_headers_for_feature_extraction(df=df, headers=header_lst, pose_config='16 body-parts', filename='test')
    assert isinstance(df, pd.DataFrame)

@pytest.mark.parametrize("df_path, column_name, file_name", [('test_data/multi_animal_dlc_two_c57/project_folder/csv/outlier_corrected_movement_location/Together_1.csv', 'Ear_left_1_x', 'Test')])
def test_check_that_column_exist(df_path, column_name, file_name):
    df = read_df(df_path, 'csv')
    _ = check_that_column_exist(df, column_name, file_name)

@pytest.mark.parametrize("ini_path, data_path, calculate_distances", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', 'outlier_corrected_movement_location', True)])
def test_ROIAnalyzer(ini_path, data_path, calculate_distances):
    roi_analyzer = ROIAnalyzer(ini_path=ini_path, data_path=data_path, calculate_distances=calculate_distances)
    roi_analyzer.read_roi_dfs()
    roi_analyzer.analyze_ROIs()
    roi_analyzer.save_data()

@pytest.mark.parametrize("ini_path, data_path, calculate_distances", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', 'outlier_corrected_movement_location', True)])
def test_ROIAnalyzer(ini_path, data_path, calculate_distances):
    roi_analyzer = ROIAnalyzer(ini_path=ini_path, data_path=data_path, calculate_distances=calculate_distances)
    roi_analyzer.read_roi_dfs()
    roi_analyzer.analyze_ROIs()
    roi_analyzer.save_data()

@pytest.mark.parametrize("config_path ", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini')])
def test_DirectingROIAnalyzer(config_path):
    roi_directing_calculator = DirectingROIAnalyzer(config_path=config_path)
    roi_directing_calculator.calc_directing_to_ROIs()

@pytest.mark.parametrize("config_path ", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini')])
def test_ROIFeatureCreator(config_path):
    roi_featurizer = ROIFeatureCreator(config_path=config_path)
    roi_featurizer.analyze_ROI_data()
    roi_featurizer.save_new_features_files()

@pytest.mark.parametrize("config_path, video_name", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', 'Together_1.avi')])
def test_ROIfeatureVisualizer(config_path, video_name):
    roi_feature_visualizer = ROIfeatureVisualizer(config_path=config_path, video_name=video_name)
    roi_feature_visualizer.create_visualization()

@pytest.mark.parametrize("config_path", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini')])
def test_ROIMovementAnalyzer(config_path):
     _ = ROIMovementAnalyzer(config_path=config_path)

@pytest.mark.parametrize("config_path, video_name", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', 'Together_1.avi')])
def test_ROIPlot(config_path, video_name):
    roi_visualizer = ROIPlot(ini_path=config_path, video_path=video_name)
    roi_visualizer.insert_data()
    roi_visualizer.visualize_ROI_data()

@pytest.mark.parametrize("config_path", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini')])
def test_RunModel(config_path):
    inferencer = RunModel(config_path=config_path)
    inferencer.run_models()

@pytest.mark.parametrize("video_path", [('test_data/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT.mp4'),
                                        ('test_data/multi_animal_dlc_two_c57/project_folder/videos/Together_1.avi')])
def test_get_video_meta_data(video_path):
    video_data = get_video_meta_data(video_path=video_path)

@pytest.mark.parametrize("config_path, no_animals", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', 2),
                                                     ('test_data/mouse_open_field/project_folder/project_config.ini', 1),
                                                     ('tests/test_data/two_white_mice_OF/project_folder/project_config.ini', 2)])
def test_check_multi_animal_status(config_path, no_animals):
    config = read_config_file(config_path)
    status, id_lst = check_multi_animal_status(config, no_animals)
    assert isinstance(status, bool)
    assert isinstance(id_lst, list)
    assert len(id_lst) > 0
    assert 'None' not in id_lst









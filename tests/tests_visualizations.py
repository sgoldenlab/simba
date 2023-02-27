import pytest
from simba.clf_validator import ClassifierValidationClips
from simba.data_plotter import DataPlotter
from simba.Directing_animals_visualizer import DirectingOtherAnimalsVisualizer
from simba.misc_tools import get_video_meta_data
from simba.misc_tools import find_files_of_filetypes_in_directory
from simba.read_config_unit_tests import (check_file_exist_and_readable,
                                          check_if_dir_exists)
from simba.train_model_functions import get_all_clf_names
from simba.distance_plotter import DistancePlotterSingleCore
import os, shutil

class TestVisualizations(object):

    @pytest.fixture(params=['test_data/visualization_tests/project_folder/project_config.ini'])
    def config_path_arg(self, request):
        return request

    @pytest.fixture(params=[True, False])
    def video_setting_args(self, request):
        return request

    @pytest.fixture(params=[True, False])
    def final_img_args(self, request):
        return request

    @pytest.fixture(params=[True, False])
    def frame_setting_args(self, request):
        return request

    @pytest.fixture()
    def distance_plot_args(self):
        args = {}
        args['style_attr'] = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8, 'opacity': 0.5, 'y_max': 'auto'}
        args['line_attr'] = {0: ['Nose_1', 'Nose_2', 'Dark-red']}
        args['data_paths'] = ['test_data/two_C57_madlc/project_folder/csv/outlier_corrected_movement_location/Together_1.csv']
        return args

    @pytest.fixture()
    def data_plotter_args(self):
        args = {}
        args['style_attr'] = {'bg_color': 'White', 'header_color': 'Black', 'font_thickness': 1, 'size': (640, 480), 'data_accuracy': 2}
        args['data_paths'] = ['test_data/visualization_tests/project_folder/csv/machine_results/Together_1.csv']
        args['body_part_attr'] = [['Ear_left_1', 'Grey'], ['Ear_right_2', 'Red']]
        return args

    @pytest.fixture()
    def directing_animal_args(self):
        args = {}
        args['style_attr'] = {'Show_pose': True, 'Pose_circle_size': 3, "Direction_color": 'Random', 'Direction_thickness': 4, 'Highlight_endpoints': True, 'Polyfill': True}
        args['data_path'] = 'test_data/visualization_tests/project_folder/csv/outlier_corrected_movement_location/Together_1.csv'
        return args

    def perform_video_checks(self, img_creator: object):
        check_file_exist_and_readable(file_path=img_creator.video_save_path)
        assert os.path.getsize(img_creator.video_save_path) > 0
        os.remove(img_creator.video_save_path)

    def perform_frame_checks(self, img_creator: object):
        check_if_dir_exists(in_dir=img_creator.frame_save_path)
        files = find_files_of_filetypes_in_directory(directory=img_creator.frame_save_path, extensions=['.png'])
        for file_path in files:
            assert os.path.getsize(file_path) > 0
        shutil.rmtree(img_creator.frame_save_path)


    def test_data_plotter(self,
                          config_path_arg,
                          video_setting_args,
                          frame_setting_args,
                          data_plotter_args):

        if video_setting_args.param or frame_setting_args.param:
            data_plotter = DataPlotter(config_path=config_path_arg.param,
                                       style_attr=data_plotter_args['style_attr'],
                                       body_part_attr=data_plotter_args['body_part_attr'],
                                       data_paths=data_plotter_args['data_paths'],
                                       video_setting=video_setting_args.param,
                                       frame_setting=frame_setting_args.param)

            data_plotter.process_movement()
            data_plotter.create_data_plots()
            if video_setting_args.param:
                self.perform_video_checks(img_creator=data_plotter)
            if frame_setting_args.param:
                self.perform_frame_checks(img_creator=data_plotter)

    def test_directing_animal_visualizer(self,
                                         config_path_arg,
                                         directing_animal_args):
        visualizer = DirectingOtherAnimalsVisualizer(config_path=config_path_arg.param,
                                                     style_attr=directing_animal_args['style_attr'],
                                                     data_path=directing_animal_args['data_path'])
        visualizer.visualize_results()
        self.perform_video_checks(img_creator=visualizer)


    def test_distance_plotter(self,
                              config_path_arg,
                              video_setting_args,
                              frame_setting_args,
                              final_img_args,
                              distance_plot_args
                              ):
        if final_img_args.param or video_setting_args.param or frame_setting_args.param:
            plotter = DistancePlotterSingleCore(config_path=config_path_arg.param,
                                                    frame_setting=frame_setting_args.param,
                                                    video_setting=video_setting_args.param,
                                                    style_attr=distance_plot_args['style_attr'],
                                                    files_found=distance_plot_args['data_paths'],
                                                    line_attr=distance_plot_args['line_attr'],
                                                    final_img=final_img_args.param)
            plotter.create_distance_plot()


    def test_validation_clips_import(self,
                                     config_path_arg):
        clf_validator = ClassifierValidationClips(config_path=config_path_arg.param,
                                                  window=5,
                                                  clf_name='Attack',
                                                  text_clr=(255,0,255),
                                                  clips=True,
                                                  concat_video=False)
        clf_validator.create_clips()













    # def test_validation_clips_import_use_case(self, data_plotter_args):
    #     print(data_plotter_args)
    #     clf_validator = ClassifierValidationClips(config_path=config_path,
    #                                               window=window,
    #                                               clf_name=clf_name,
    #                                               text_clr=text_clr,
    #                                               clips=clips,
    #                                               concat_video=concat)
    #     # clf_validator.create_clips()








    # @pytest.fixture
    # def data_plotter(self):
    #     data_plotter_style_attr = [{'bg_color': 'White', 'header_color': 'Black', 'font_thickness': 1, 'size': (640, 480), 'data_accuracy': 2}]
    #     data_plotter_paths = [['test_data/two_C57_madlc/project_folder/csv/machine_results/Together_1.csv']]
    #     data_plotter_body_part_attr = [[['Ear_left_1', 'Grey'], ['Ear_right_2', 'Red']]]
    #     data_plotter_video_settings = [True, False]
    #     data_plotter_frame_settings = [False, True]
    #     return data_plotter_style_attr, data_plotter_paths, data_plotter_body_part_attr, data_plotter_video_settings, data_plotter_frame_settings
    #




import pytest

from simba.plotting.distance_plotter import DistancePlotterSingleCore
from simba.plotting.distance_plotter_mp import DistancePlotterMultiCore


class TestDistancePlotter(object):

    @pytest.fixture(params=['/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/two_c57/project_folder/project_config.ini'])
    def config_path_args(self, request):
        return request

    @pytest.fixture(params=[['/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/two_c57/project_folder/csv/outlier_corrected_movement_location/Together_1.csv']])
    def data_path_args(self, request):
        return request

    @pytest.fixture(params=[640])
    def width_args(self, request):
        return request

    @pytest.fixture(params=[480])
    def height_args(self, request):
        return request

    @pytest.fixture(params=[6])
    def line_width_args(self, request):
        return request

    @pytest.fixture(params=[8])
    def font_size_args(self, request):
        return request

    @pytest.fixture(params=['auto'])
    def y_max_args(self, request):
        return request

    @pytest.fixture(params=[True])
    def final_img_args(self, request):
        return request

    @pytest.fixture(params=[False])
    def frame_setting_args(self, request):
        return request

    @pytest.fixture(params=[True])
    def video_setting_args(self, request):
        return request

    @pytest.fixture(params=[0.9])
    def opacity_args(self, request):
        return request

    @pytest.fixture(params=[1])
    def core_cnt_args(self, request):
        return request

    @pytest.fixture(params=[{0: ['Center_1', 'Center_2', 'Green'], 1: ['Ear_left_2', 'Ear_left_1', 'Red']}])
    def line_attr_args(self, request):
        return request

    def test_distance_plotter_single_core(self,
                                          data_path_args,
                                          config_path_args,
                                          width_args,
                                          height_args,
                                          line_width_args,
                                          font_size_args,
                                          y_max_args,
                                          opacity_args,
                                          line_attr_args,
                                          final_img_args,
                                          frame_setting_args,
                                          video_setting_args):

        style_attr = {'width': width_args.param, 'height': height_args.param, 'line width': line_width_args.param, 'font size': font_size_args.param, 'y_max': y_max_args.param, 'opacity': opacity_args.param}

        distance_plotter = DistancePlotterSingleCore(config_path=config_path_args.param,
                                                     frame_setting=False,
                                                     video_setting=True,
                                                     style_attr=style_attr,
                                                     final_img=True,
                                                     files_found=data_path_args.param,
                                                     line_attr=line_attr_args.param)
        distance_plotter.run()


    def test_distance_plotter_single_multicore(self,
                                               core_cnt_args,
                                               data_path_args,
                                               config_path_args,
                                               width_args,
                                               height_args,
                                               line_width_args,
                                               font_size_args,
                                               y_max_args,
                                               opacity_args,
                                               line_attr_args,
                                               final_img_args,
                                               frame_setting_args,
                                               video_setting_args):

        style_attr = {'width': width_args.param, 'height': height_args.param, 'line width': line_width_args.param, 'font size': font_size_args.param, 'y_max': y_max_args.param, 'opacity': opacity_args.param}

        distance_plotter = DistancePlotterMultiCore(config_path=config_path_args.param,
                                                    frame_setting=False,
                                                    video_setting=True,
                                                    style_attr=style_attr,
                                                    final_img=True,
                                                    files_found=data_path_args.param,
                                                    line_attr=line_attr_args.param,
                                                    core_cnt=core_cnt_args.param)
        distance_plotter.run()

# style_attr = {'width': ,
#               'height': ,
#               'line width': ,
#               'font size': 8,
#               'y_max': '',
#               'opacity': }
# line_attr = {0: ['Center_1', 'Center_2', 'Green'], 1: ['Ear_left_2', 'Ear_left_1', 'Red']}
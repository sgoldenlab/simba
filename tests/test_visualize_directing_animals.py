import pytest
from simba.plotting.directing_animals_visualizer import DirectingOtherAnimalsVisualizer
from simba.plotting.directing_animals_visualizer_mp import DirectingOtherAnimalsVisualizerMultiprocess


class TestDirectingAnimalsVisualizer(object):

    @pytest.fixture(params=['/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/two_c57/project_folder/csv/outlier_corrected_movement_location/Together_1.csv'])
    def data_path_args(self, request):
        return request

    @pytest.fixture(params=['/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/two_c57/project_folder/project_config.ini'])
    def config_path_args(self, request):
        return request

    @pytest.fixture(params=[True, False])
    def show_pose_args(self, request):
        return request

    @pytest.fixture(params=[3])
    def circle_size_args(self, request):
        return request

    @pytest.fixture(params=[5])
    def core_cnt_args(self, request):
        return request

    @pytest.fixture(params=['Random', 'Orange'])
    def direction_color_args(self, request):
        return request

    @pytest.fixture(params=[4])
    def direction_thickness_args(self, request):
        return request

    @pytest.fixture(params=[True, False])
    def highlight_endpoints_args(self, request):
        return request

    @pytest.fixture(params=[True, False])
    def polyfill_args(self, request):
        return request

    def test_directing_animal_visualizer_single_core(self,
                                                     data_path_args,
                                                     config_path_args,
                                                     show_pose_args,
                                                     circle_size_args,
                                                     direction_color_args,
                                                     direction_thickness_args,
                                                     highlight_endpoints_args,
                                                     polyfill_args):

        style_attr = {'Show_pose': show_pose_args.param,
                      'Pose_circle_size': circle_size_args.param,
                      "Direction_color": direction_color_args.param,
                      'Direction_thickness': direction_thickness_args.param,
                      'Highlight_endpoints': highlight_endpoints_args.param,
                      'Polyfill': polyfill_args.param}

        single_core_visualizer = DirectingOtherAnimalsVisualizer(config_path=config_path_args.param,
                                                                 style_attr=style_attr,
                                                                 data_path=data_path_args.param)
        single_core_visualizer.run()

    def test_directing_animal_visualizer_multi_core(self,
                                                    data_path_args,
                                                    core_cnt_args,
                                                    config_path_args,
                                                    show_pose_args,
                                                    circle_size_args,
                                                    direction_color_args,
                                                    direction_thickness_args,
                                                    highlight_endpoints_args,
                                                    polyfill_args):

        style_attr = {'Show_pose': show_pose_args.param,
                      'Pose_circle_size': circle_size_args.param,
                      "Direction_color": direction_color_args.param,
                      'Direction_thickness': direction_thickness_args.param,
                      'Highlight_endpoints': highlight_endpoints_args.param,
                      'Polyfill': polyfill_args.param}

        multi_core_visualizer = DirectingOtherAnimalsVisualizerMultiprocess(config_path=config_path_args.param,
                                                                            style_attr=style_attr,
                                                                            data_path=data_path_args.param,
                                                                            core_cnt=core_cnt_args.param)
        multi_core_visualizer.run()
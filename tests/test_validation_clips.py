import pytest
from simba.plotting.clf_validator import ClassifierValidationClips

class TestValidationClips(object):

    @pytest.fixture(params=['tests/data/test_projects/two_c57/project_folder/project_config.ini'])
    def config_path_args(self, request):
        return request

    @pytest.fixture(params=[['tests/data/test_projects/two_c57/project_folder/csv/machine_results/Together_1.csv']])
    def data_path_arg(self, request):
        return request

    @pytest.fixture(params=['Attack'])
    def clf_name_args(self, request):
        return request

    @pytest.fixture(params=[2])
    def window_args(self, request):
        return request

    @pytest.fixture(params=[True, False])
    def clip_args(self, request):
        return request

    @pytest.fixture(params=[True])
    def concat_args(self, request):
        return request

    @pytest.fixture(params=[(255, 0, 0)])
    def highlight_clr_args(self, request):
        return request

    @pytest.fixture(params=[0.5])
    def video_speed_args(self, request):
        return request

    @pytest.fixture(params=[(0, 0, 255)])
    def text_clr_args(self, request):
        return request


    def test_classifier_validation_clips(self,
                                         config_path_args,
                                         window_args,
                                         clf_name_args,
                                         data_path_arg,
                                         clip_args,
                                         concat_args,
                                         highlight_clr_args,
                                         video_speed_args,
                                         text_clr_args):

        clip_creator = ClassifierValidationClips(config_path=config_path_args.param,
                                                 window=window_args.param,
                                                 clf_name=clf_name_args.param,
                                                 data_paths=data_path_arg.param,
                                                 clips=clip_args.param,
                                                 concat_video=concat_args.param,
                                                 highlight_clr=highlight_clr_args.param,
                                                 video_speed=video_speed_args.param,
                                                 text_clr=text_clr_args.param)

        clip_creator.run()





# test = ClassifierValidationClips(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                  window=10,
#                                  clf_name='Attack',
#                                  clips=False,
#                                  concat_video=True,
#                                  highlight_clr=(255, 0, 0),
#                                  video_speed=0.5,
#                                  text_clr=(0, 0, 255))
# test.create_clips()

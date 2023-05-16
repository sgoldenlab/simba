__author__ = "Simon Nilsson"

import os
import cv2
import platform
import multiprocessing
import functools
from simba.utils.enums import Paths
from simba.utils.read_write import read_df, get_video_meta_data, get_fn_ext, concatenate_videos_in_folder
from simba.utils.checks import check_file_exist_and_readable
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.printing import stdout_success, SimbaTimer
from simba.utils.warnings import NoDataFoundWarning
from simba.utils.lookups import get_color_dict
from simba.utils.data import create_color_palettes
from simba.data_processors.directing_other_animals_calculator import DirectingOtherAnimalsAnalyzer
from simba.mixins.config_reader import ConfigReader

class DirectingOtherAnimalsVisualizerMultiprocess(ConfigReader, PlottingMixin):
    """
    Class for visualizing when animals are directing towards body-parts of other animals using multiprocessing.

    .. important::
       Requires the pose-estimation data for the left ear, right ears and nose of individual animals.

    .. note::
        `Example of expected output <https://www.youtube.com/watch?v=d6pAatreb1E&list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl&index=22`_.


    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter str video_name: Video to visualize directionality for in the project_folder/videos directory (e.g., ``My_video.mp4``)
    :parameter dict style_attr: Video style attribitions (colors and sizes etc.)
    :parameter dict core_cnt: How many cores to use to create the video.


    Examples
    -----
    >>> style_attr = {'Show_pose': True, 'Pose_circle_size': 3, "Direction_color": 'Random', 'Direction_thickness': 4, 'Highlight_endpoints': True}
    >>> directing_visualizer = DirectingOtherAnimalsVisualizerMultiprocess(config_path='project_folder/project_config.ini', video_name='Testing_Video_3.mp4', style_attr=style_attr)
    >>> directing_visualizer.run()
    """

    def __init__(self,
                 config_path: str,
                 data_path: str,
                 style_attr: dict,
                 core_cnt: int):

        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)

        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        self.data_path = data_path
        _, self.video_name, _ = get_fn_ext(self.data_path)
        self.direction_analyzer = DirectingOtherAnimalsAnalyzer(config_path=config_path)
        self.direction_analyzer.process_directionality()
        self.direction_analyzer.create_directionality_dfs()
        self.style_attr, self.pose_colors, self.core_cnt = style_attr, [], core_cnt
        self.colors = get_color_dict()
        if self.style_attr['Show_pose']:
            self.pose_colors = create_color_palettes(self.animal_cnt, int(len(self.x_cols) + 1))
        if self.style_attr['Direction_color'] == 'Random':
            self.direction_colors = create_color_palettes(1, int(self.animal_cnt**2))
        else:
            self.direction_colors = [self.colors[self.style_attr['Direction_color']]]
        self.data_dict = self.direction_analyzer.directionality_df_dict
        self.video_path = self.find_video_of_file(self.video_dir, self.video_name)
        self.save_directory = os.path.join(self.project_path, Paths.DIRECTING_BETWEEN_ANIMALS_OUTPUT_PATH.value)
        if not os.path.exists(self.save_directory): os.makedirs(self.save_directory)
        self.data_path = os.path.join(self.outlier_corrected_dir, self.video_name + '.' + self.file_type)
        check_file_exist_and_readable(file_path=self.data_path)
        print(f'Processing video {self.video_name}...')

    def run(self):
        """
        Method to create directionality videos. Results are stored in
        the `project_folder/frames/output/ROI_directionality_visualize` directory of the SimBA project

        Returns
        ----------
        None
        """


        self.data_df = read_df(self.data_path, file_type=self.file_type)
        self.save_path = os.path.join(self.save_directory, self.video_name + '.mp4')
        self.save_temp_path = os.path.join(self.save_directory, 'temp')
        if os.path.exists(self.save_temp_path):
            self.remove_a_folder(folder_dir=self.save_temp_path)
        os.makedirs(self.save_temp_path)
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_meta_data = get_video_meta_data(self.video_path)
        self.video_data = self.data_dict[self.video_name]
        if self.video_name in list(self.video_data['Video']):
            self.__create_video()
        else:
            NoDataFoundWarning(msg=f'SimBA skipping video {self.video_name}: No animals are directing each other in the video.')

    def __create_video(self):
        video_timer = SimbaTimer()
        video_timer.start_timer()
        data_arr, frm_per_core = self.split_and_group_df(df=self.data_df, splits=self.core_cnt, include_split_order=True)
        print('Creating ROI images, multiprocessing (determined chunksize: {}, cores: {})...'.format(str(self.multiprocess_chunksize), str(self.core_cnt)))
        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
            constants = functools.partial(self.directing_animals_mp,
                                          directionality_data=self.video_data,
                                          video_meta_data=self.video_meta_data,
                                          style_attr=self.style_attr,
                                          save_temp_dir=self.save_temp_path,
                                          video_path=self.video_path,
                                          bp_names=self.animal_bp_dict,
                                          colors=self.direction_colors)
            for cnt, result in enumerate(pool.imap(constants, data_arr, chunksize=self.multiprocess_chunksize)):
                print('Image {}/{}, Video {}...'.format(str(int(frm_per_core * (result + 1))), str(len(self.data_df)), self.video_name))

            concatenate_videos_in_folder(in_folder=self.save_temp_path, save_path=self.save_path, video_format='mp4')
            video_timer.stop_timer()
            pool.terminate()
            pool.join()
            self.timer.stop_timer()
            stdout_success(msg=f'Video {self.video_name} created. Video saved at {self.save_path}', elapsed_time=self.timer.elapsed_time_str)

# style_attr = {'Show_pose': True,
#               'Pose_circle_size': 3,
#               "Direction_color": 'Random',
#               'Direction_thickness': 4,
#               'Highlight_endpoints': True,
#               'Polyfill': True}
# test = DirectingOtherAnimalsVisualizerMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/project_config.ini',
#                                                    data_path='/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/csv/outlier_corrected_movement_location/Testing_Video_3.csv',
#                                                    style_attr=style_attr,
#                                                    core_cnt=5)
#
# test.run()


# style_attr = {'Show_pose': True, 'Pose_circle_size': 3, "Direction_color": 'Random', 'Direction_thickness': 4, 'Highlight_endpoints': True, 'Polyfill': True}
# test = DirectingOtherAnimalsVisualizerMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                        data_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location/Together_1.csv',
#                                        style_attr=style_attr,
#                                                    core_cnt=5)
#
# test.run()

__author__ = "Simon Nilsson"

import pandas as pd
import os
import cv2
import numpy as np
import random

from simba.data_processors.directing_other_animals_calculator import DirectingOtherAnimalsAnalyzer
from simba.utils.enums import Formats
from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.printing import stdout_success, SimbaTimer
from simba.utils.warnings import NoDataFoundWarning
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import read_df, get_video_meta_data, get_fn_ext
from simba.utils.data import create_color_palettes


class DirectingOtherAnimalsVisualizer(ConfigReader, PlottingMixin):
    """
    Visualize when animals are directing towards body-parts of other animals.

    .. important::
       Requires the pose-estimation data for the left ear, right ears and nose of individual animals.
       For better runtime, use :meth:`simba.plotting.Directing_animals_visualizer.DirectingOtherAnimalsVisualizerMultiprocess`.

    .. note::
       `Example of expected output <https://www.youtube.com/watch?v=d6pAatreb1E&list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl&index=22`_.
        `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-data-tables`>_.

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter str data_path: path to data file
    :parameter dict style_attr: Visualisation attributes (colors and sizes etc.)


    Examples
    -----
    >>> style_attr = {'Show_pose': True, 'Pose_circle_size': 3, "Direction_color": 'Random', 'Direction_thickness': 4, 'Highlight_endpoints': True}
    >>> _ = DirectingOtherAnimalsVisualizer(config_path='/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/project_config.ini', video_name='Testing_Video_3.mp4', style_attr=style_attr).run()
    """


    def __init__(self,
                 config_path: str,
                 data_path: str,
                 style_attr: dict):

        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)

        self.data_path = data_path
        _, self.video_name, _ = get_fn_ext(self.data_path)
        self.direction_analyzer = DirectingOtherAnimalsAnalyzer(config_path=config_path)
        self.direction_analyzer.process_directionality()
        self.direction_analyzer.create_directionality_dfs()
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        self.style_attr, self.pose_colors = style_attr, []
        self.colors = get_color_dict()
        if self.style_attr['Show_pose']:
            self.pose_colors = create_color_palettes(self.animal_cnt, int(len(self.x_cols) + 1))
        if self.style_attr['Direction_color'] == 'Random':
            self.direction_colors = create_color_palettes(1, int(self.animal_cnt**2))
        else:
            self.direction_colors = [self.colors[self.style_attr['Direction_color']]]
        self.data_dict = self.direction_analyzer.directionality_df_dict
        self.video_path = self.find_video_of_file(video_dir=self.video_dir, filename=self.video_name)
        if not os.path.exists(self.directing_animals_video_output_path): os.makedirs(self.directing_animals_video_output_path)
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
        self.video_save_path = os.path.join(self.directing_animals_video_output_path, self.video_name + '.mp4')
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_meta_data = get_video_meta_data(self.video_path)
        self.video_data = self.data_dict[self.video_name]
        self.writer = cv2.VideoWriter(self.video_save_path, self.fourcc, self.video_meta_data['fps'], (self.video_meta_data['width'], self.video_meta_data['height']))
        if self.video_name in list(self.video_data['Video']):
            self.__create_video()
        else:
            NoDataFoundWarning(msg=f'SimBA skipping video {self.video_name}: No animals are directing each other in the video.')
        self.timer.stop_timer()


    def __draw_individual_lines(self,
                                animal_img_data:pd.DataFrame,
                                img: np.array):

        color = self.direction_colors[0]
        for cnt, (i, r) in enumerate(animal_img_data.iterrows()):
            if self.style_attr['Direction_color'] == 'Random':
                color = random.sample(self.direction_colors[0], 1)[0]
            cv2.line(img, (int(r['Eye_x']), int(r['Eye_y'])), (int(r['Animal_2_bodypart_x']), int(r['Animal_2_bodypart_y'])), color, self.style_attr['Direction_thickness'])
            if self.style_attr['Highlight_endpoints']:
                cv2.circle(img, (int(r['Eye_x']), int(r['Eye_y'])), self.style_attr['Pose_circle_size'] + 2, color, self.style_attr['Pose_circle_size'])
                cv2.circle(img, (int(r['Animal_2_bodypart_x']), int(r['Animal_2_bodypart_y'])), self.style_attr['Pose_circle_size'] + 1, color, self.style_attr['Pose_circle_size'])

        return img


    def __create_video(self):
        img_cnt = 0
        video_timer = SimbaTimer()
        video_timer.start_timer()
        color = self.direction_colors[0]
        while (self.cap.isOpened()):
            ret, img = self.cap.read()
            try:
                if ret:
                    if self.style_attr['Show_pose']:
                        bp_data = self.data_df.iloc[img_cnt]
                        for cnt, (animal_name, animal_bps) in enumerate(self.animal_bp_dict.items()):
                            for bp in zip(animal_bps['X_bps'], animal_bps['Y_bps']):
                                x_bp, y_bp = bp_data[bp[0]], bp_data[bp[1]]
                                cv2.circle(img, (int(x_bp), int(y_bp)), self.style_attr['Pose_circle_size'], self.animal_bp_dict[animal_name]['colors'][cnt], self.style_attr['Direction_thickness'])

                    if img_cnt in list(self.video_data['Frame_#'].unique()):
                        img_data = self.video_data[self.video_data['Frame_#'] == img_cnt]
                        unique_animals = img_data['Animal_1'].unique()
                        for animal in unique_animals:
                            animal_img_data = img_data[img_data['Animal_1'] == animal].reset_index(drop=True)
                            if self.style_attr['Polyfill']:
                                convex_hull_arr = animal_img_data.loc[0, ['Eye_x', 'Eye_y']].values.reshape(-1, 2)
                                for animal_2 in animal_img_data['Animal_2'].unique():
                                    convex_hull_arr = np.vstack((convex_hull_arr, animal_img_data[['Animal_2_bodypart_x', 'Animal_2_bodypart_y']][animal_img_data['Animal_2'] == animal_2].values)).astype('int')
                                    convex_hull_arr = np.unique(convex_hull_arr, axis=0)
                                    if convex_hull_arr.shape[0] >= 3:
                                        if self.style_attr['Direction_color'] == 'Random':
                                            color = random.sample(self.direction_colors[0], 1)[0]
                                        cv2.fillPoly(img, [convex_hull_arr], color)
                                    else:
                                        img = self.__draw_individual_lines(animal_img_data=animal_img_data, img=img)

                            else:
                                img = self.__draw_individual_lines(animal_img_data=animal_img_data, img=img)

                    img_cnt += 1
                    self.writer.write(np.uint8(img))
                    print('Frame: {} / {}. Video: {}'.format(str(img_cnt), str(self.video_meta_data['frame_count']),
                                                                     self.video_name))
                else:
                    self.cap.release()
                    self.writer.release()
                    break

            except IndexError:
                self.cap.release()
                self.writer.release()

        video_timer.stop_timer()
        stdout_success(msg=f'Directionality video {self.video_name} saved in {self.directing_animals_video_output_path} directory', elapsed_time=video_timer.elapsed_time_str)

# style_attr = {'Show_pose': True, 'Pose_circle_size': 3, "Direction_color": 'Random', 'Direction_thickness': 4, 'Highlight_endpoints': True, 'Polyfill': True}
# test = DirectingOtherAnimalsVisualizer(config_path='/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/project_config.ini',
#                                        data_path='/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/csv/outlier_corrected_movement_location/Testing_Video_3.csv',
#                                        style_attr=style_attr)
#
# test.run()


# style_attr = {'Show_pose': True, 'Pose_circle_size': 3, "Direction_color": 'Random', 'Direction_thickness': 4, 'Highlight_endpoints': True, 'Polyfill': True}
# test = DirectingOtherAnimalsVisualizer(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                        data_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location/Together_1.csv',
#                                        style_attr=style_attr)
#
# test.run()

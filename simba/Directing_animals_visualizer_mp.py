__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd

from simba.Directing_animals_analyzer import DirectingOtherAnimalsAnalyzer
from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_if_filepath_list_is_empty,
                                          read_project_path_and_file_type,
                                          check_file_exist_and_readable)
import os, glob
from simba.drop_bp_cords import (get_fn_ext,
                                 getBpNames,
                                 createColorListofList,
                                 create_body_part_dictionary)
from simba.misc_tools import (find_video_of_file,
                              get_video_meta_data,
                              check_multi_animal_status,
                              SimbaTimer,
                              get_color_dict,
                              split_and_group_df,
                              remove_a_folder,
                              concatenate_videos_in_folder)
import cv2
import numpy as np
import random
from simba.rw_dfs import read_df
import platform
import multiprocessing
import functools
from simba.enums import (Formats,
                         Defaults,
                         ReadConfig,
                         Dtypes,
                         Paths)


def _img_creator(data: pd.DataFrame,
                 directionality_data: pd.DataFrame,
                 bp_names: dict,
                 style_attr: dict,
                 save_temp_dir: str,
                 video_path: str,
                 video_meta_data: dict,
                 colors: list):

    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    group_cnt = int(data.iloc[0]['group'])
    start_frm, current_frm, end_frm = data.index[0], data.index[0], data.index[-1]
    save_path = os.path.join(save_temp_dir, '{}.mp4'.format(str(group_cnt)))
    _, video_name, _ = get_fn_ext(filepath=video_path)
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'], (video_meta_data['width'], video_meta_data['height']))
    cap = cv2.VideoCapture(video_path)
    cap.set(1, start_frm)
    color = colors[0]

    def __draw_individual_lines(animal_img_data:pd.DataFrame,
                                img: np.array):
        color = colors[0]
        for cnt, (i, r) in enumerate(animal_img_data.iterrows()):
            if style_attr['Direction_color'] == 'Random':
                color = random.sample(colors[0], 1)[0]
            cv2.line(img, (int(r['Eye_x']), int(r['Eye_y'])), (int(r['Animal_2_bodypart_x']), int(r['Animal_2_bodypart_y'])), color, style_attr['Direction_thickness'])
            if style_attr['Highlight_endpoints']:
                cv2.circle(img, (int(r['Eye_x']), int(r['Eye_y'])), style_attr['Pose_circle_size'] + 2, color, style_attr['Pose_circle_size'])
                cv2.circle(img, (int(r['Animal_2_bodypart_x']), int(r['Animal_2_bodypart_y'])), style_attr['Pose_circle_size'] + 1, color, style_attr['Pose_circle_size'])

        return img

    while current_frm < end_frm:
        ret, img = cap.read()
        try:
            if ret:
                if style_attr['Show_pose']:
                    bp_data = data.loc[current_frm]
                    for cnt, (animal_name, animal_bps) in enumerate(bp_names.items()):
                        for bp in zip(animal_bps['X_bps'], animal_bps['Y_bps']):
                            x_bp, y_bp = bp_data[bp[0]], bp_data[bp[1]]
                            cv2.circle(img, (int(x_bp), int(y_bp)), style_attr['Pose_circle_size'], bp_names[animal_name]['colors'][cnt], style_attr['Direction_thickness'])

                if current_frm in list(directionality_data['Frame_#'].unique()):
                    img_data = directionality_data[directionality_data['Frame_#'] == current_frm]
                    unique_animals = img_data['Animal_1'].unique()
                    for animal in unique_animals:
                        animal_img_data = img_data[img_data['Animal_1'] == animal].reset_index(drop=True)
                        if style_attr['Polyfill']:
                            convex_hull_arr = animal_img_data.loc[0, ['Eye_x', 'Eye_y']].values.reshape(-1, 2)
                            for animal_2 in animal_img_data['Animal_2'].unique():
                                convex_hull_arr = np.vstack((convex_hull_arr, animal_img_data[['Animal_2_bodypart_x', 'Animal_2_bodypart_y']][animal_img_data['Animal_2'] == animal_2].values)).astype('int')
                                convex_hull_arr = np.unique(convex_hull_arr, axis=0)
                                if convex_hull_arr.shape[0] >= 3:
                                    if style_attr['Direction_color'] == 'Random':
                                        color = random.sample(colors[0], 1)[0]
                                    cv2.fillPoly(img, [convex_hull_arr], color)
                                else:
                                    img = __draw_individual_lines(animal_img_data=animal_img_data, img=img)

                        else:
                            img = __draw_individual_lines(animal_img_data=animal_img_data, img=img)


                img = np.uint8(img)

                current_frm += 1
                writer.write(img)
                print('Multi-processing video frame {} on core {}...'.format(str(current_frm), str(group_cnt)))

            else:
                cap.release()
                writer.release()
                break

        except IndexError:
            cap.release()
            writer.release()
            break

    return group_cnt

class DirectingOtherAnimalsVisualizerMultiprocess(object):
    """
    Class for visualizing when animals are directing towards body-parts of other animals using multiprocessing.

    > Note: Requires the pose-estimation data for the left ear, right ears and nose of individual animals.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    video_name: str
        Video to visualize directionality for (e.g., ``My_video.mp4``)
    style_attr: dict
        Video style attribitions.
    core_cnt: int
        How many cores to use to create the video.

    Notes
    -----
    Requires the pose-estimation data for the ``left ear``, ``right ear`` and ``nose`` of
    each individual animals. `YouTube example of expected output <https://youtu.be/4WXs3sKu41I>`__.

    Examples
    -----
    >>> style_attr = {'Show_pose': True, 'Pose_circle_size': 3, "Direction_color": 'Random', 'Direction_thickness': 4, 'Highlight_endpoints': True}
    >>> directing_visualizer = DirectingOtherAnimalsVisualizerMultiprocess(config_path='project_folder/project_config.ini', video_name='Testing_Video_3.mp4', style_attr=style_attr)
    >>> directing_visualizer.visualize_results()
    """

    def __init__(self,
                 config_path: str,
                 data_path: str,
                 style_attr: dict,
                 core_cnt: int):

        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        self.config, self.data_path = read_config_file(config_path), data_path
        _, self.video_name, _ = get_fn_ext(self.data_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.direction_analyzer = DirectingOtherAnimalsAnalyzer(config_path=config_path)
        self.direction_analyzer.process_directionality()
        self.direction_analyzer.create_directionality_dfs()
        self.core_cnt = core_cnt
        self.maxtasksperchild = Defaults.MAX_TASK_PER_CHILD.value
        self.chunksize = Defaults.CHUNK_SIZE.value
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.no_animals = read_config_entry(self.config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.ANIMAL_CNT.value, Dtypes.INT.value)
        self.style_attr, self.pose_colors, self.core_cnt = style_attr, [], core_cnt
        self.colors = get_color_dict()
        if self.style_attr['Show_pose']:
            self.pose_colors = createColorListofList(self.no_animals, int(len(self.x_cols) + 1))
        if self.style_attr['Direction_color'] == 'Random':
            self.direction_colors = createColorListofList(1, int(self.no_animals**2))
        else:
            self.direction_colors = [self.colors[self.style_attr['Direction_color']]]
        self.data_in_dir = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
        self.data_dict = self.direction_analyzer.directionality_df_dict
        self.video_directory = os.path.join(self.project_path, 'videos')
        self.video_path = find_video_of_file(self.video_directory, self.video_name)
        self.save_directory = os.path.join(self.project_path, Paths.DIRECTING_ANIMALS_OUTPUT_PATH.value)
        if not os.path.exists(self.save_directory): os.makedirs(self.save_directory)
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.no_animals, self.x_cols, self.y_cols, [], self.pose_colors)
        self.data_path = os.path.join(self.data_in_dir, self.video_name + '.' + self.file_type)
        check_file_exist_and_readable(file_path=self.data_path)
        self.session_timer = SimbaTimer()
        self.session_timer.start_timer()
        print(f'Processing video {self.video_name}...')

    def visualize_results(self):
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
            remove_a_folder(folder_dir=self.save_temp_path)
        os.makedirs(self.save_temp_path)
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_meta_data = get_video_meta_data(self.video_path)
        self.video_data = self.data_dict[self.video_name]
        if self.video_name in list(self.video_data['Video']):
            self.__create_video()
        else:
            print('SimBA skipping video {}: No animals are directing each other in the video.')
        self.session_timer.stop_timer()

    def __create_video(self):
        video_timer = SimbaTimer()
        video_timer.start_timer()
        data_arr, frm_per_core = split_and_group_df(df=self.data_df, splits=self.core_cnt, include_split_order=True)
        print('Creating ROI images, multiprocessing (determined chunksize: {}, cores: {})...'.format(str(self.chunksize), str(self.core_cnt)))
        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
            constants = functools.partial(_img_creator,
                                          directionality_data=self.video_data,
                                          video_meta_data=self.video_meta_data,
                                          style_attr=self.style_attr,
                                          save_temp_dir=self.save_temp_path,
                                          video_path=self.video_path,
                                          bp_names=self.animal_bp_dict,
                                          colors=self.direction_colors)
            for cnt, result in enumerate(pool.imap(constants, data_arr, chunksize=self.chunksize)):
                print('Image {}/{}, Video {}...'.format(str(int(frm_per_core * (result + 1))), str(len(self.data_df)), self.video_name))

            concatenate_videos_in_folder(in_folder=self.save_temp_path, save_path=self.save_path, video_format='mp4')
            video_timer.stop_timer()
            pool.terminate()
            pool.join()
            print('SIMBA COMPLETE: Video {} created. Video saved in project_folder/frames/output/ROI_directionality_visualize (elapsed time: {}s).'.format(self.video_name, video_timer.elapsed_time_str))


# style_attr = {'Show_pose': True, 'Pose_circle_size': 3, "Direction_color": 'Random', 'Direction_thickness': 4, 'Highlight_endpoints': True, 'Polyfill': True}
# test = DirectingOtherAnimalsVisualizerMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/project_config.ini',
#                                        video_name='Testing_Video_3.mp4',
#                                        style_attr=style_attr,
#                                                    core_cnt=5)
#
# test.visualize_results()


# style_attr = {'Show_pose': True, 'Pose_circle_size': 3, "Direction_color": 'Random', 'Direction_thickness': 4, 'Highlight_endpoints': True, 'Polyfill': True}
# test = DirectingOtherAnimalsVisualizerMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                        data_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location/Together_1.csv',
#                                        style_attr=style_attr,
#                                                    core_cnt=5)
#
# test.visualize_results()

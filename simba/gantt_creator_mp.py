__author__ = "Simon Nilsson", "JJ Choong"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from simba.misc_tools import (detect_bouts,
                              concatenate_videos_in_folder,
                              SimbaTimer,
                              get_named_colors)
from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_if_filepath_list_is_empty,
                                          read_project_path_and_file_type)
from simba.features_scripts.unit_tests import (read_video_info_csv,
                                               read_video_info)
from simba.enums import (ReadConfig,
                         Formats,
                         Paths,
                         Dtypes,
                         Defaults)
from simba.misc_visualizations import make_gantt_plot
from simba.train_model_functions import get_all_clf_names
from simba.drop_bp_cords import get_fn_ext
from simba.rw_dfs import read_df
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import multiprocessing
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import functools
import shutil
import platform

def _image_creator(data: np.array,
                   frame_setting: bool,
                   video_setting: bool,
                   video_save_dir: str,
                   frame_folder_dir: str,
                   bouts_df: pd.DataFrame,
                   clf_names: list,
                   colors: list,
                   color_tuple: tuple,
                   fps: int,
                   rotation: int,
                   font_size: int,
                   width: int,
                   height: int,
                   video_name: str):

    group, frame_rng = data[0], data[1:]
    start_frm, end_frm, current_frm = frame_rng[0], frame_rng[-1], frame_rng[0]

    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_save_path = os.path.join(video_save_dir, '{}.mp4'.format(str(group)))
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))

    while current_frm < end_frm:
        fig, ax = plt.subplots()
        bout_rows = bouts_df.loc[bouts_df['End_frame'] <= current_frm]
        for i, event in enumerate(bout_rows.groupby("Event")):
            for x in clf_names:
                if event[0] == x:
                    ix = clf_names.index(x)
                    data_event = event[1][["Start_time", "Bout_time"]]
                    ax.broken_barh(data_event.values, (color_tuple[ix], 3), facecolors=colors[ix])

        x_ticks_locs = x_lbls = np.round(np.linspace(0, round((current_frm / fps), 3), 6))
        ax.set_xticks(x_ticks_locs)
        ax.set_xticklabels(x_lbls)
        ax.set_ylim(0, color_tuple[len(clf_names)])
        ax.set_yticks(np.arange(5, 5 * len(clf_names) + 1, 5))
        ax.tick_params(axis='both', labelsize=font_size)
        ax.set_yticklabels(clf_names, rotation=rotation)
        ax.set_xlabel('Session (s)', fontsize=font_size)
        ax.yaxis.grid(True)
        canvas = FigureCanvas(fig)
        canvas.draw()
        mat = np.array(canvas.renderer._renderer)
        img = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        img = np.uint8(cv2.resize(img, (width, height)))
        if video_setting:
            video_writer.write(img)
        if frame_setting:
            frame_save_name = os.path.join(frame_folder_dir, '{}.png'.format(str(current_frm)))
            cv2.imwrite(frame_save_name, img)
        plt.close(fig)
        current_frm += 1

        print('Gantt frame created: {}, Video: {}, Processing core: {}'.format(str(current_frm+1), video_name, str(group+1)))

    if video_setting:
        video_writer.release()

    return group

class GanttCreatorMultiprocess(object):

    """
    Class for multiprocess creation of classifier gantt charts in video and/or image format.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    frame_setting: bool
        If True, creates individual frames
    video_setting: bool
        If True, creates videos
    files_found: list
        File paths representing files with machine predictions e.g., ['project_folder/csv/machine_results/My_results.csv']
    cores: int
        Number of cores to use
    style_attr: dict
        Output image style attributes, e.g., {'width': 640, 'height': 480, 'font size': 8, 'font rotation': 45}


    Notes
    ----------
    `GitHub gantt tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#gantt-plot>`__.

    Examples
    ----------
    >>> gantt_creator = GanttCreatorMultiprocess(config_path='tests/test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', frame_setting=False, video_setting=True, files_found=['tests/test_data/multi_animal_dlc_two_c57/project_folder/csv/machine_results/Together_1.csv'], cores=5, style_attr={'width': 640, 'height': 480, 'font size': 8, 'font rotation': 45})
    >>> gantt_creator.create_gannt()

    """

    def __init__(self,
                 config_path: str,
                 frame_setting: bool,
                 video_setting: bool,
                 files_found: list,
                 cores: int,
                 style_attr: dict,
                 last_frm_setting: bool):

        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        self.frame_setting, self.video_setting, self.files_found, self.style_attr, self.cores, self.last_frm_setting = frame_setting, video_setting, files_found, style_attr, cores, last_frm_setting
        if (not self.frame_setting) and (not self.video_setting) and (not self.last_frm_setting):
            print('SIMBA ERROR: Please select gantt videos, frames, and/or last frame.')
            raise ValueError('SIMBA ERROR: Please select gantt videos, frames, and/or last frame.')
        self.config = read_config_file(config_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.data_in_dir = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.target_cnt = read_config_entry(self.config, ReadConfig.SML_SETTINGS.value, ReadConfig.TARGET_CNT.value, data_type=Dtypes.INT.value)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg='SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. Create classification results before visualizing gantt charts')
        self.colours = get_named_colors()[:-1]
        self.colour_tuple_x = list(np.arange(3.5, 203.5, 5))
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.target_cnt)
        self.out_parent_dir = os.path.join(self.project_path, Paths.GANTT_PLOT_DIR.value)
        if not os.path.exists(self.out_parent_dir):
            os.makedirs(self.out_parent_dir)
        self.y_rotation, self.y_fontsize = self.style_attr['font rotation'], self.style_attr['font size']
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        self.out_width, self.out_height = self.style_attr['width'], self.style_attr['height']
        self.maxtasksperchild = Defaults.MAX_TASK_PER_CHILD.value
        self.chunksize = Defaults.CHUNK_SIZE.value
        self.timer = SimbaTimer()
        self.timer.start_timer()

        print('Processing {} video(s)...'.format(str(len(self.files_found))))

    def create_gannt(self):
        '''
        Creates gantt charts. Results are stored in the `project_folder/frames/gantt_plots` directory of SimBA project.

        Returns
        ----------
        None
        '''

        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer()
            video_timer.start_timer()
            _, self.video_name, _ = get_fn_ext(file_path)
            self.data_df = read_df(file_path, self.file_type).reset_index(drop=True)
            print('Processing video {}, Frame count: {} (Video {}/{})...'.format(self.video_name, str(len(self.data_df)), str(file_cnt+1), str(len(self.files_found))))
            self.video_info_settings, _, self.fps = read_video_info(self.vid_info_df, self.video_name)
            self.bouts_df = detect_bouts(data_df=self.data_df, target_lst=list(self.clf_names), fps=int(self.fps))
            self.temp_folder = os.path.join(self.out_parent_dir, self.video_name, 'temp')
            self.save_frame_folder_dir = os.path.join(self.out_parent_dir, self.video_name)
            if self.frame_setting:
                if os.path.exists(self.save_frame_folder_dir): shutil.rmtree(self.save_frame_folder_dir)
                if not os.path.exists(self.save_frame_folder_dir): os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                self.video_folder = os.path.join(self.out_parent_dir, self.video_name)
                if os.path.exists(self.temp_folder):
                    shutil.rmtree(self.temp_folder)
                    shutil.rmtree(self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(self.out_parent_dir, self.video_name + '.mp4')

            if self.last_frm_setting:
                _ = make_gantt_plot(data_df=self.data_df,
                                    bouts_df=self.bouts_df,
                                    clf_names=self.clf_names,
                                    fps=self.fps,
                                    style_attr=self.style_attr,
                                    video_name=self.video_name,
                                    save_path=os.path.join(self.out_parent_dir, self.video_name + '_final_image.png'))

            if self.video_setting or self.frame_setting:
                frame_array = np.array_split(list(range(0, len(self.data_df))), self.cores)
                frm_per_core = len(frame_array[0])
                for group_cnt, rng in enumerate(frame_array):
                    frame_array[group_cnt] = np.insert(rng, 0, group_cnt)

                print('Creating gantt, multiprocessing (chunksize: {}, cores: {})...'.format(str(self.chunksize), str(self.cores)))
                with multiprocessing.pool.Pool(self.cores, maxtasksperchild=self.maxtasksperchild) as pool:
                    constants = functools.partial(_image_creator,
                                                  video_setting=self.video_setting,
                                                  frame_setting=self.frame_setting,
                                                  video_save_dir=self.temp_folder,
                                                  frame_folder_dir=self.save_frame_folder_dir,
                                                  bouts_df=self.bouts_df,
                                                  rotation=self.y_rotation,
                                                  clf_names=self.clf_names,
                                                  colors=self.colours,
                                                  color_tuple=self.colour_tuple_x,
                                                  fps=self.fps,
                                                  font_size=self.y_fontsize,
                                                  width=self.out_width,
                                                  height=self.out_height,
                                                  video_name=self.video_name)

                    for cnt, result in enumerate(pool.imap(constants, frame_array, chunksize=self.chunksize)):
                        print('Image {}/{}, Video {}/{}...'.format(str(int(frm_per_core * (result+1))), str(len(self.data_df)), str(file_cnt + 1), str(len(self.files_found))))
                    pool.terminate()
                    pool.join()

                if self.video_setting:
                    print('Joining {} multiprocessed video...'.format(self.video_name))
                    concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.save_video_path)
                video_timer.stop_timer()
                print('Gantt video {} complete (elapsed time: {}s) ...'.format(self.video_name, video_timer.elapsed_time_str))

        self.timer.stop_timer()
        print('SIMBA COMPLETE: Gantt visualizations for {} videos created in project_folder/frames/output/gantt_plots directory (elapsed time: {}s)'.format(str(len(self.files_found)), self.timer.elapsed_time_str))

# test = GanttCreatorMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 frame_setting=False,
#                                 video_setting=True,
#                                 files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                                 cores=5,
#                                 last_frm_setting=False,
#                                 style_attr={'width': 640, 'height': 480, 'font size': 12, 'font rotation': 45})
# test.create_gannt()


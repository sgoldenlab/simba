__author__ = "Simon Nilsson", "JJ Choong"


from simba.read_config_unit_tests import (read_config_entry,
                                          check_that_column_exist,
                                          read_config_file,
                                          read_project_path_and_file_type)
from simba.features_scripts.unit_tests import (read_video_info_csv,
                                               read_video_info)
import functools
import pandas as pd
from simba.misc_tools import (SimbaTimer,
                              concatenate_videos_in_folder)
from simba.misc_visualizations import make_probability_plot
from simba.enums import (Formats,
                         ReadConfig,
                         Paths,
                         Defaults)
import os, glob
from simba.drop_bp_cords import get_fn_ext
from simba.rw_dfs import read_df
import matplotlib.pyplot as plt
import cv2
import numpy as np
import shutil
import multiprocessing
import platform
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def _create_probability_plots(data: list,
                              probability_lst: list,
                              clf_name: str,
                              video_setting: bool,
                              frame_setting: bool,
                              video_dir: str,
                              frame_dir: str,
                              highest_p: float,
                              fps: int,
                              style_attr: dict,
                              video_name: str):

    group, data = data[0], data[1:]
    start_frm, end_frm, current_frm = data[0], data[-1], data[0]

    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_save_path = os.path.join(video_dir, '{}.mp4'.format(str(group)))
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (style_attr['width'], style_attr['height']))

    while current_frm < end_frm:
        fig, ax = plt.subplots()
        current_lst = probability_lst[0:current_frm+1]
        ax.plot(current_lst, color=style_attr['color'], linewidth=style_attr['line width'])
        ax.plot(current_frm, current_lst[-1], "o", markersize=style_attr['circle size'], color=style_attr['color'])
        ax.set_ylim([0, highest_p])
        x_ticks_locs = x_lbls = np.linspace(0, current_frm, 5)
        x_lbls = np.round((x_lbls / fps), 1)
        ax.xaxis.set_ticks(x_ticks_locs)
        ax.set_xticklabels(x_lbls, fontsize=style_attr['font size'])
        ax.set_xlabel('Time (s)', fontsize=style_attr['font size'])
        ax.set_ylabel('{} {}'.format(clf_name, 'probability'), fontsize=style_attr['font size'])
        plt.suptitle(clf_name, x=0.5, y=0.92, fontsize=style_attr['font size'] + 4)
        canvas = FigureCanvas(fig)
        canvas.draw()
        mat = np.array(canvas.renderer._renderer)
        image = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        image = np.uint8(cv2.resize(image, (style_attr['width'], style_attr['height'])))
        if video_setting:
            video_writer.write(image)
        if frame_setting:
            frame_save_name = os.path.join(frame_dir, '{}.png'.format(str(current_frm)))
            cv2.imwrite(frame_save_name, image)
        plt.close()
        current_frm += 1

        print('Probability frame created: {}, Video: {}, Processing core: {}'.format(str(current_frm+1), video_name, str(group+1)))


    return group

class TresholdPlotCreatorMultiprocess(object):
    """
    Class for line chart visualizations displaying the classification probabilities of a single classifier.
    Uses multiprocessing.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    clf_name: str
        Name of the classifier to create visualizations for
    frame_setting: bool
       When True, SimBA creates individual frames in png format
    video_setting: bool
       When True, SimBA creates compressed video in mp4 format
    files_found: list
        File paths to create probability plots for, e.g., ['project_folder/csv/machine_results/MyVideo.csv]
    style_attr: dict
        Output image style attributes, e.g., {'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'magneta', 'circle size': 20}
    cores: int
        Number of cores to use

    Notes
    ----------
    `Visualization tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.


    Examples
    ----------
    >>> plot_creator = TresholdPlotCreatorMultiprocess(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', frame_setting=True, video_setting=True, clf_name='Attack', style_attr={'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'magneta', 'circle size': 20}, cores=5)
    >>> plot_creator.create_plot()
    """


    def __init__(self,
                 config_path: str,
                 clf_name: str,
                 frame_setting: bool,
                 video_setting: bool,
                 last_frame: bool,
                 cores: int,
                 style_attr: dict,
                 files_found: list):

        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        self.frame_setting, self.video_setting, self.cores, self.style_attr, self.last_frame = frame_setting, video_setting, cores, style_attr, last_frame
        if (not self.frame_setting) and (not self.video_setting) and (not self.last_frame):
            print('SIMBA ERROR: Please choose to either probability videos, frames, and/or last frame.')
            raise ValueError('SIMBA ERROR: Please choose to either probability videos, frames, and/or last frame.')

        self.config = read_config_file(config_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.data_in_dir = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.files_found = glob.glob(self.data_in_dir + '/*.' + self.file_type)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        self.out_parent_dir = os.path.join(self.project_path, Paths.PROBABILITY_PLOTS_DIR.value)
        self.clf_name, self.files_found = clf_name, files_found
        self.probability_col = 'Probability_' + self.clf_name
        self.fontsize = self.style_attr['font size']
        self.out_width, self.out_height = self.style_attr['width'], self.style_attr['height']
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        self.maxtasksperchild = Defaults.MAX_TASK_PER_CHILD.value
        self.chunksize = Defaults.CHUNK_SIZE.value
        if not os.path.exists(self.out_parent_dir): os.makedirs(self.out_parent_dir)
        print('Processing {} video(s)...'.format(str(len(self.files_found))))
        self.timer = SimbaTimer()
        self.timer.start_timer()

    def create_plots(self):
        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer()
            video_timer.start_timer()
            _, self.video_name, _ = get_fn_ext(file_path)
            video_info, self.px_per_mm, self.fps = read_video_info(vid_info_df=self.vid_info_df, video_name=self.video_name)
            data_df = read_df(file_path, self.file_type)
            check_that_column_exist(df=data_df, column_name=self.clf_name, file_name=self.video_name)
            self.save_frame_folder_dir = os.path.join(self.out_parent_dir, self.video_name + '_' + self.clf_name)
            self.video_folder = os.path.join(self.out_parent_dir, self.video_name + '_' + self.clf_name)
            self.temp_folder = os.path.join(self.out_parent_dir, self.video_name + '_' + self.clf_name, 'temp')
            if self.frame_setting:
                if os.path.exists(self.save_frame_folder_dir): shutil.rmtree(self.save_frame_folder_dir)
                if not os.path.exists(self.save_frame_folder_dir): os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                if os.path.exists(self.temp_folder):
                    shutil.rmtree(self.temp_folder)
                    shutil.rmtree(self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(self.out_parent_dir, '{}_{}.mp4'.format(self.video_name, self.clf_name))

            probability_lst = list(data_df[self.probability_col])

            if self.last_frame:
                _ = make_probability_plot(data=pd.Series(probability_lst),
                                          style_attr=self.style_attr,
                                          clf_name=self.clf_name,
                                          fps=self.fps,
                                          save_path=os.path.join(self.out_parent_dir, self.video_name + '_{}_{}.png'.format(self.clf_name, 'final_image')))


            if self.video_setting or self.frame_setting:
                if self.style_attr['y_max'] == 'auto':
                    highest_p = data_df[self.probability_col].max()
                else:
                    highest_p = float(self.style_attr['y_max'])
                data_split = np.array_split(list(data_df.index), self.cores)
                frm_per_core = len(data_split[0])
                for group_cnt, rng in enumerate(data_split):
                    data_split[group_cnt] = np.insert(rng, 0, group_cnt)


                print('Creating probability images, multiprocessing (determined chunksize: {}, cores: {})...'.format(str(self.chunksize), str(self.cores)))
                with multiprocessing.Pool(self.cores, maxtasksperchild=self.maxtasksperchild) as pool:
                    constants = functools.partial(_create_probability_plots,
                                                  clf_name=self.clf_name,
                                                  probability_lst=probability_lst,
                                                  highest_p= highest_p,
                                                  video_setting=self.video_setting,
                                                  frame_setting=self.frame_setting,
                                                  fps=self.fps,
                                                  video_dir=self.temp_folder,
                                                  frame_dir=self.save_frame_folder_dir,
                                                  style_attr=self.style_attr,
                                                  video_name=self.video_name)
                    for cnt, result in enumerate(pool.imap(constants, data_split, chunksize=self.chunksize)):
                        print('Image {}/{}, Video {}/{}...'.format(str(int(frm_per_core*(result+1))), str(len(data_df)), str(file_cnt+1), str(len(self.files_found))))

                pool.join()
                pool.terminate()
                if self.video_setting:
                    print('Joining {} multiprocessed video...'.format(self.video_name))
                    concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.save_video_path)

                video_timer.stop_timer()
                print('Probability video {} complete (elapsed time: {}s) ...'.format(self.video_name, video_timer.elapsed_time_str))

        self.timer.stop_timer()
        print('SIMBA COMPLETE: Probability visualizations for {} videos created in project_folder/frames/output/gantt_plots directory (elapsed time: {}s)'.format(str(len(self.files_found)), self.timer.elapsed_time_str))


# test = TresholdPlotCreatorMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                         frame_setting=True,
#                                         video_setting=False,
#                                         last_frame=True,
#                                         clf_name='Attack',
#                                         cores=5,
#                                         files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                                         style_attr={'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'blue', 'circle size': 20})
# test.create_plots()












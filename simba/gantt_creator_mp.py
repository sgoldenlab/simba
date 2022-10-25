__author__ = "Simon Nilsson", "JJ Choong"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from simba.misc_tools import detect_bouts, find_core_cnt, remove_a_folder
from simba.read_config_unit_tests import read_config_entry, read_config_file
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.train_model_functions import get_all_clf_names
from simba.drop_bp_cords import get_fn_ext
from simba.rw_dfs import read_df
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import multiprocessing
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pathlib
import time
import re
import functools
import shutil
import platform

def _image_creator(frm_range: list=None,
         bouts_df: pd.DataFrame=None,
         clf_names: list=None,
         colors: list=None,
         color_tuple: tuple=None,
         fps: int=None,
         rotation: int=None,
         font_size: int=None,
         width: int=None,
         height: int=None):

    img_lst = []
    for image_number in frm_range:
        fig, ax = plt.subplots()
        relevant_rows = bouts_df.loc[bouts_df['End_frame'] <= image_number]
        for i, event in enumerate(relevant_rows.groupby("Event")):
            for x in clf_names:
                if event[0] == x:
                    ix = clf_names.index(x)
                    data_event = event[1][["Start_time", "Bout_time"]]
                    ax.broken_barh(data_event.values, (color_tuple[ix], 3), facecolors=colors[ix])
        x_length = (round(image_number / fps)) + 1
        if x_length < 10: x_length = 10
        ax.set_xlim(0, x_length)
        ax.set_ylim(0, color_tuple[len(clf_names)])
        ax.set_yticks(np.arange(5, 5 * len(clf_names) + 1, 5))
        ax.set_yticklabels(clf_names, rotation=rotation, fontsize=font_size)
        ax.set_xlabel('Session (s)', fontsize=font_size)
        ax.yaxis.grid(True)
        canvas = FigureCanvas(fig)
        canvas.draw()
        mat = np.array(canvas.renderer._renderer)
        image = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        image = np.uint8(cv2.resize(image, (width, height)))
        img_lst.append(image)
    return img_lst


class GanttCreator(object):

    """
    Class for multiprocess creation of classifier gantt charts.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    frame_setting: bool
        If True, creates individual frames
    video_setting: bool
        If True, creates videos

    Notes
    ----------
    `GitHub gantt tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#gantt-plot>`__.

    Examples
    ----------
    >>> gantt_creator = GanttCreator(config_path='MyProjectConfi', frame_setting=False, video_setting=True)
    >>> gantt_creator.create_gannt()

    """

    def __init__(self,
                 config_path: str,
                 frame_setting: bool,
                 video_setting: bool):

        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        self.frame_setting = frame_setting
        self.video_setting = video_setting
        if (self.frame_setting != True) and (self.video_setting != True):
            print('SIMBA ERROR: Please choose to select either videos, frames, or both frames and videos.')
            raise ValueError('SIMBA ERROR: Please choose to select either videos, frames, or both frames and videos.')
        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.data_in_dir = os.path.join(self.project_path, 'csv', 'machine_results')
        self.target_cnt = read_config_entry(self.config, 'SML settings', 'No_targets', data_type='int')
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.files_found = glob.glob(self.data_in_dir + '/*.' + self.file_type)
        self.colours = ['red', 'green', 'pink', 'orange', 'blue', 'purple', 'lavender', 'grey', 'sienna', 'tomato', 'azure', 'crimson', 'aqua', 'plum', 'teal', 'maroon', 'lime', 'coral', 'deeppink']
        self.colour_tuple_x = list(np.arange(3.5, 203.5, 5))
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.target_cnt)
        self.out_parent_dir = os.path.join(self.project_path, 'frames', 'output', 'gantt_plots')
        self.cpu_cnt, self.cpu_to_use = find_core_cnt()
        if not os.path.exists(self.out_parent_dir): os.makedirs(self.out_parent_dir)
        self.y_rotation = 45
        self.y_fontsize = 8
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_width, self.out_height = 640, 480
        self.maxtasksperchild = 10
        self.chunksize = 1

        print('Processing {} video(s)...'.format(str(len(self.files_found))))

    def create_gannt(self):
        '''
        Creates gantt charts. Results are stored in the `project_folder/frames/gantt_plots` directory of SimBA project.

        Returns
        ----------
        None
        '''

        start_time = time.time()
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            self.data_df = read_df(file_path, self.file_type).reset_index(drop=True)
            print('Processing video {}, Frame count: {} (Video {}/{})...'.format(self.video_name, str(len(self.data_df)), str(file_cnt+1), str(len(self.files_found))))
            self.video_info_settings, _, self.fps = read_video_info(self.vid_info_df, self.video_name)
            self.bouts_df = detect_bouts(data_df=self.data_df, target_lst=self.clf_names, fps=self.fps)
            if self.frame_setting:
                self.save_frame_folder_dir = os.path.join(self.out_parent_dir, self.video_name)
                if os.path.exists(self.save_frame_folder_dir): shutil.rmtree(self.save_frame_folder_dir)
                if not os.path.exists(self.save_frame_folder_dir): os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                self.video_folder = os.path.join(self.out_parent_dir, self.video_name)
                self.temp_folder = os.path.join(self.out_parent_dir, self.video_name, 'temp')
                if os.path.exists(self.temp_folder):
                    shutil.rmtree(self.temp_folder)
                    shutil.rmtree(self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(self.out_parent_dir, self.video_name + '.mp4')

            frame_rng = list(range(0, len(self.data_df)))
            frame_rng = np.array_split(frame_rng, int(len(self.data_df) / (self.fps * 2)))
            imgs_peer_loop = len(frame_rng[0])
            if len(self.data_df) < 100:
                self.chunksize = 1
            print('Creating gantt images, multiprocessing (determined chunksize: {}, cores img creation: {}, cores img writing: {})...'.format(str(self.chunksize), str(self.cpu_to_use), str(self.cpu_to_use)))
            with multiprocessing.pool.Pool(self.cpu_to_use, maxtasksperchild=self.maxtasksperchild) as pool:
                functools.partial(_image_creator, b=self.bouts_df)
                constants = functools.partial(_image_creator,
                                              bouts_df=self.bouts_df,
                                              rotation=self.y_rotation,
                                              clf_names=self.clf_names,
                                              colors=self.colours,
                                              color_tuple=self.colour_tuple_x,
                                              fps=self.fps,
                                              font_size=self.y_fontsize,
                                              width=self.out_width,
                                              height=self.out_height)
                for cnt, result in enumerate(pool.imap(constants, frame_rng, chunksize=self.chunksize)):
                    if self.video_setting:
                        save_path = os.path.join(self.temp_folder, str(cnt) + '.mp4')
                        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.out_width, self.out_height))
                        for img in result:
                            writer.write(img)
                        writer.release()
                    if self.frame_setting:
                        for img_cnt, img in enumerate(result):
                            file_save_path = os.path.join(self.save_frame_folder_dir, str(len(os.listdir(self.save_frame_folder_dir))) + '.png')
                            cv2.imwrite(file_save_path, img)
                    if int(imgs_peer_loop*cnt) < len(self.data_df):
                        print('Image {}/{}, Video {}/{}...'.format(str(int(imgs_peer_loop*cnt)), str(len(self.data_df)), str(file_cnt+1), str(len(self.files_found))))
                pool.terminate()
                pool.join()

            if self.video_setting:
                files = glob.glob(self.temp_folder + '/*.mp4')
                files.sort(key=lambda f: int(re.sub('\D', '', f)))
                temp_txt_path = pathlib.Path(self.temp_folder, 'files.txt')
                with open(temp_txt_path, 'w') as f:
                    for file in files:
                        f.write("file '" + str(pathlib.Path(file)) + "'\n")
                if os.path.exists(self.save_video_path): os.remove(self.save_video_path)
                returned = os.system('ffmpeg -f concat -safe 0 -i {} {} -hide_banner -loglevel error'.format(temp_txt_path, self.save_video_path))
                while True:
                    if returned != 0:
                        pass
                    else:
                        remove_a_folder(folder_dir=self.temp_folder)
                        break

        elapsed_time = str(round(time.time() - start_time, 2)) + 's'
        print('SIMBA COMPLETE: All gantt visualizations created in project_folder/frames/output/gantt_plots directory. Elapsed time {}'.format(elapsed_time))


# if __name__== "__main__":
# # #     test = GanttCreator(config_path="Z:\Classifiers\Mounting_080920\project_folder\project_config.ini",
# # #                         frame_setting=False,
# # #                         video_setting=True)
# # #     test.create_gannt()
# #
# #
#     test = GanttCreator(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini',
#                         frame_setting=False,
#                         video_setting=True)
#     test.create_gannt()
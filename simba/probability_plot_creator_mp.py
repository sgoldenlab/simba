__author__ = "Simon Nilsson", "JJ Choong"


from simba.read_config_unit_tests import (read_config_entry,
                                          check_that_column_exist,
                                          read_config_file)
from simba.features_scripts.unit_tests import (read_video_info_csv,
                                               read_video_info)
import functools
from simba.misc_tools import find_core_cnt, remove_a_folder
import os, glob
from simba.drop_bp_cords import get_fn_ext
from simba.rw_dfs import read_df
import matplotlib.pyplot as plt
import cv2
import numpy as np
from copy import deepcopy
import shutil
import multiprocessing
import platform
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pathlib
import re
import time

def _create_probability_plots(frm_range: list,
                             data: list,
                             clf_name: str,
                             highest_p: float,
                             fps: int,
                             font_size: int,
                             width: int,
                             height: int):

    img_lst = []
    for frm_cnt in frm_range:
        fig, ax = plt.subplots()
        current_lst = data[0:frm_cnt+1]
        ax.plot(current_lst, color="m", linewidth=6)
        ax.plot(frm_cnt, current_lst[-1], "o", markersize=20, color="m")
        ax.set_ylim([0, highest_p])
        ax.set_ylabel(clf_name + ' probability', fontsize=font_size)
        if frm_cnt > fps * 60:
            ax.xaxis.set_ticks(np.arange(0, frm_cnt, int(fps*30)))
            x_labels = [int(x / fps) for x in list(range(0, frm_cnt, int(fps * 30)))]
            ax.set_xticklabels(x_labels, fontsize=font_size)
        elif frm_cnt > ((fps * 60) * 10):
            ax.xaxis.set_ticks(np.arange(0, frm_cnt, int(fps*60)))
            x_labels = [int(x / fps) for x in list(range(0, frm_cnt, int(fps * 60)))]
            ax.set_xticklabels(x_labels, fontsize=font_size)
        elif frm_cnt < fps * 60:
            ax.xaxis.set_ticks(np.arange(0, frm_cnt, int(fps*1)))
            x_labels = [int(x / fps) for x in list(range(0, frm_cnt, int(fps*1)))]
            ax.set_xticklabels(x_labels, fontsize=font_size)

        ax.set_xlabel('Time (s)', fontsize=font_size)
        plt.suptitle(clf_name, x=0.5, y=0.92, fontsize=12)
        canvas = FigureCanvas(fig)
        canvas.draw()
        mat = np.array(canvas.renderer._renderer)
        image = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        image = np.uint8(cv2.resize(image, (width, height)))
        img_lst.append(image)
        plt.close()

    return img_lst

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

    Notes
    ----------
    `Visualization tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.


    Examples
    ----------
    >>> plot_creator = TresholdPlotCreatorMultiprocess(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', frame_setting=True, video_setting=True, clf_name='Attack')
    >>> plot_creator.create_plot()
    """


    def __init__(self,
                 config_path: str,
                 clf_name: str,
                 frame_setting: bool,
                 video_setting: bool):



        self.frame_setting = frame_setting
        self.video_setting = video_setting
        if (not self.frame_setting) and (not self.video_setting):
            print('SIMBA ERROR: Please choose to select either videos, frames, or both frames and videos.')
            raise ValueError('SIMBA ERROR: Please choose to select either videos, frames, or both frames and videos.')

        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        self.config = read_config_file(config_path)
        self.animal_cnt = read_config_entry(self.config, 'Path plot settings', 'no_animal_pathplot', 'int', 1)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.data_in_dir = os.path.join(self.project_path, 'csv', 'machine_results')
        self.files_found = glob.glob(self.data_in_dir + '/*.' + self.file_type)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.out_parent_dir = os.path.join(self.project_path, 'frames', 'output', 'probability_plots')
        self.clf_name = clf_name
        self.probability_col = 'Probability_' + self.clf_name
        self.fontsize = 10
        self.cpu_cnt, self.cpu_to_use = find_core_cnt()
        self.out_width, self.out_height = 640, 480
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.maxtasksperchild = 10
        self.chunksize = 1
        if not os.path.exists(self.out_parent_dir): os.makedirs(self.out_parent_dir)
        print('Processing {} video(s)...'.format(str(len(self.files_found))))

    def create_plot(self):
        start_time = time.time()
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            video_info, self.px_per_mm, self.fps = read_video_info(vid_info_df=self.vid_info_df, video_name=self.video_name)
            data_df = read_df(file_path, self.file_type)
            check_that_column_exist(df=data_df, column_name=self.clf_name, file_name=self.video_name)
            if self.frame_setting:
                self.save_frame_folder_dir = os.path.join(self.out_parent_dir, self.video_name + '_' + self.clf_name)
                if os.path.exists(self.save_frame_folder_dir): shutil.rmtree(self.save_frame_folder_dir)
                if not os.path.exists(self.save_frame_folder_dir): os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                self.video_folder = os.path.join(self.out_parent_dir, self.video_name + '_' + self.clf_name)
                self.temp_folder = os.path.join(self.out_parent_dir, self.video_name + '_' + self.clf_name, 'temp')
                if os.path.exists(self.temp_folder):
                    shutil.rmtree(self.temp_folder)
                    shutil.rmtree(self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(self.out_parent_dir, '{}_{}.mp4'.format(self.video_name, self.clf_name))

            self.probability_lst = list(data_df[self.probability_col])
            highest_p = max(self.probability_lst)
            frame_rng = list(range(0, len(self.probability_lst)))
            frame_rng = np.array_split(frame_rng, int(len(self.probability_lst) / (self.fps * 2)))
            imgs_peer_loop = len(frame_rng[0])
            print('Creating probability images, multiprocessing (determined chunksize: {}, cores img creation: {}, cores img writing: {})...'.format(str(self.chunksize), str(self.cpu_to_use), str(self.cpu_to_use)))
            with multiprocessing.Pool(self.cpu_to_use, maxtasksperchild=self.maxtasksperchild) as pool:
                functools.partial(_create_probability_plots, b=self.probability_lst)
                constants = functools.partial(_create_probability_plots,
                                              data=self.probability_lst,
                                              clf_name=self.clf_name,
                                              highest_p= highest_p,
                                              fps=self.fps,
                                              font_size=self.fontsize,
                                              width=self.out_width,
                                              height=self.out_height)
                for cnt, result in enumerate(pool.imap(constants, frame_rng, chunksize=self.chunksize)):
                    if self.video_setting:
                        save_path = os.path.join(self.temp_folder, str(cnt) + '.mp4')
                        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps,(self.out_width, self.out_height))
                        for img in result:
                            writer.write(img)
                        writer.release()
                    if self.frame_setting:
                        for img_cnt, img in enumerate(result):
                            file_save_path = os.path.join(self.save_frame_folder_dir, str(len(os.listdir(self.save_frame_folder_dir))) + '.png')
                            cv2.imwrite(file_save_path, img)
                    print('Image {}/{}, Video {}/{}...'.format(str(int(imgs_peer_loop*cnt)), str(len(self.probability_lst)), str(file_cnt+1), str(len(self.files_found))))
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
                returned = os.system('ffmpeg -f concat -safe 0 -i "{}" "{}" -hide_banner -loglevel error'.format(temp_txt_path, self.save_video_path))
                while True:
                    if returned != 0:
                        pass
                    else:
                        remove_a_folder(folder_dir=self.temp_folder)
                        break
            print('Video {} complete...'.format(self.video_name))

        elapsed_time = str(round(time.time() - start_time, 2)) + 's'
        print('SIMBA COMPLETE: All probability visualizations created in project_folder/frames/output/probability_plots directory. Elapsed time {}'.format(elapsed_time))

# if __name__== "__main__":
#     test = TresholdPlotCreator(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini',
#                                frame_setting=False, video_setting=True, clf_name='Attack')
#     test.create_plot()












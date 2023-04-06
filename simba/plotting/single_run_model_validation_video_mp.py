__author__ = "Simon Nilsson"

import warnings
import time
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
import pickle
import imutils
from PIL import Image
from simba.mixins.config_reader import ConfigReader
import cv2
import warnings
from simba.drop_bp_cords import drop_bp_cords
import matplotlib.pyplot as plt
import numpy as np
from simba.rw_dfs import *
from simba.misc_tools import (find_video_of_file,
                              get_fn_ext,
                              get_video_meta_data,
                              plug_holes_shortest_bout,
                              get_bouts_for_gantt,
                              create_gantt_img,
                              resize_gantt)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from simba.misc_tools import concatenate_videos_in_folder
from simba.feature_extractors.unit_tests import read_video_info
import platform
import functools
import multiprocessing
import matplotlib

def _multiprocess_validation_video(data: pd.DataFrame,
                                   bp_dict: dict,
                                   video_save_dir: str,
                                   settings: dict,
                                   video_path: str,
                                   video_meta_data: dict,
                                   gantt_setting: str or None,
                                   final_gantt: np.array or None,
                                   clf_data: np.array,
                                   clrs: list,
                                   clf_name: str,
                                   bouts_df: pd.DataFrame):

    dpi = plt.rcParams['figure.dpi']
    def create_gantt(bouts_df: pd.DataFrame,
                         clf_name: str,
                         image_index: int,
                         fps: int):

        fig, ax = plt.subplots(figsize=(final_gantt.shape[1]/dpi, final_gantt.shape[0]/dpi))
        matplotlib.font_manager._get_font.cache_clear()
        relRows = bouts_df.loc[bouts_df['End_frame'] <= image_index]

        for i, event in enumerate(relRows.groupby("Event")):
            data_event = event[1][["Start_time", "Bout_time"]]
            ax.broken_barh(data_event.values, (4, 4), facecolors='red')
        xLength = (round(image_index / fps)) + 1
        if xLength < 10: xLength = 10

        ax.set_xlim(0, xLength)
        ax.set_ylim([0, 12])
        ax.set_xlabel('Session (s)', fontsize=12)
        ax.set_ylabel(clf_name, fontsize=12)
        ax.set_title(f'{clf_name} GANTT CHART', fontsize=12)
        ax.set_yticks([])
        ax.yaxis.set_ticklabels([])
        ax.yaxis.grid(True)
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.array(np.uint8(np.array(canvas.renderer._renderer)))[:, :, :3]
        plt.close(fig)
        return img

    fourcc, font = cv2.VideoWriter_fourcc(*'mp4v'), cv2.FONT_HERSHEY_COMPLEX
    cap = cv2.VideoCapture(video_path)
    group = data['group'].iloc[0]
    start_frm, current_frm, end_frm = data.index[0], data.index[0], data.index[-1]
    video_save_path = os.path.join(video_save_dir, '{}.mp4'.format(str(group)))
    if gantt_setting is not None:
        writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data['fps'], (int(video_meta_data['width'] + final_gantt.shape[1]), int(video_meta_data['height'])))
    else:
        writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data['fps'], (video_meta_data['width'], video_meta_data['height']))
    cap.set(1, start_frm)

    while current_frm < end_frm:
        clf_frm_cnt = np.sum(clf_data[0:current_frm])
        ret, img = cap.read()
        if settings['pose']:
            for animal_cnt, (animal_name, animal_data) in enumerate(bp_dict.items()):
                for bp_cnt, bp in enumerate(range(len(animal_data['X_bps']))):
                    x_header, y_header = animal_data['X_bps'][bp], animal_data['Y_bps'][bp]
                    animal_cords = tuple(data.loc[current_frm, [x_header, y_header]])
                    cv2.circle(img, (int(animal_cords[0]), int(animal_cords[1])), 0, clrs[animal_cnt][bp_cnt], settings['styles']['circle size'])
        if settings['animal_names']:
            for animal_cnt, (animal_name, animal_data) in enumerate(bp_dict.items()):
                x_header, y_header = animal_data['X_bps'][0], animal_data['Y_bps'][0]
                animal_cords = tuple(data.loc[current_frm, [x_header, y_header]])
                cv2.putText(img, animal_name, (int(animal_cords[0]), int(animal_cords[1])), font, settings['styles']['font size'], clrs[animal_cnt][0], 1)
        target_timer = round((1 / video_meta_data['fps']) * clf_frm_cnt, 2)
        cv2.putText(img, 'Timer', (10, settings['styles']['space_scale']), font, settings['styles']['font size'], (0, 255, 0), 2)
        addSpacer = 2
        cv2.putText(img, (f'{clf_name} {target_timer}s'), (10, settings['styles']['space_scale'] * addSpacer), font, settings['styles']['font size'], (0, 0, 255), 2)
        addSpacer += 1
        cv2.putText(img, 'Ensemble prediction', (10, settings['styles']['space_scale'] * addSpacer), font, settings['styles']['font size'], (0, 255, 0), 2)
        addSpacer += 2
        if clf_data[current_frm] == 1:
            cv2.putText(img, clf_name, (10, + settings['styles']['space_scale'] * addSpacer), font, settings['styles']['font size'], (2, 166, 249), 2)
            addSpacer += 1
        if gantt_setting == 'Gantt chart: final frame only (slightly faster)':
            img = np.concatenate((img, final_gantt), axis=1)
        elif gantt_setting == 'Gantt chart: video':
            gantt_img = create_gantt(bouts_df, clf_name, current_frm, video_meta_data['fps'], 'Behavior gantt chart')
            img = np.concatenate((img, gantt_img), axis=1)

        writer.write(np.uint8(img))
        current_frm += 1
        print('Multi-processing video frame {} on core {}...'.format(str(current_frm), str(group)))

    cap.release()
    writer.release()

    return group

class ValidateModelOneVideoMultiprocess(ConfigReader):
    """
    Class for creating classifier validation video for a single input video. Results are stored in the
    `project_folder/frames/output/validation directory`.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    feature_file_path: str
        path to SimBA file (parquet or CSV) containing pose-estimation and feature fields.
    model_path: str
        path to pickled classifier object
    discrimination_threshold: float
        classification threshold
    shortest_bout: int
        Allowed classified bout length expressed in milliseconds. E.g., `1000` will shift frames classified
        as containing the behavior, but occuring in a bout shorter than `1000`, from `target present to `target absent`.
    create_gantt: str
        If SimBA should create gantt charts alongside the validation video. OPTIONS: 'None', 'Gantt chart: final frame only (slightly faster)',
        'Gantt chart: video'.
    settings: dict
        User style settings for video. E.g., {'pose': True, 'animal_names': True, 'styles': None}

    """

    def __init__(self,
                 config_path: str,
                 feature_file_path: str,
                 model_path: str,
                 discrimination_threshold: float,
                 shortest_bout: int,
                 cores:int,
                 create_gantt: str,
                 settings: None=None):

        super().__init__(config_path=config_path)
        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        _, self.feature_filename, ext = get_fn_ext(feature_file_path)
        self.discrimination_threshold, self.cores, self.shortest_bout, self.create_gantt, self.settings = float(discrimination_threshold), cores, shortest_bout, create_gantt, settings
        if not os.path.exists(self.single_validation_video_save_dir): os.makedirs(self.single_validation_video_save_dir)
        _, _, self.fps = read_video_info(self.video_info_df, self.feature_filename)
        self.clf_name = os.path.basename(model_path).replace('.sav', '')
        self.video_path = find_video_of_file(self.video_dir, self.feature_filename)
        self.clf_data_save_path = os.path.join(self.clf_data_validation_dir, self.feature_filename + '.csv')
        self.video_meta_data = get_video_meta_data(video_path=self.video_path)
        self.clf = pickle.load(open(model_path, 'rb'))
        self.in_df = read_df(feature_file_path, self.file_type)
        self.temp_dir = os.path.join(self.single_validation_video_save_dir, 'temp')
        self.video_save_path = os.path.join(self.single_validation_video_save_dir, self.feature_filename + '.mp4')
        if not os.path.exists(self.temp_dir): os.makedirs(self.temp_dir)

    def __run_clf(self):
        self.prob_col_name = f'Probability_{self.clf_name}'
        self.in_df[self.prob_col_name] = self.clf.predict_proba(drop_bp_cords(self.in_df, self.config_path))[:, 1]
        self.in_df[self.clf_name] = np.where(self.in_df[self.prob_col_name] > self.discrimination_threshold, 1, 0)



    def __plug_bouts(self):
        self.data_df = plug_holes_shortest_bout(data_df=self.in_df, clf_name=self.clf_name, fps=self.fps, shortest_bout=self.shortest_bout)

    def __save(self):
        save_df(self.data_df, self.file_type, self.clf_data_save_path)
        print(f'Predictions created for video {self.feature_filename}...')

    def __index_df_for_multiprocessing(self, data: list) -> list:
        for cnt, df in enumerate(data):
            df['group'] = cnt
        return data

    def __create_video(self):
        self.final_gantt_img = None
        self.bouts_df = None
        if self.create_gantt is not None:
            self.bouts_df = get_bouts_for_gantt(data_df=self.data_df, clf_name=self.clf_name, fps=self.fps)
            self.final_gantt_img = create_gantt_img(self.bouts_df, self.clf_name, len(self.data_df), self.fps, 'Behavior gantt chart (entire session)')
            self.final_gantt_img = resize_gantt(self.final_gantt_img, self.video_meta_data['height'])

        if self.settings['styles'] is None:
            self.settings['styles'] = {}
            space_scaler, radius_scaler, resolution_scaler, font_scaler = 60, 20, 1500, 1.5
            max_dim = max(self.video_meta_data['width'], self.video_meta_data['height'])
            self.settings['styles']['circle size'] = int(radius_scaler / (resolution_scaler / max_dim))
            self.settings['styles']['font size'] = float(font_scaler / (resolution_scaler / max_dim))
            self.settings['styles']['space_scale'] = int(space_scaler / (resolution_scaler / max_dim))

        data = np.array_split(self.in_df, self.cores)
        frm_per_core = data[0].shape[0]
        data = self.__index_df_for_multiprocessing(data=data)
        with multiprocessing.Pool(self.cores, maxtasksperchild=self.maxtasksperchild) as pool:
            constants = functools.partial(_multiprocess_validation_video,
                                          bp_dict=self.animal_bp_dict,
                                          video_save_dir=self.temp_dir,
                                          settings=self.settings,
                                          video_meta_data=self.video_meta_data,
                                          video_path=self.video_path,
                                          gantt_setting=self.create_gantt,
                                          final_gantt=self.final_gantt_img,
                                          clf_data=self.data_df[self.clf_name].values,
                                          clrs=self.clr_lst,
                                          clf_name=self.clf_name,
                                          bouts_df=self.bouts_df)
            for cnt, result in enumerate(pool.imap(constants, data, chunksize=self.multiprocess_chunksize)):
                print('Image {}/{}, Video {}...'.format(str(int(frm_per_core * (result+1))), str(len(self.data_df)), self.feature_filename))
            print('Joining {} multiprocessed video...'.format(self.feature_filename))
        concatenate_videos_in_folder(in_folder=self.temp_dir, save_path=self.video_save_path)
        self.timer.stop_timer()
        pool.terminate()
        pool.join()
        print(f'SIMBA COMPLETE: Video {self.feature_filename} complete (elapsed time: {self.timer.elapsed_time_str}s).')

    def run(self):
        self.__run_clf()
        if self.shortest_bout > 1:
            self.__plug_bouts()
        self.__save()
        self.__create_video()

# test = ValidateModelOneVideoMultiprocess(config_path=r'/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/project_config.ini',
#                              feature_file_path=r'/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/csv/features_extracted/SF2.csv',
#                              model_path='/Users/simon/Desktop/envs/troubleshooting/naresh/models/generated_models/Top.sav',
#                              discrimination_threshold=0.6,
#                              shortest_bout=50,
#                              cores=6,
#                              settings={'pose': True, 'animal_names': True, 'styles': None},
#                              create_gantt='Gantt chart: video')
# test.run()

#  final frame only (slightly faster), (494, 1318, 3)


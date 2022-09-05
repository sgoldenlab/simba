import time
from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file)
from sklearn.cluster import KMeans
import os, glob
import itertools
import pandas as pd
from simba.rw_dfs import read_df
from simba.drop_bp_cords import get_fn_ext
from simba.misc_tools import find_video_of_file, get_video_meta_data
import cv2
import numpy as np
import multiprocessing
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.pybursts import kleinberg_burst_detection
import functools
import time

def get_intensity_scores_in_rois(frm_list: list=None,
                                 video_path: str = None,
                                 rectangles_df: pd.DataFrame = None,
                                 circles_df: pd.DataFrame = None,
                                 polygon_df: pd.DataFrame = None,
                                 cue_light_names: list=None
                                 ):
    cap = cv2.VideoCapture(video_path)
    start, end = frm_list[0], frm_list[-1]
    cap.set(1, start)
    frm_cnt = start
    results_dict = {}
    while frm_cnt <= end:
        _, img = cap.read()
        results_dict[frm_cnt] = {}
        for idx, rectangle in rectangles_df.iterrows():
            roi_image = img[rectangle['topLeftY']:rectangle['Bottom_right_Y'], rectangle['topLeftX']:rectangle['Bottom_right_X']]
            results_dict[frm_cnt][rectangle['Name']] = int(np.average(np.linalg.norm(roi_image, axis=2)) / np.sqrt(3))
        frm_cnt+=1
    return results_dict

class CueLightAnalyzer(object):
    def __init__(self,
                 config_path: str=None,
                 in_dir: str=None,
                 cue_light_names: list=None):

        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.logs_path, self.video_dir = os.path.join(self.project_path, 'logs'), os.path.join(self.project_path, 'videos')
        self.cue_light_names, self.in_dir = cue_light_names, in_dir
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.vid_info_df = read_video_info_csv(os.path.join(self.logs_path, 'video_info.csv'))
        self.files_found = glob.glob(self.in_dir + '/*' + self.file_type)
        self.read_roi_dfs()

    def read_roi_dfs(self):
        if not os.path.isfile(os.path.join(self.logs_path, 'measures', 'ROI_definitions.h5')):
            raise FileNotFoundError(print('No ROI definitions were found in your SimBA project. Please draw some ROIs before analyzing your ROI data'))
        else:
            self.roi_h5_path = os.path.join(self.logs_path, 'measures', 'ROI_definitions.h5')
            self.rectangles_df = pd.read_hdf(self.roi_h5_path, key='rectangles')
            self.circles_df = pd.read_hdf(self.roi_h5_path, key='circleDf')
            self.polygon_df = pd.read_hdf(self.roi_h5_path, key='polygons')
            self.shape_names = list(itertools.chain(self.rectangles_df['Name'].unique(), self.circles_df['Name'].unique(), self.polygon_df['Name'].unique()))

    def calculate_descriptive_statistics(self):
        self.light_descriptive_statistics = {}
        for shape_name in self.shape_names:
            self.light_descriptive_statistics[shape_name] = {}
            self.light_descriptive_statistics[shape_name]['frame_by_frame'] = np.full((self.video_meta_data['frame_count']), -1)
            self.light_descriptive_statistics[shape_name]['descriptive_statistics'] = {}
        for cnt, (k, v) in enumerate(self.intensity_results.items()):
            for shape_name in self.shape_names:
                self.light_descriptive_statistics[shape_name]['frame_by_frame'][cnt] = v[shape_name]
        for shape_name in self.shape_names:
            self.light_descriptive_statistics[shape_name]['kmeans'] = KMeans(n_clusters=2, random_state=0).fit_predict(self.light_descriptive_statistics[shape_name]['frame_by_frame'].reshape(-1, 1))
            for i in list(range(0, 2)):
                self.light_descriptive_statistics[shape_name]['descriptive_statistics']['mean_cluster_{}'.format(str(i))] = np.mean(self.light_descriptive_statistics[shape_name]['frame_by_frame'][np.argwhere(self.light_descriptive_statistics[shape_name]['kmeans'] == i).flatten()])
                self.light_descriptive_statistics[shape_name]['descriptive_statistics']['std_cluster_{}'.format(str(i))] = np.std(self.light_descriptive_statistics[shape_name]['frame_by_frame'][np.argwhere(self.light_descriptive_statistics[shape_name]['kmeans'] == i).flatten()])
            self.light_descriptive_statistics[shape_name]['ON_kmeans_cluster'] = int(max(self.light_descriptive_statistics[shape_name]['descriptive_statistics'], key=self.light_descriptive_statistics[shape_name]['descriptive_statistics'].get)[-1])
            self.light_descriptive_statistics[shape_name]['ON_FRAMES'] = np.argwhere(self.light_descriptive_statistics[shape_name]['kmeans'] == self.light_descriptive_statistics[shape_name]['ON_kmeans_cluster']).flatten()

    def perform_kleinberg_smoothing(self):
        for shape_name in self.shape_names:
            self.light_descriptive_statistics[shape_name]['ON_FRAMES'] = pd.DataFrame(kleinberg_burst_detection(offsets=self.light_descriptive_statistics[shape_name]['ON_FRAMES'], s=2, gamma=0.1), columns = ['Hierarchy', 'Start', 'Stop'])

    def insert_light_data(self):
        for shape_name in self.shape_names:
            self.data_df.loc[list(self.light_descriptive_statistics[shape_name]['ON_FRAMES']), shape_name] = 1

    def analyze_files(self):
        for file_cnt, file_path in enumerate(self.files_found):
            self.data_df = read_df(file_path, self.file_type)
            _, self.video_name, _ = get_fn_ext(file_path)
            video_settings, pix_per_mm, self.fps = read_video_info(self.vid_info_df, self.video_name)
            self.video_recs = self.rectangles_df.loc[(self.rectangles_df['Video'] == self.video_name) & (self.rectangles_df['Name'].isin(self.cue_light_names))]
            self.video_circs = self.circles_df.loc[(self.circles_df['Video'] == self.video_name)  & (self.circles_df['Name'].isin(self.cue_light_names))]
            self.video_polys = self.polygon_df.loc[(self.polygon_df['Video'] == self.video_name) & (self.polygon_df['Name'].isin(self.cue_light_names))]
            self.shape_names = list(itertools.chain(self.rectangles_df['Name'].unique(), self.circles_df['Name'].unique(),self.polygon_df['Name'].unique()))
            self.video_path = find_video_of_file(self.video_dir, self.video_name)
            self.video_meta_data = get_video_meta_data(self.video_path)
            self.frm_lst = list(range(0, self.video_meta_data['frame_count'], 1))
            self.frame_chunks = np.array_split(self.frm_lst, int(self.video_meta_data['frame_count'] / self.fps))
            imgs_peer_loop = len(self.frame_chunks[0])
            self.intensity_results = {}
            start_time = time.time()
            with multiprocessing.pool.Pool(8, maxtasksperchild=75) as pool:
                functools.partial(get_intensity_scores_in_rois, b=self.video_recs, c=self.circles_df, d=self.video_polys)
                constants = functools.partial(get_intensity_scores_in_rois,
                                              video_path=self.video_path,
                                              rectangles_df=self.video_recs,
                                              circles_df=self.circles_df,
                                              polygon_df=self.video_polys,
                                              cue_light_names=self.cue_light_names)

                for cnt, result in enumerate(pool.imap(constants, self.frame_chunks, chunksize=10)):
                    self.intensity_results.update(result)
                    print('Image {}/{}, Video {}/{}...'.format(str(int(imgs_peer_loop*cnt)), str(len(self.data_df)), str(file_cnt+1), str(len(self.files_found))))
            pool.terminate()
            pool.join()
            self.calculate_descriptive_statistics()
            #self.perform_kleinberg_smoothing()
            self.insert_light_data()

            elapsed_time = str(round(time.time() - start_time, 2)) + 's'
            print('SIMBA COMPLETE. Elapsed time {}'.format(elapsed_time))


test = CueLightAnalyzer(config_path='/Users/simon/Desktop/troubleshooting/light_analyzer/project_folder/project_config.ini',
                        in_dir='/Users/simon/Desktop/troubleshooting/light_analyzer/project_folder/csv/outlier_corrected_movement_location',
                        cue_light_names=['Cue_light'])
test.analyze_files()



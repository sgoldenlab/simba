import time
from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file)
from sklearn.cluster import KMeans
import os, glob
import itertools
import pandas as pd
from simba.rw_dfs import read_df, save_df
from simba.drop_bp_cords import (get_fn_ext,
                                 create_body_part_dictionary)
from simba.misc_tools import (find_video_of_file,
                              get_video_meta_data,
                              detect_bouts,
                              find_core_cnt,
                              check_multi_animal_status)
from simba.drop_bp_cords import getBpNames
import cv2
import numpy as np
import multiprocessing
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
import functools
import time
import platform
from shapely.geometry import Polygon
from joblib import Parallel, delayed



def get_intensity_scores_in_rois(frm_list: list=None,
                                 video_path: str = None,
                                 rectangles_df: pd.DataFrame = None,
                                 polygon_df: pd.DataFrame = None,
                                 circles_df: pd.DataFrame = None):
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
        for idx, polygon in polygon_df.iterrows():
            x,y,w,h = cv2.boundingRect(polygon['vertices'])
            roi_img = img[y:y + h, x:x + w].copy()
            pts = polygon['vertices'] - polygon['vertices'].min(axis=0)
            mask = np.zeros(roi_img.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
            dst = cv2.bitwise_and(roi_img, roi_img, mask=mask)
            bg = np.ones_like(roi_img, np.uint8)
            cv2.bitwise_not(bg, bg, mask=mask)
            roi_image = bg + dst
            results_dict[frm_cnt][polygon['Name']] = int(np.average(np.linalg.norm(roi_image, axis=2)) / np.sqrt(3))
        for idx, circle in circles_df.iterrows():
            roi_img = img[circle['centerY']:(circle['centerY'] + 2 * circle['radius']), circle['centerX']:(circle['centerX'] + 2 * circle['radius'])]
            mask = np.zeros(roi_img.shape[:2], np.uint8)
            circle_img = cv2.circle(mask, (circle['centerX'], circle['centerY']), circle['radius'], (255, 255, 255), thickness=-1)
            dst = cv2.bitwise_and(roi_img, roi_img, mask=circle_img)
            bg = np.ones_like(roi_img, np.uint8)
            cv2.bitwise_not(bg, bg, mask=mask)
            roi_image = bg + dst
            results_dict[frm_cnt][circle['Name']] = int(np.average(np.linalg.norm(roi_image, axis=2)) / np.sqrt(3))
        frm_cnt+=1
    return results_dict

class CueLightAnalyzer(object):
    """
    Class for analyzing when cue lights are in ON and OFF states. Results are stored in the
    `project_folder/csv/cue_lights` cue lights directory.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    in_dir: str
        directory holding pose-estimation data. E.g., `project_folder/csv/outlier_corrected_movement_location`
    cue_light_names: list
        Names of cue lights, as defined in the SimBA ROI interface.

    Notes
    ----------
    `Cue light tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md>`__.

    Examples
    ----------
    >>> cue_light_analyzer = CueLightAnalyzer(config_path='MyProjectConfig', in_dir='project_folder/csv/outlier_corrected_movement_location', cue_light_names=['Cue_light'])
    >>> cue_light_analyzer.analyze_files()
    """

    def __init__(self,
                 config_path: str,
                 in_dir: str,
                 cue_light_names: list):




        if len(cue_light_names) == 0:
            print('SIMBA ERROR: Please select one or more cue lights')
            raise ValueError('SIMBA ERROR: Please select one or more cue lights')

        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.logs_path, self.video_dir = os.path.join(self.project_path, 'logs'), os.path.join(self.project_path, 'videos')
        self.out_dir = os.path.join(self.project_path, 'csv', 'cue_lights')
        if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)
        self.cue_light_names, self.in_dir = cue_light_names, in_dir
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.vid_info_df = read_video_info_csv(os.path.join(self.logs_path, 'video_info.csv'))
        self.files_found = glob.glob(self.in_dir + '/*' + self.file_type)
        self.x_cols, self.y_cols, self.p_cols = getBpNames(config_path)
        self.no_animals = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.no_animals, self.x_cols, self.y_cols, self.p_cols, [])
        if len(self.files_found) == 0:
            print('SIMBA ERROR: Zero tracking files detected in the "project_folder/csv/outlier_corrected_movement_location" directory')
            raise ValueError('SIMBA ERROR: Zero tracking files detected in the "project_folder/csv/outlier_corrected_movement_location" directory')
        _, self.cpu_cnt_to_use = find_core_cnt()
        self.maxtasksperchild = 10
        self.chunksize = 1
        self.read_roi_dfs()
        print('Processing {} cue light(s) in {} data file(s)...'.format(str(len(cue_light_names)), str(len(self.files_found))))

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
        for shape_name in self.cue_light_names:
            self.light_descriptive_statistics[shape_name] = {}
            self.light_descriptive_statistics[shape_name]['frame_by_frame'] = np.full((self.video_meta_data['frame_count']), -1)
            self.light_descriptive_statistics[shape_name]['descriptive_statistics'] = {}
        for cnt, (k, v) in enumerate(self.intensity_results.items()):
            for shape_name in self.cue_light_names:
                self.light_descriptive_statistics[shape_name]['frame_by_frame'][cnt] = v[shape_name]
        for shape_name in self.cue_light_names:
            self.light_descriptive_statistics[shape_name]['kmeans'] = KMeans(n_clusters=2, random_state=0).fit_predict(self.light_descriptive_statistics[shape_name]['frame_by_frame'].reshape(-1, 1))
            for i in list(range(0, 2)):
                self.light_descriptive_statistics[shape_name]['descriptive_statistics']['mean_cluster_{}'.format(str(i))] = np.mean(self.light_descriptive_statistics[shape_name]['frame_by_frame'][np.argwhere(self.light_descriptive_statistics[shape_name]['kmeans'] == i).flatten()])
                self.light_descriptive_statistics[shape_name]['descriptive_statistics']['std_cluster_{}'.format(str(i))] = np.std(self.light_descriptive_statistics[shape_name]['frame_by_frame'][np.argwhere(self.light_descriptive_statistics[shape_name]['kmeans'] == i).flatten()])
            self.light_descriptive_statistics[shape_name]['ON_kmeans_cluster'] = int(max(self.light_descriptive_statistics[shape_name]['descriptive_statistics'], key=self.light_descriptive_statistics[shape_name]['descriptive_statistics'].get)[-1])
            self.light_descriptive_statistics[shape_name]['ON_FRAMES'] = np.argwhere(self.light_descriptive_statistics[shape_name]['kmeans'] == self.light_descriptive_statistics[shape_name]['ON_kmeans_cluster']).flatten()

    def remove_outlier_events(self, frame_threshold=2):
        for cue_light_name in self.cue_light_names:
            que_light_bouts = detect_bouts(data_df=self.data_df, target_lst=[cue_light_name], fps=int(self.fps))
            que_light_bouts['frame_length'] = que_light_bouts['End_frame'] - que_light_bouts['Start_frame']
            que_light_negative_outliers = que_light_bouts[que_light_bouts['frame_length'] <= frame_threshold]
            self.que_light_inliers = que_light_bouts[que_light_bouts['frame_length'] > frame_threshold]
            for idx, r in que_light_negative_outliers.iterrows():
                self.data_df.loc[r['Start_frame'] - 1:r['End_frame']+1,cue_light_name] = 0

    def insert_light_data(self):
        for shape_name in self.cue_light_names:
            self.data_df.loc[list(self.light_descriptive_statistics[shape_name]['ON_FRAMES']), shape_name] = 1
        self.data_df = self.data_df.fillna(0)

    def analyze_files(self):
        start_time = time.time()
        for file_cnt, file_path in enumerate(self.files_found):
            self.data_df = read_df(file_path, self.file_type)
            _, self.video_name, _ = get_fn_ext(file_path)
            self.save_path = os.path.join(self.out_dir, self.video_name + '.' + self.file_type)
            video_settings, pix_per_mm, self.fps = read_video_info(self.vid_info_df, self.video_name)
            self.video_recs = self.rectangles_df.loc[(self.rectangles_df['Video'] == self.video_name) & (self.rectangles_df['Name'].isin(self.cue_light_names))]
            self.video_circs = self.circles_df.loc[(self.circles_df['Video'] == self.video_name)  & (self.circles_df['Name'].isin(self.cue_light_names))]
            self.video_polys = self.polygon_df.loc[(self.polygon_df['Video'] == self.video_name) & (self.polygon_df['Name'].isin(self.cue_light_names))]
            self.shape_names = list(itertools.chain(self.rectangles_df['Name'].unique(), self.circles_df['Name'].unique(),self.polygon_df['Name'].unique()))
            self.video_path = find_video_of_file(self.video_dir, self.video_name)
            self.video_meta_data = get_video_meta_data(self.video_path)

            #self.frm_lst = list(range(0, len(self.data_df), 1))
            self.frm_lst = list(range(0, self.video_meta_data['frame_count'], 1))
            #self.frame_chunks = np.array_split(self.frm_lst, len(self.data_df) / self.fps)
            self.frame_chunks = np.array_split(self.frm_lst, int(self.video_meta_data['frame_count'] / self.fps))
            imgs_peer_loop = len(self.frame_chunks[0])
            self.intensity_results = {}

            with multiprocessing.pool.Pool(self.cpu_cnt_to_use, maxtasksperchild=self.maxtasksperchild) as pool:
                functools.partial(get_intensity_scores_in_rois, b=self.video_recs, c=self.circles_df, d=self.video_polys)
                constants = functools.partial(get_intensity_scores_in_rois,
                                              video_path=self.video_path,
                                              rectangles_df=self.video_recs,
                                              circles_df=self.circles_df,
                                              polygon_df=self.video_polys)

                for cnt, result in enumerate(pool.imap(constants, self.frame_chunks, chunksize=self.chunksize)):
                    self.intensity_results.update(result)
                    print('Image {}/{}, Video {}/{}...'.format(str(int(imgs_peer_loop*cnt)), str(len(self.data_df)), str(file_cnt+1), str(len(self.files_found))))
            pool.terminate()
            pool.join()
            self.calculate_descriptive_statistics()
            self.insert_light_data()
            self.remove_outlier_events()
            save_df(self.data_df, self.file_type, self.save_path)
        elapsed_time = str(round(time.time() - start_time, 2)) + 's'
        print('SIMBA CUE LIGHT ANALYSIS COMPLETE: Analysed {} files. Data stored in project_folder/csv/cue_lights. Elapsed time {}'.format(str(len(self.files_found)), elapsed_time))

# test = CueLightAnalyzer(config_path='/Users/simon/Desktop/troubleshooting/light_analyzer/project_folder/project_config.ini',
#                         in_dir='/Users/simon/Desktop/troubleshooting/light_analyzer/project_folder/csv/outlier_corrected_movement_location',
#                         cue_light_names=['Cue_light'])
# test.analyze_files()



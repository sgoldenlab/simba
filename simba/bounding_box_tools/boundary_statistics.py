import pandas as pd
from simba.read_config_unit_tests import read_config_file, read_config_entry
from simba.drop_bp_cords import create_body_part_dictionary, getBpNames
from simba.misc_tools import check_multi_animal_status
from simba.rw_dfs import read_df, save_df
from joblib import Parallel, delayed
from shapely.geometry import Point
from copy import deepcopy
from collections import defaultdict
import os
import pickle

class BoundaryStatisticsCalculator(object):
    def __init__(self,
                 config_path: str):

        self.config, self.config_path = read_config_file(ini_path=config_path), config_path
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.anchored_roi_path = os.path.join(self.project_path, 'logs', 'anchored_rois.pickle')
        self.input_dir = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.save_folder = os.path.join(self.project_path, 'csv', 'anchored_roi_data')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        if not os.path.isfile(self.anchored_roi_path):
            print('No anchored ROI data detected. Extract anchored ROIs before computing statistics')
            raise FileNotFoundError()
        with open(self.anchored_roi_path, 'rb') as fp: self.polygons = pickle.load(fp)
        self.no_animals = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, list(self.multi_animal_id_lst), self.no_animals, list(self.x_cols), list(self.y_cols), [], [])
        self.calculate_statistics()

    def _find_intersections(self,
                            animal_roi: list,
                            other_animals: dict):
        results = []
        for first_animal, second_animal in zip(animal_roi, other_animals):
            results.append(first_animal.intersects(second_animal))
        return results

    def _find_points_in_roi(self,
                             animal_roi: list,
                             second_animal_bps: list):
        results = []
        for polygon_cnt, polygon in enumerate(animal_roi):
            frm_results = []
            for k, v in second_animal_bps[polygon_cnt].items():
                frm_results.append(Point(v).within(polygon))
            results.append(frm_results)
        return results

    def _sort_keypoint_results(self,
                              first_animal_name: str,
                              second_animal_name: str):
        results = defaultdict(list)
        for frm in self.results:
            for body_part_cnt, body_part in enumerate(frm):
                body_part_name = self.animal_bp_dict[second_animal_name]['X_bps'][body_part_cnt][:-4]
                results['{}:{}:{}'.format(first_animal_name, second_animal_name, body_part_name)].append(int(body_part))
        return pd.DataFrame(results)

    def _sort_intersection_results(self):
        results = defaultdict(list)
        for animal_one_name, animal_one_data in self.intersecting_rois.items():
            for animal_two_name, animal_two_data in animal_one_data.items():
                results['{}:{}'.format(animal_one_name, animal_two_name)] = [int(x) for x in animal_two_data]
        return pd.DataFrame(results)

    def calculate_statistics(self):
        self.intersection_dfs = {}
        self.keypoint_dfs = {}
        for video_cnt, (video_name, video_data) in enumerate(self.polygons.items()):
            self.data_df = read_df(os.path.join(self.input_dir, video_name + '.' + self.file_type), self.file_type).astype(int)
            self.intersecting_rois = {}
            print('Calculate intersecting anchored ROIs...')
            for first_animal in self.animal_bp_dict.keys():
                first_animal_anchored_rois = [video_data[first_animal][i:i + 100] for i in range(0, len(video_data[first_animal]), 100)]
                self.intersecting_rois[first_animal] = {}
                for second_animal in {k: v for k, v in video_data.items() if k != first_animal}.keys():
                    second_animal_anchored_rois = [video_data[second_animal][i:i + 100] for i in range(0, len(video_data[second_animal]), 100)]
                    results = Parallel(n_jobs=5, verbose=2, backend="loky")(delayed(self._find_intersections)(i, j) for i,j in zip(first_animal_anchored_rois,second_animal_anchored_rois))
                    self.intersecting_rois[first_animal][second_animal] = [i for s in results for i in s]
            self.intersection_dfs[video_name] = self._sort_intersection_results()

            keypoints_df_lst = []
            for first_animal in self.animal_bp_dict.keys():
                first_animal_anchored_rois = [video_data[first_animal][i:i + 100] for i in range(0, len(video_data[first_animal]), 100)]
                for second_animal in {k: v for k, v in self.animal_bp_dict.items() if k != first_animal}.keys():
                    second_animal_name = deepcopy(second_animal)
                    second_animal_df_tuples = pd.DataFrame()
                    for x_col, y_col in zip(self.animal_bp_dict[second_animal]['X_bps'], self.animal_bp_dict[second_animal]['Y_bps']):
                        second_animal_df_tuples[x_col[:-4]] = list(zip(self.data_df[x_col], self.data_df[y_col]))
                    second_animal = second_animal_df_tuples.to_dict(orient='records')
                    second_animal = [second_animal[i:i + 100] for i in range(0, len(second_animal), 100)]
                    results = Parallel(n_jobs=5, verbose=2, backend="loky")(delayed(self._find_points_in_roi)(i, j) for i, j in zip(first_animal_anchored_rois, second_animal))
                    self.results = [i for s in results for i in s]
                    keypoints_df_lst.append(self._sort_keypoint_results(first_animal_name=first_animal, second_animal_name=second_animal_name))
            self.keypoint_dfs[video_name] = pd.concat(keypoints_df_lst, axis=1)

    def save_results(self):
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
        for video_name in self.keypoint_dfs.keys():
            save_path = os.path.join(self.save_folder, video_name + '.' + self.file_type)
            out_df = pd.concat([self.intersection_dfs[video_name], self.keypoint_dfs[video_name]], axis=1)
            save_df(df=out_df,file_type=self.file_type, save_path=save_path)

test = BoundaryStatisticsCalculator(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini')
test.save_results()
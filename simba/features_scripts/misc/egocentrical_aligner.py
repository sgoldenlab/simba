import glob
import itertools
from copy import deepcopy
import pandas as pd
import cv2
import numpy as np
from itertools import combinations
from numba import jit, prange
import math
from scipy.spatial import ConvexHull
from joblib import Parallel, delayed


class EgocentricalAlignmentFeaturizer(object):
    def __init__(self,
                 data_path: str,
                 anchor: str='Centroid',
                 fps: int=30):
        self.data_files = glob.glob(data_path + '/*.csv')
        self.anchor = (f'{anchor}_x', f'{anchor}_y')
        self.fps = fps
        self.img_size = (500, 500)
        self.rolling_window_sizes = {}
        for i in [1, 1.5, 3]:
            self.rolling_window_sizes[f'{str(i)}s'] = int(fps * i)
        for i in [2, 4, 10]:
            self.rolling_window_sizes[f'{str(1/i)}s'] = int(fps / i)
        self.run()

    def run(self):
        for file_path in self.data_files:
            df = pd.read_csv(file_path, header=[0,1,2], index_col=0)
            df.columns = df.columns.droplevel().map('_'.join)
            self.bp_headers, self.bp_dict = {}, {}
            for (i, j) in zip(['_x', '_y', '_p'], ['x', 'y', 'p']):
                self.bp_headers[j] = [x for x in df.columns if x.endswith(i)]
            df = df[self.bp_headers['x'] + self.bp_headers['y']]
            self.scaled_df = deepcopy(df)
            for bp in self.bp_headers['x']:
                self.bp_dict[bp.rstrip('_x')] = (bp, bp.rstrip('_x') + '_y')
            df['correction_x'] = df[self.anchor[0]] - (self.img_size[0]/2)
            df['correction_y'] = df[self.anchor[1]] - (self.img_size[1]/2)
            for c in self.bp_dict.values():
                self.scaled_df[c[0]] = self.scaled_df[c[0]] - df['correction_x']
                self.scaled_df[c[1]] = self.scaled_df[c[1]] - df['correction_y']
            self.scaled_df = self.scaled_df.fillna((self.img_size[0]/2))
            self.featurize()
            #self.visualize()

    def visualize(self):
        max_x, max_y = np.nanmax(self.scaled_df[self.bp_headers['x']].values), np.nanmax(self.scaled_df[self.bp_headers['y']].values)
        img = np.zeros(shape=[int(max_x), int(max_y), 3], dtype=np.uint8)
        for frm in range(len(self.scaled_df)):
            frm_data = self.scaled_df.iloc[frm].astype(int)
            frm_img = deepcopy(img)
            for bp_name, bp in self.bp_dict.items():
                x, y = frm_data[bp[0]], frm_data[bp[1]]
                cv2.circle(frm_img, (int(x), int(y)), 0, (255, 255, 0), 8)
            for bp_c in combinations(list(self.bp_dict.keys()), 2):
                bp_1_x, bp_1_y = self.bp_dict[bp_c[0]][0], self.bp_dict[bp_c[0]][1]
                bp_2_x, bp_2_y = self.bp_dict[bp_c[1]][0], self.bp_dict[bp_c[1]][1]
                point_one, point_two = (frm_data[bp_1_x], frm_data[bp_1_y]), (frm_data[bp_2_x], frm_data[bp_2_y])
                cv2.line(frm_img, point_one, point_two, (255, 255, 0), 1)
            cv2.imshow('img', frm_img)
            cv2.waitKey(33)

    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def three_point_angles(data: np.array):
        results = np.full((data.shape[0]), 0)
        for i in prange(data.shape[0]):
            angle = math.degrees(math.atan2(data[i][5] - data[i][3], data[i][4] - data[i][2]) - math.atan2(data[i][1] - data[i][3], data[i][0] - data[i][2]))
            if angle < 0:
                angle += 360
            results[i] = angle
        return results

    @staticmethod
    def subhull_calculator(data: np.array):
        results = np.full((len(data)), np.nan)
        data = np.reshape(data.values, (len(data), -1, 2))
        for cnt, i in enumerate(data):
            results[cnt] = ConvexHull(i).area
        return results.astype(int)

    @staticmethod
    @jit(nopython=True)
    def euclidean_distance(bp_1_x_vals, bp_2_x_vals, bp_1_y_vals, bp_2_y_vals):
        return np.sqrt((bp_1_x_vals - bp_2_x_vals) ** 2 + (bp_1_y_vals - bp_2_y_vals) ** 2)

    @staticmethod
    def convex_hull_calculator_mp(data: np.array):
        results = np.full((data.shape[0]), np.nan)
        data = np.reshape(data.values, (len(data), -1, 2))
        for cnt, i in enumerate(data):
            results[cnt] = ConvexHull(i).area
        return results.astype(int)

    def featurize(self):
        three_point_combinations = np.array(list(combinations(list(self.bp_dict.keys()), 3)))
        four_point_combinations = np.array(list(combinations(list(self.bp_dict.keys()), 4)))
        two_point_combinations = np.array(list(combinations(list(self.bp_dict.keys()), 2)))
        results = pd.DataFrame()
        split_data = np.array_split(self.scaled_df, 100)
        hull_area = Parallel(n_jobs=-1, verbose=0, backend="threading")(delayed(self.convex_hull_calculator_mp)(x) for x in split_data)
        results['hull_area'] = np.concatenate(hull_area).ravel().tolist()
        for c in three_point_combinations:
            col_names = list(sum([(x + '_x', y + '_y') for (x,y) in zip(c, c)], ()))
            split_data = np.array_split(self.scaled_df[col_names], 100)
            three_point_hull = Parallel(n_jobs=-1, verbose=0, backend="threading")(delayed(self.subhull_calculator)(x) for x in split_data)
            results[f'hull_{c[0]}_{c[1]}_{c[2]}'] = np.concatenate(three_point_hull).ravel().tolist()
            results[f'angle_{c[0]}_{c[1]}_{c[2]}'] = self.three_point_angles(data=self.scaled_df[col_names].values)
        for c in four_point_combinations:
            col_names = list(sum([(x + '_x', y + '_y') for (x,y) in zip(c, c)], ()))
            split_data = np.array_split(self.scaled_df[col_names], 100)
            four_point_hull = Parallel(n_jobs=-1, verbose=0, backend="threading")(delayed(self.subhull_calculator)(x) for x in split_data)
            results[f'hull_{c[0]}_{c[1]}_{c[2]}_{c[3]}'] = np.concatenate(four_point_hull).ravel().tolist()
        for c in two_point_combinations:
            col_names = list(sum([(x + '_x', y + '_y') for (x,y) in zip(c, c)], ()))
            results[f'distance_{c[0]}_{c[1]}'] = self.euclidean_distance(self.scaled_df[col_names[0]].values,
                                                                         self.scaled_df[col_names[2]].values,
                                                                         self.scaled_df[col_names[1]].values,
                                                                         self.scaled_df[col_names[3]].values)

        for c, t in (list(itertools.product(results.columns, self.rolling_window_sizes.keys()))):
            results[f'{c}_rolling_{t}_window_mean'] = results[c].rolling(int(self.rolling_window_sizes[t]), min_periods=1).mean()
            results[f'{c}_rolling_{t}_window_stdev'] = results[c].rolling(int(self.rolling_window_sizes[t]), min_periods=1).std()
            results[f'{c}_rolling_{t}_window_kurt'] = results[c].rolling(int(self.rolling_window_sizes[t])).kurt().fillna(-1)
        self.results = results.fillna(0).astype(int)

#aligner = EgocentricalAlignmentFeaturizer(data_path= '/Users/simon/Desktop/envs/simba_dev/tests/test_data/mouse_open_field/project_folder/csv/input_csv')
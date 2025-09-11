import glob
import os
import time
import warnings
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from numba import jit
from numba.core.errors import (NumbaDeprecationWarning,
                               NumbaPendingDeprecationWarning)
from tqdm import tqdm

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

WINDOW_SIZES = [5, 10, 20, 40]

class GerbilFeaturizer(object):
    """
    Compute features from SLEAP pose-estimation data in NPY format with two animals and a single body-part tracked per animal.

    .. image:: _static/GerbilFeaturizer.webp
       :width: 400
       :align: center

    :param str folder_path: folder path containing SLP tracking data in .npy format.
    :param str out_path: folder path storing featurized data in multiple .parquet files.

    :example:
    >>> gerbil_featurizer = GerbilFeaturizer(in_path='_gerbil_data/pose_estimation_data', out_path='gerbil_data/featurized_data')
    >>> gerbil_featurizer.extract_features()
    >>> gerbil_featurizer.save()

    :references:
       .. [1] Mitelut, C, M Diez Castro, Re Peterson, M Gonçalves, J Li, Mm Gamer, Sro Nilsson, Td Pereira, and Dh Sanes. “A Behavioral Roadmap for the Development of Agency in the Rodent.” Animal Behavior and Cognition, November 13, 2023. https://doi.org/10.1101/2023.11.10.566632.
       .. [2] Mitelut, Catalin, Marielisa Diez Castro, Ralph E. Peterson, et al. “Continuous Monitoring and Machine Vision Reveals That Developing Gerbils Exhibit Structured Social Behaviors Prior to the Emergence of Autonomy.” PLOS Biology 23, no. 9 (2025): e3003348. https://doi.org/10.1371/journal.pbio.3003348.
    """

    def __init__(self,
                 in_path: str,
                 out_path: str):

        self.in_path = in_path
        self.out_path = out_path
        self.file_paths = glob.glob(in_path + '/*')
        self.feature_frame_windows = WINDOW_SIZES
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')

    def check_file_is_readable(self, file_path):
        if not os.access(file_path, os.R_OK):
            raise FileNotFoundError('{} is not readable.'.format(file_path))

    @staticmethod
    @jit(nopython=True)
    def calc_individual_animal_movements_in_time_windows(input_array=np.ndarray, frm_windows=List[int]) -> (np.ndarray, np.ndarray, np.ndarray):
        '''
        Jitted compute of individual frame-by-frame pixel movement and aggregate movement statistics (mean, sum) in rolling time-windows for each animal.

        :param np.ndarray input_array: array of size len(frames) x 2 representing body-part coordinates
        :param frm_windows: list of ints representing frame window sizes to calculate aggregate statistics within

        :return: np.ndarray single_frm_move: array of size len(frames) x 1 representing frame-by-frame body-part movements in pixels
        :return: np.ndarray agg_frm_move_mean: array of size len(frames) x len(frm_windows) representing mean frame-by-frame body-part movements in rolling time-windows
        :return: np.ndarray agg_frm_move_sum: array of size len(frames) x len(frm_windows) representing summed frame-by-frame body-part movements in rolling time-windows
        '''

        single_frm_move = np.zeros((len(input_array), 1))
        agg_frm_move_mean = np.zeros((len(input_array), len(frm_windows)))
        agg_frm_move_sum = np.zeros((len(input_array), len(frm_windows)))
        for frm_idx in range(1, len(input_array)):
            frme_x_y, prior_frm_x_y = input_array[frm_idx], input_array[frm_idx - 1]
            single_frm_move[frm_idx] = np.sqrt(
                (prior_frm_x_y[0] - frme_x_y[0]) ** 2 + (prior_frm_x_y[1] - frme_x_y[1]) ** 2)
        for frm_win_cnt, frm_win_size in enumerate(frm_windows):
            for frm_idx in range(frm_win_size, len(single_frm_move)):
                window_movement = single_frm_move[frm_idx - frm_win_size:frm_idx].flatten()
                agg_frm_move_mean[frm_idx, frm_win_cnt] = np.mean(window_movement)
                agg_frm_move_sum[frm_idx, frm_win_cnt] = np.sum(window_movement)

        return single_frm_move, agg_frm_move_mean, agg_frm_move_sum

    @staticmethod
    @jit(nopython=True)
    def calc_animal_distances_in_time_windows(input_array=np.ndarray, frm_windows=List[int]) -> (np.ndarray, np.ndarray, np.ndarray):

        """
        Jitted compute of individual frame-by-frame pixel distances and aggregate distance statistics (mean, sum) in rolling time-windows between the two animals.

        :param np.ndarray input_array: array of size len(frames) x 4 representing body-part coordinates of two animals [x1, y1, x2, y2].
        :param List[int] frm_windows: list of ints representing frame window sizes to calculate aggregate statistics within.

        :return: np.ndarray single_frm_move: array of size len(frames) x 1 representing frame-by-frame body-part distances in pixels
        :return: np.ndarray agg_frm_move_mean: array of size len(frames) x len(frm_windows) representing mean frame-by-frame body-part distances in rolling time-windows
        :return: np.nparray agg_frm_move_sum: array of size len(frames) x len(frm_windows) representing summed frame-by-frame body-part distances in rolling time-windows
        """

        single_frm_dists = np.zeros((len(input_array), 1))
        agg_frm_dists_mean = np.zeros((len(input_array), len(frm_windows)))
        agg_frm_dists_sum = np.zeros((len(input_array), len(frm_windows)))
        for frm_idx in range(0, len(input_array)):
            animal_1_x_y, animal_2_x_y = input_array[frm_idx][0:2], input_array[frm_idx][2:4]
            single_frm_dists[frm_idx] = np.sqrt(
                (animal_1_x_y[0] - animal_2_x_y[0]) ** 2 + (animal_1_x_y[1] - animal_2_x_y[1]) ** 2)
        for frm_win_cnt, frm_win_size in enumerate(frm_windows):
            for frm_idx in range(frm_win_size, len(single_frm_dists)):
                window_dist = single_frm_dists[frm_idx - frm_win_size:frm_idx].flatten()
                agg_frm_dists_mean[frm_idx, frm_win_cnt] = np.mean(window_dist)
                agg_frm_dists_sum[frm_idx, frm_win_cnt] = np.sum(window_dist)

        return single_frm_dists, agg_frm_dists_mean, agg_frm_dists_sum

    @staticmethod
    @jit(nopython=True)
    def calc_relative_data_in_time_windows(input_array=np.ndarray, frm_windows=List[int]) -> np.ndarray:
        """
        Jitted compute of individual frame-by-frame pixel distances and aggregate distance statistics (mean, sum) in rolling time-windows between the two animals.
        E.g., returns the relative distance between the two animals in the current frame vs 5, 10, 20, 40 frames earlier.

        :param np.ndarray input_array: array of size len(frames) x 1 representing a distances or movements of body-parts (one value per framde)
        :param List[int] frm_windows: list of ints representing frame window sizes to calculate aggregate statistics within

        :return: np.ndarray relative_dist_results: array of size len(frames) x len(frm_windows) representing deviation between the current window value from the preceding window value.
        """

        relative_dist_results = np.zeros((len(input_array), len(frm_windows)))
        for field_cnt in range(input_array.shape[1]):
            frm_window = frm_windows[field_cnt]
            data_arr = input_array[:, field_cnt]
            for frm_cnt in range(frm_window, data_arr.shape[0]):
                relative_dist_results[frm_cnt, field_cnt] = data_arr[frm_cnt] - data_arr[frm_cnt - frm_window]
        return relative_dist_results

    def extract_features(self) -> None:
        """ Main entry for feature extraction """
        self.data_results = {}
        for file_path in tqdm(self.file_paths):
            self.check_file_is_readable(file_path)
            filename = os.path.basename(file_path)
            data_arr = np.load(file_path)
            animal_1_frame_move, animal_1_mean_move, animal_1_sum_move = self.calc_individual_animal_movements_in_time_windows(input_array=data_arr[:, 0, :, 0], frm_windows=self.feature_frame_windows)
            animal_2_frame_move, animal_2_mean_move, animal_2_sum_move = self.calc_individual_animal_movements_in_time_windows( input_array=data_arr[:, 1, :, 0], frm_windows=self.feature_frame_windows)
            frame_dist, frame_dist_mean, frame_dist_sum = self.calc_animal_distances_in_time_windows(input_array=np.hstack([data_arr[:, 0, :, 0], data_arr[:, 1, :, 0]]),frm_windows=self.feature_frame_windows)
            relative_distances = self.calc_relative_data_in_time_windows(input_array=frame_dist_mean, frm_windows=self.feature_frame_windows)
            relative_move_animal_1 = self.calc_relative_data_in_time_windows(input_array=animal_1_frame_move, frm_windows=self.feature_frame_windows)
            relative_move_animal_2 = self.calc_relative_data_in_time_windows(input_array=animal_2_mean_move, frm_windows=self.feature_frame_windows)
            self.data_results[filename] = {}
            self.data_results[filename]['original_data'] = data_arr
            self.data_results[filename]['animal_1_cords'] = data_arr[:, 0, :, 0]
            self.data_results[filename]['animal_2_cords'] = data_arr[:, 1, :, 0]
            self.data_results[filename]['targets'] = data_arr[:, 0, :, 1][:, 1]

            self.data_results[filename]['features'] = np.hstack([animal_1_frame_move,
                                                                 animal_1_mean_move,
                                                                 animal_1_sum_move,
                                                                 animal_2_frame_move,
                                                                 animal_2_mean_move,
                                                                 animal_2_sum_move,
                                                                 frame_dist,
                                                                 frame_dist_mean,
                                                                 frame_dist_sum,
                                                                 relative_distances,
                                                                 relative_move_animal_1,
                                                                 relative_move_animal_2])

    def save(self):
        """Saves featurized data in parquet files (one per video) named after the video name indexed by the columns 'FRAME' and 'VIDEO' """

        for video_cnt, (video_name, video_data) in enumerate(self.data_results.items()):
            animal_df_1 = pd.DataFrame(video_data['animal_1_cords'], columns=['animal_1_x', 'animal_1_y'])
            animal_df_2 = pd.DataFrame(video_data['animal_2_cords'], columns=['animal_2_x', 'animal_2_y'])
            features_df = pd.DataFrame(video_data['features'])
            target_df = pd.DataFrame(video_data['targets'], columns=['target'])
            out_df = pd.concat([animal_df_1, animal_df_2, features_df, target_df], axis=1)
            out_df = out_df.reset_index().rename(columns={'index': 'FRAME'})
            out_df.insert(loc=0, column='VIDEO', value=video_name)
            out_df.columns = out_df.columns.map(str)
            save_path = os.path.join(self.out_path, f'features_{video_name.split(".")[0]}.parquet')
            out_df.fillna(0).to_parquet(save_path)
            print(f'Featurized data saved @ {save_path} (Video {video_cnt+1} / {len(self.data_results.keys())})')


# test = GerbilFeaturizer(in_path='gerbil_data/input',
#                                      out_path='gerbil_data/featurized_data_092223')
# start = time.time()
# test.extract_features()
# end = time.time()
# test.save()


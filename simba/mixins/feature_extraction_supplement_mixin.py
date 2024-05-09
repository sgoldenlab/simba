__author__ = "Simon Nilsson"

import glob
import itertools
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba import jit, prange, typed

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.timeseries_features_mixin import TimeseriesFeatureMixin
from simba.utils.checks import (check_float, check_if_dir_exists,
                                check_if_filepath_list_is_empty,
                                check_instance, check_str,
                                check_that_column_exist, check_valid_array,
                                check_valid_dataframe, check_valid_lst)
from simba.utils.data import detect_bouts
from simba.utils.errors import CountError, InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df


class FeatureExtractionSupplemental(FeatureExtractionMixin):
    """
    Additional feature extraction method not called by default feature extraction classes from ``simba.feature_extractors``.

    """

    def __init__(self):
        FeatureExtractionMixin.__init__(self)

    @staticmethod
    @jit(nopython=True)
    def _helper_euclidean_distance_timeseries_change(
        distances: np.ndarray, time_windows: np.ndarray, fps: int
    ):
        """
        Private jitted helper called by ``simba.mixins.feature_extraction_supplemental_mixin.FeatureExtractionSupplemental.euclidean_distance_timeseries_change``
        """
        results = np.full((distances.shape[0], time_windows.shape[0]), np.nan)
        for window_cnt in prange(time_windows.shape[0]):
            frms = int(time_windows[window_cnt] * fps)
            shifted_distances = np.copy(distances)
            shifted_distances[0:frms] = np.nan
            shifted_distances[frms:] = distances[:-frms]
            shifted_distances[np.isnan(shifted_distances)] = distances[
                np.isnan(shifted_distances)
            ]
            results[:, window_cnt] = distances - shifted_distances

        return results

    def euclidean_distance_timeseries_change(
        self,
        location_1: np.ndarray,
        location_2: np.ndarray,
        fps: int,
        px_per_mm: float,
        time_windows: np.ndarray = np.array([0.2, 0.4, 0.8, 1.6]),
    ) -> np.ndarray:
        """
        Compute the difference in distance between two points in the current frame versus N.N seconds ago. E.g.,
        computes if two points are traveling away from each other (positive output values) or towards each other
        (negative output values) relative to reference time-point(s)

        .. image:: _static/img/euclid_distance_change.png
           :width: 700
           :align: center

        :parameter ndarray location_1: 2D array of size len(frames) x 2 representing pose-estimated locations of body-part one
        :parameter ndarray location_2: 2D array of size len(frames) x 2 representing pose-estimated locations of body-part two
        :parameter int fps: Fps of the recorded video.
        :parameter float px_per_mm: The pixels per millimeter in the video.
        :parameter np.ndarray time_windows: Time windows to compare.
        :return np.array: Array of size location_1.shape[0] x time_windows.shape[0]

        :example:
        >>> location_1 = np.random.randint(low=0, high=100, size=(2000, 2)).astype('float32')
        >>> location_2 = np.random.randint(low=0, high=100, size=(2000, 2)).astype('float32')
        >>> distances = self.euclidean_distance_timeseries_change(location_1=location_1, location_2=location_2, fps=10, px_per_mm=4.33, time_windows=np.array([0.2, 0.4, 0.8, 1.6]))
        """
        distances = self.framewise_euclidean_distance(
            location_1=location_1, location_2=location_2, px_per_mm=px_per_mm
        )
        return self._helper_euclidean_distance_timeseries_change(
            distances=distances, fps=fps, time_windows=time_windows
        ).astype(int)

    @staticmethod
    @jit(nopython=True)
    def peak_ratio(data: np.ndarray, bin_size_s: int, fps: int):
        """
        Compute the ratio of peak values relative to number of values within each seqential
        time-period represented of ``bin_size_s`` seconds. Peak is defined as value is higher than
        in the prior observation (i.e., no future data is involved in comparison).

        .. image:: _static/img/peak_cnt.png
           :width: 700
           :align: center

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter int bin_size_s: The size of the buckets in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :return np.ndarray: Array of size data.shape[0] with peak counts as ratio of len(frames).


        :example:
        >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> FeatureExtractionSupplemental().peak_ratio(data=data, bin_size_s=1, fps=10)
        >>> [0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9]
        >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        >>> FeatureExtractionSupplemental().peak_ratio(data=data, bin_size_s=1, fps=10)
        >>> [0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.  0.  0.  0.  0.  0.  0.  0. 0.  0. ]
        """

        window_size, results = int(bin_size_s * fps), np.full((data.shape[0]), -1.0)
        data = np.split(data, list(range(window_size, data.shape[0], window_size)))
        start, end = 0, data[0].shape[0]
        for i in prange(len(data)):
            peak_cnt = 0
            if data[i][0] > data[i][1]:
                peak_cnt += 1
            if data[i][-1] > data[i][-2]:
                peak_cnt += 1
            for j in prange(1, len(data[i]) - 1):
                if data[i][j] > data[i][j - 1]:
                    peak_cnt += 1
            peak_ratio = peak_cnt / data[i].shape[0]
            results[start:end] = peak_ratio
            start, end = start + len(data[i]), end + len(data[i])
        return results

    @staticmethod
    @jit(nopython=True)
    def rolling_peak_count_ratio(data: np.ndarray, time_windows: np.ndarray, fps: int):

        results = np.full((data.shape[0], time_windows.shape[0]), -1.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            for j in prange(window_size, data.shape[0]):
                window_data = data[j - window_size : j]
                peak_cnt = 0
                if window_data[0] > window_data[1]:
                    peak_cnt += 1
                if window_data[-1] > window_data[-2]:
                    peak_cnt += 1
                for k in prange(1, len(window_data) - 1):
                    if window_data[j] > window_data[j - 1]:
                        peak_cnt += 1
                peak_ratio = peak_cnt / window_data.shape[0]
                results[j, i] = peak_ratio
        print(results)

    @staticmethod
    @jit(nopython=True)
    def rolling_categorical_switches_ratio(
        data: np.ndarray, time_windows: np.ndarray, fps: int
    ) -> np.ndarray:
        """
        Compute the ratio of in categorical feature switches within rolling windows.

        .. attention::
           Output for initial frames where [current_frm - window_size] < 0, are populated with ``0``.

        .. image:: _static/img/feature_switches.png
           :width: 700
           :align: center

        :parameter np.ndarray data: 1d array of feature values
        :parameter np.ndarray time_windows: Rolling time-windows as floats in seconds. E.g., [0.2, 0.4, 0.6]
        :parameter int fps: fps of the recorded video
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0] array

        :example:
        >>> data = np.array([0, 1, 1, 1, 4, 5, 6, 7, 8, 9])
        >>> FeatureExtractionSupplemental().rolling_categorical_switches_ratio(data=data, time_windows=np.array([1.0]), fps=10)
        >>> [[-1][-1][-1][-1][-1][-1][-1][-1][-1][ 0.7]]
        >>> data = np.array(['A', 'B', 'B', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        >>> FeatureExtractionSupplemental().rolling_categorical_switches_ratio(data=data, time_windows=np.array([1.0]), fps=10)
        >>> [[-1][-1][-1][-1][-1][-1][-1][-1][-1][ 0.7]]
        """

        results = np.full((data.shape[0], time_windows.shape[0]), -1.0)
        for time_window in prange(time_windows.shape[0]):
            jump_frms = int(time_windows[time_window] * fps)
            for current_frm in prange(jump_frms, data.shape[0] + 1):
                time_slice = data[current_frm - jump_frms : current_frm]
                current_value, unique_cnt = time_slice[0], 0
                for i in prange(1, time_slice.shape[0]):
                    if time_slice[i] != current_value:
                        unique_cnt += 1
                    current_value = time_slice[i]
                print(unique_cnt, time_slice.shape[0])
                results[current_frm - 1][time_window] = unique_cnt / time_slice.shape[0]
        return results

    @staticmethod
    @jit(nopython=True)
    def consecutive_time_series_categories_count(data: np.ndarray, fps: int):
        """
        Compute the count of consecutive milliseconds the feature value has remained static. For example,
        compute for how long in milleseconds the animal has remained in the current cardinal direction or the
        within an ROI.

        .. image:: _static/img/categorical_consecitive_time.png
           :width: 700
           :align: center

        :parameter np.ndarray data: 1d array of feature values
        :parameter int fps: Frame-rate of video.
        :returns np.ndarray: Array of size data.shape[0]

        :example:
        >>> data = np.array([0, 1, 1, 1, 4, 5, 6, 7, 8, 9])
        >>> FeatureExtractionSupplemental().consecutive_time_series_categories_count(data=data, fps=10)
        >>> [0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        >>> data = np.array(['A', 'B', 'B', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        >>> [0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        """

        results = np.full((data.shape[0]), 0.0)
        results[0] = 1
        for i in prange(1, data.shape[0]):
            if data[i] == data[i - 1]:
                results[i] = results[i - 1] + 1
            else:
                results[i] = 1

        return results / fps

    @staticmethod
    @jit(nopython=True)
    def rolling_horizontal_vs_vertical_movement(
        data: np.ndarray, pixels_per_mm: float, time_windows: np.ndarray, fps: int
    ) -> np.ndarray:
        """
        Compute the movement along the x-axis relative to the y-axis in rolling time bins.

        .. attention::
           Output for initial frames where [current_frm - window_size] < 0, are populated with ``0``.

        .. image:: _static/img/x_vs_y_movement.png
           :width: 700
           :align: center

        :parameter np.ndarray data: 2d array of size len(frames)x2 with body-part coordinates.
        :parameter int fps: FPS of the recorded video
        :parameter float pixels_per_mm: Pixels per millimeter of recorded video.
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0] array
        :parameter np.ndarray time_windows: Rolling time-windows as floats in seconds. E.g., [0.2, 0.4, 0.6]
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0]. Greater values denote greater movement on x-axis relative to y-axis.

        :example:
        >>> data = np.array([[250, 250], [250, 250], [250, 250], [250, 500], [500, 500], 500, 500]]).astype(float)
        >>> FeatureExtractionSupplemental().rolling_horizontal_vs_vertical_movement(data=data, time_windows=np.array([1.0]), fps=2, pixels_per_mm=1)
        >>> [[  -1.][   0.][   0.][-250.][ 250.][   0.]]
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0)
        for time_window in prange(time_windows.shape[0]):
            jump_frms = int(time_windows[time_window] * fps)
            for current_frm in prange(jump_frms, results.shape[0] + 1):
                x_movement = (
                    np.sum(
                        np.abs(
                            np.ediff1d(data[current_frm - jump_frms : current_frm, 0])
                        )
                    )
                    / pixels_per_mm
                )
                y_movement = (
                    np.sum(
                        np.abs(
                            np.ediff1d(data[current_frm - jump_frms : current_frm, 1])
                        )
                    )
                    / pixels_per_mm
                )
                results[current_frm - 1][time_window] = x_movement - y_movement

        return results

    @staticmethod
    @jit(nopython=True)
    def border_distances(
        data: np.ndarray,
        pixels_per_mm: float,
        img_resolution: np.ndarray,
        time_window: float,
        fps: int,
    ):
        """
        Compute the mean distance of key-point to the left, right, top, and bottom sides of the image in
        rolling time-windows. Uses a straight line.

        .. image:: _static/img/border_distance.png
           :width: 700
           :align: center

        .. attention::
           Output for initial frames where [current_frm - window_size] < 0 will be populated with ``-1``.

        :parameter np.ndarray data: 2d array of size len(frames)x2 with body-part coordinates.
        :parameter np.ndarray img_resolution: Resolution of video in WxH format.
        :parameter float pixels_per_mm: Pixels per millimeter of recorded video.
        :parameter int fps: FPS of the recorded video
        :parameter float time_windows: Rolling time-window as floats in seconds. E.g., ``0.2``
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0] array with millimeter distances from LEFT, RIGH, TOP, BOTTOM,

        :example:
        >>> data = np.array([[250, 250], [250, 250], [250, 250], [500, 500],[500, 500], [500, 500]]).astype(float)
        >>> img_resolution = np.array([500, 500])
        >>> FeatureExtractionSupplemental().border_distances(data=data, img_resolution=img_resolution, time_window=1, fps=2, pixels_per_mm=1)
        >>> [[-1, -1, -1, -1][250, 250, 250, 250][250, 250, 250, 250][375, 125, 375, 125][500, 0, 500, 0][500, 0, 500, 0]]
        """

        results = np.full((data.shape[0], 4), -1.0)
        window_size = int(time_window * fps)
        for current_frm in prange(window_size, results.shape[0] + 1):
            distances = np.full((4, window_size, 1), np.nan)
            windowed_locs = data[current_frm - window_size : current_frm]
            for bp_cnt, bp_loc in enumerate(windowed_locs):
                distances[0, bp_cnt] = np.linalg.norm(
                    np.array([0, bp_loc[1]]) - bp_loc
                )  # left
                distances[1, bp_cnt] = np.linalg.norm(
                    np.array([img_resolution[0], bp_loc[1]]) - bp_loc
                )  # right
                distances[2, bp_cnt] = np.linalg.norm(
                    np.array([bp_loc[0], 0]) - bp_loc
                )  # top
                distances[3, bp_cnt] = np.linalg.norm(
                    np.array([bp_loc[0], img_resolution[1]]) - bp_loc
                )  # bottpm
            for i in prange(4):
                results[current_frm - 1][i] = np.mean(distances[i]) / pixels_per_mm

        return results.astype(np.int32)

    @staticmethod
    def velocity_aggregator(
        config_path: Union[str, os.PathLike],
        data_dir: Union[str, os.PathLike],
        body_part: str,
        ts_plot: Optional[bool] = True,
    ):
        """
        Aggregate and plot velocity data from multiple pose-estimation files.

        :param Union[str, os.PathLike] config_path: Path to SimBA configuration file.
        :param Union[str, os.PathLike] data_dir: Directory containing data files.
        :param str data_dir body_part: Body part to use when calculating velocity.
        :param Optional[bool] data_dir ts_plot: Whether to generate a time series plot of velocities for each data file. Defaults to True.

        :example:
        >>> config_path = '/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini'
        >>> data_dir = '/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location'
        >>> body_part = 'Nose_1'
        >>> FeatureExtractionSupplemental.velocity_aggregator(config_path=config_path, data_dir=data_dir, body_part=body_part)
        """

        timer = SimbaTimer(start=True)
        check_if_dir_exists(in_dir=data_dir)
        config = ConfigReader(config_path=config_path, create_logger=False)
        file_paths = glob.glob(data_dir + f"/*.{config.file_type}")
        save_dir = os.path.join(
            config.logs_path, f"rolling_velocities_{config.datetime}"
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        check_if_filepath_list_is_empty(
            filepaths=file_paths,
            error_msg=f"No data in {config.file_type} format found in {data_dir}",
        )
        bp_cols = [f"{body_part}_x", f"{body_part}_y"]
        mean_velocities = {}
        for file_cnt, file_path in enumerate(file_paths):
            rolling_results = pd.DataFrame()
            _, video_name, _ = get_fn_ext(filepath=file_path)
            print(f"Analyzing {video_name}...")
            data_df = read_df(
                file_path=file_path, file_type=config.file_type, usecols=bp_cols
            ).astype(int)
            _, px_per_mm, fps = config.read_video_info(video_name=video_name)
            shifted_df = FeatureExtractionMixin.create_shifted_df(df=data_df).drop(
                bp_cols, axis=1
            )
            frm_dist = (
                FeatureExtractionMixin()
                .framewise_euclidean_distance(
                    location_1=data_df.values,
                    location_2=shifted_df.values,
                    px_per_mm=px_per_mm,
                    centimeter=True,
                )
                .astype(np.float32)
            )
            rolling = TimeseriesFeatureMixin.sliding_descriptive_statistics(
                data=frm_dist,
                window_sizes=np.array([1.0]),
                sample_rate=int(fps),
                statistics=typed.List(["sum"]),
            ).flatten()
            rolling_results[f"Rolling velocity (cm/s) - {video_name}"] = rolling
            mean_velocities[video_name] = np.mean(rolling)
            rolling_results.to_csv(os.path.join(save_dir, f"{video_name}.csv"))
            if ts_plot:
                sns.set(style="whitegrid")  # Set the style
                plt.figure(figsize=(10, 6))  # Set the figure size
                sns.lineplot(data=rolling_results, palette="tab10", linewidth=2.5)
                plt.savefig(
                    os.path.join(
                        save_dir,
                        f"rolling_velocities_{video_name}_{config.datetime}.png",
                    )
                )
                plt.close("all")
        mean_velocities = pd.DataFrame.from_dict(
            mean_velocities, orient="index", columns=["MEAN VELOCITY (CM/S)"]
        )
        mean_velocities.to_csv(
            os.path.join(config.logs_path, f"mean_velocities_{config.datetime}.csv")
        )
        timer.stop_timer()
        stdout_success(
            msg=f"Velocity aggregator for {len(file_paths)} files complete. Data saved in {config.logs_path} directory ",
            source=FeatureExtractionSupplemental.velocity_aggregator.__name__,
        )

    @staticmethod
    def spontaneous_alternations(data: pd.DataFrame, arm_names: List[str], center_name: str) -> Tuple[Dict[Union[str, Tuple[int]], int]]:
        """
        Detects spontaneous alternations between a set of user-defined ROIs.

            .. image:: _static/img/spontaneous_alternations.png
               :width: 500
               :align: center

        :param pd.DataFrame data: DataFrame containing shape data where each row represents a frame and each column represents a shape where 0 represents not in ROI and 1 represents inside the ROI
        :param List[str] shape_names: List of column names in the DataFrame corresponding to shape names.
        :returns Dict[Union[str, Tuple[str], Union[int, float, List[int]]]]: Dict with the following keys and values:

        Dict with the following keys and values:

        - 'pct_alternation': Percent alternation computed as `(spontaneous alternation cnt / (total number of arm entries - (number of arms - 1))) × 100`
        - 'alternation_cnt': The sliding count of ROI entry sequences of length `len(shape_names)` that are all unique.
        - 'same_arm_returns_cnt': Aggregate count of sequential visits to the same ROI.
        - 'alternate_arm_returns_cnt': Aggregate count of errors which are not same-arm-return errors.
        - 'error_cnt': Aggregate error count (`same_arm_returns_cnt + alternate_arm_returns_cnt`),
        - 'same_arm_returns_dict': Dictionary with the keys being the name of the ROI and values are a list of frames when the same-arm-return errors were committed.
        - 'alternate_arm_returns_cnt': Dictionary with the keys being the name of the ROI and values are a list of frames when the alternate-arm-return errors were committed.
        - 'alternations_dict': Dictionary with the keys being unique ROI name tuple sequences of length `len(shape_names)` and values are a list of frames when the sequence was completed.
        - 'arm_entry_sequence': Pandas dataframe with two columns: sequence of arm names entered, the frame the animal entered the arm, the frame that the animal left the arm.

        :example:
        >>> data = np.zeros((100, 4), dtype=int)
        >>> random_indices = np.random.randint(0, 4, size=100)
        >>> for i in range(100): data[i, random_indices[i]] = 1
        >>> df = pd.DataFrame(data, columns=['left', 'top', 'right', 'bottom'])
        >>> spontanous_alternations = FeatureExtractionSupplemental.spontaneous_alternations(data=df, shape_names=['left', 'top', 'right', 'bottom'])
        """

        def get_sliding_alternation(
            data: np.ndarray,
        ) -> Tuple[
            Union[
                Dict[int, List[int]], Dict[int, List[int]], Dict[Tuple[int], List[int]]
            ]
        ]:
            stride, same_arm_return_cnt, alternate_arm_return_cnt = (
                len(arm_names) - 1,
                0,
                0,
            )
            alternation_cnt = 0
            alternations, same_arm_returns, alternate_arm_returns = {}, {}, {}
            for i in list(itertools.permutations(arm_names)):
                alternations[i] = []
            for i in arm_names:
                same_arm_returns[i] = []
                alternate_arm_returns[i] = []
            for i in range(stride, data.shape[0]):
                current, priors = data[i], data[i - (stride) : i]
                sequence = np.append(priors[:, 0].flatten(), current[0])
                if np.unique(sequence).shape[0] == sequence.shape[0]:
                    alternations[tuple(sequence)].append(current[1])
                    alternation_cnt += 1
                elif sequence[-1] == sequence[-2]:
                    same_arm_returns[sequence[-1]].append(current[1])
                    same_arm_return_cnt += 1
                else:
                    alternate_arm_returns[sequence[-1]].append(current[1])
                    alternate_arm_return_cnt += 1
            return (
                alternate_arm_returns,
                same_arm_returns,
                alternations,
                alternate_arm_return_cnt,
                same_arm_return_cnt,
                alternation_cnt,
            )

        check_instance(source=FeatureExtractionSupplemental.spontaneous_alternations.__name__, instance=data, accepted_types=(pd.DataFrame,))
        check_valid_lst(data=arm_names,source=FeatureExtractionSupplemental.spontaneous_alternations.__name__,valid_dtypes=(str,),valid_values=data.columns)
        check_str(name="center name", value=center_name, options=data.columns)
        if center_name in arm_names: InvalidInputError(msg="One ROI is defined both as an arm ans as the center", source=FeatureExtractionSupplemental.spontaneous_alternations.__name__)
        if len(list(set(arm_names))) != len(arm_names): InvalidInputError(msg=f"Each arm has to be unique but got {arm_names}", source=FeatureExtractionSupplemental.spontaneous_alternations.__name__)
        roi_names = arm_names + [center_name]
        data_df = data[roi_names]
        invalid_vals = list(set(np.unique(data_df.values.flatten())) - {0, 1})
        if len(invalid_vals) > 0:
            raise CountError(msg=f"When computing spontaneous alternation, ROI fields can only be 0 or 1. Found the value(s): {invalid_vals}", source=FeatureExtractionSupplemental.spontaneous_alternations.__name__)
        multiple_rois_frm_idx = np.argwhere(np.sum(data_df.values, axis=1) > 1)
        if multiple_rois_frm_idx.shape[0] > 0:
            raise CountError(msg=f"When computing spontaneous alternation, animals should only exist in <=1 ROIs in any one frame. In {multiple_rois_frm_idx.shape[0]} frames, the animal exist in more than one ROI.", source=FeatureExtractionSupplemental.spontaneous_alternations.__name__)
        bout_df = (detect_bouts(data_df=data_df, target_lst=data_df.columns, fps=1)[["Event", "Start_frame", "End_frame"]].sort_values(["Start_frame"]).reset_index(drop=True))
        shifted_ = pd.concat([bout_df, bout_df.shift(-1).add_suffix("_shifted").reset_index(drop=True)], axis=1)[["Event", "Event_shifted"]].values
        unique_counts = [len(list(set(list(x)))) for x in shifted_]
        drop_idx = np.argwhere(np.array(unique_counts) == 1) + 1
        bout_df = bout_df.drop(drop_idx.flatten(), axis=0).reset_index(drop=True)
        arm_entry_sequence = bout_df[bout_df["Event"] != center_name]

        alternate_arm_returns, same_arm_returns, alternations, alternate_arm_return_cnt, same_arm_return_cnt, alternation_cnt = get_sliding_alternation(data=arm_entry_sequence[["Event", "Start_frame"]].values)

        pct_alternation = alternation_cnt / (len(arm_entry_sequence) - (len(arm_names) - 1))

        return {
            "pct_alternation": pct_alternation * 100,
            "alternation_cnt": alternation_cnt,
            "error_cnt": same_arm_return_cnt + alternate_arm_return_cnt,
            "same_arm_returns_cnt": same_arm_return_cnt,
            "alternate_arm_returns_cnt": alternate_arm_return_cnt,
            "same_arm_returns_dict": same_arm_returns,
            "alternate_arm_returns_dict": alternate_arm_returns,
            "alternations_dict": alternations,
            "arm_entry_sequence": arm_entry_sequence,
        }

    @staticmethod
    def find_path_loops(data: np.ndarray) -> Dict[Tuple[int], List[int]]:
        """
        Compute the loops detected within a 2-dimensional path.

        .. image:: _static/img/find_path_loops.png
           :width: 700
           :align: center

        :param np.ndarray data: Nx2 2-dimensional array with the x and y coordinated represented on axis 1.
        :return: Dictionary with the coordinate tuple(x, y) as keys, and sequential frame numbers as values when animals visited, and re-visited the key coordinate.

        :example:
        >>> data = read_df(file_path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/csv/outlier_corrected_movement_location/SI_DAY3_308_CD1_PRESENT.csv', usecols=['Center_x', 'Center_y'], file_type='csv').values.astype(int)
        >>> FeatureExtractionSupplemental.find_path_loops(data=data)
        """

        check_valid_array(
            data=data,
            source=FeatureExtractionSupplemental.find_path_loops.__name__,
            accepted_ndims=(2,),
            accepted_dtypes=(np.int32, np.int64, np.int8),
        )
        seen = {}
        for i in range(data.shape[0]):
            value = tuple(data[i])
            if value not in seen.keys():
                seen[value] = [i]
            else:
                seen[value].append(i)
        seen_dedup = {}
        for k, v in seen.items():
            seen_dedup[k] = [
                x for cnt, x in enumerate(v) if cnt == 0 or v[cnt] > v[cnt - 1] + 1
            ]
        return {k: v for k, v in seen_dedup.items() if len(v) > 1}

    @staticmethod
    def sequential_lag_analysis(
        data: pd.DataFrame, criterion: str, target: str, time_window: float, fps: float
    ):
        """
        Perform sequential lag analysis to determine the temporal relationship between two events.

        For every onset of behavior C, count the proportions of behavior T onsets in the time-window preceding the onset
        of behavior C vs the proportion of behavior T onsets in the time-window proceeding the onset of behavior C.

        A value closer to 1.0 indicates that behavior T always precede behavior C. A value closer to 0.0 indicates that behavior T follows behavior C. A value of -1.0 indicates
        that behavior T never precede nor proceed behavior C.


        .. seealso::
           :class:`simba.data_processorsfsttc_calculator.FSTTCCalculator`

        :example:
        >>> df = pd.DataFrame(np.random.randint(0, 2, (100, 2)), columns=['Attack', 'Sniffing'])
        >>> FeatureExtractionSupplemental.sequential_lag_analysis(data=df, criterion='Attack', target='Sniffing', fps=5, time_window=2.0)

        References
        ----------
        .. [1] Casarrubea et al., Structural analyses in the study of behavior: From rodents to non-human primates, `Frontiers in Psychology`,
               2022.
        """

        check_str(
            name=f"{FeatureExtractionSupplemental.sequential_lag_analysis.__name__} criterion",
            value=criterion,
        )
        check_str(
            name=f"{FeatureExtractionSupplemental.sequential_lag_analysis.__name__} target",
            value=target,
        )
        check_float(
            name=f"{FeatureExtractionSupplemental.sequential_lag_analysis.__name__} fps",
            value=fps,
            min_value=1.0,
        )
        check_float(
            name=f"{FeatureExtractionSupplemental.sequential_lag_analysis.__name__} time-window",
            value=time_window,
            min_value=10e-6,
        )
        check_valid_dataframe(
            df=data,
            source=f"{FeatureExtractionSupplemental.sequential_lag_analysis.__name__} data",
            valid_dtypes=(np.float32, np.float64, np.int64, np.int32, float, int),
            required_fields=[criterion, target],
        )

        bouts = detect_bouts(data_df=data, target_lst=[criterion, target], fps=fps)
        if len(bouts) == 0:
            raise CountError(
                msg=f"No events of behaviors {criterion} and {target} detected in data.",
                source=FeatureExtractionSupplemental.sequential_lag_analysis,
            )
        criterion_starts = bouts["Start_frame"][bouts["Event"] == criterion].values
        target_starts = bouts["Start_frame"][bouts["Event"] == target].values
        preceding_cnt, proceeding_cnt = 0, 0
        window = int(fps * time_window)
        if window < 1.0:
            window = 1
        for criterion_start in criterion_starts:
            preceeding_events = target_starts[
                np.argwhere(
                    (target_starts < criterion_start)
                    & (target_starts >= (criterion_start - window))
                )
            ].flatten()
            preceding_cnt += preceeding_events.shape[0]
            target_starts = np.array(
                [x for x in target_starts if x not in preceeding_events]
            )
            proceeding_events = target_starts[
                np.argwhere(
                    (target_starts > criterion_start)
                    & (target_starts <= (criterion_start + window))
                )
            ].flatten()
            proceeding_cnt += proceeding_events.shape[0]
            target_starts = np.array(
                [x for x in target_starts if x not in proceeding_events]
            )
        if preceding_cnt == 0 and proceeding_cnt == 0:
            return -1.0
        elif preceding_cnt == 0:
            return 0.0
        elif proceeding_cnt == 0:
            return 1.0
        else:
            return np.round(preceding_cnt / (preceding_cnt + proceeding_cnt), 3)

    @staticmethod
    def distance_and_velocity(
        x: np.ndarray,
        fps: float,
        pixels_per_mm: float,
        centimeters: Optional[bool] = True,
    ) -> Tuple[float, float]:
        """
        Calculate total movement and mean velocity from a sequence of position data.

        :param x: Array containing movement data. For example, created by ``simba.mixins.FeatureExtractionMixin.framewise_euclidean_distance``.
        :param fps: Frames per second of the data.
        :param pixels_per_mm: Conversion factor from pixels to millimeters.
        :param Optional[bool] centimeters: If True, results are returned in centimeters and centimeters per second. Defaults to True.
        :return Tuple[float, float]: A tuple containing total movement and mean velocity.

        :example:
        >>> x = np.random.randint(0, 100, (100,))
        >>> sum_movement, avg_velocity = FeatureExtractionSupplemental.distance_and_velocity(x=x, fps=10, pixels_per_mm=10, centimeters=True)
        """

        check_valid_array(
            data=x,
            source=FeatureExtractionSupplemental.distance_and_velocity.__name__,
            accepted_ndims=(1, 2),
            accepted_dtypes=(np.float32, np.float64, np.int32, np.int64, int, float),
        )
        check_float(
            name=f"{FeatureExtractionSupplemental.distance_and_velocity.__name__} fps",
            value=fps,
            min_value=1,
        )
        check_float(
            name=f"{FeatureExtractionSupplemental.distance_and_velocity.__name__} pixels_per_mm",
            value=pixels_per_mm,
            min_value=10e-6,
        )
        if x.ndim == 2:
            check_valid_array(
                data=x,
                source=FeatureExtractionSupplemental.distance_and_velocity.__name__,
                accepted_axis_1_shape=(2,),
            )
            t = np.full((x.shape[0]), 0.0)
            for i in range(1, x.shape[0]):
                t[i] = np.linalg.norm(x[i] - x[i - 1])
            x = np.copy(t) / pixels_per_mm
        movement = np.sum(x) / pixels_per_mm
        v = []
        for i in range(0, x.shape[0], int(fps)):
            w = x[i : (i + int(fps))]
            v.append((np.sum(w) / pixels_per_mm) * (1 / (w.shape[0] / int(fps))))
        if centimeters:
            v = [vi / 10 for vi in v]
            movement = movement / 10
        return movement, np.mean(v)


# x = np.random.randint(0, 100, (100, 2))
# FeatureExtractionSupplemental.distance_and_velocity(
#     x=x, fps=10, pixels_per_mm=10, centimeters=True
# )
# # sum_movement, avg_velocity =


# df = read_df(file_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/targets_inserted/Together_1.csv', file_type='csv')
#
# df = pd.DataFrame(np.random.randint(0, 2, (100, 2)), columns=['Attack', 'Sniffing'])
# FeatureExtractionSupplemental.sequential_lag_analysis(data=df, criterion='Attack', target='Sniffing', fps=5, time_window=2.0)

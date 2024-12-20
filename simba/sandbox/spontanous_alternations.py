
import os
from typing import List, Tuple, Dict, Union, Optional
import pandas as pd
import numpy as np
from copy import deepcopy
import itertools
import warnings
warnings.filterwarnings("ignore")
from shapely.geometry import Polygon, Point

from simba.utils.checks import check_instance, check_valid_lst, check_that_column_exist, check_file_exist_and_readable, check_if_dir_exists, check_that_column_exist
from simba.utils.errors import CountError, NoFilesFoundError, NoROIDataError
from simba.mixins.config_reader import ConfigReader
from simba.utils.read_write import get_file_name_info_in_directory, get_fn_ext, read_df
from simba.utils.warnings import NoFileFoundWarning
from simba.utils.enums import Keys
from simba.roi_tools.ROI_analyzer import ROIAnalyzer
from simba.mixins.geometry_mixin import GeometryMixin

# def __spontaneous_alternations(data: pd.DataFrame,
#                               roi_names: List[str]) -> Tuple[int, Dict[str, np.ndarray]]:
#     """
#     Detects spontaneous alternations between a set of user-defined ROIs.
#
#     :param pd.DataFrame data: DataFrame containing shape data where each row represents a frame and each column represents a shape where 0 represents not in ROI and 1 represents inside the ROI
#     :param pd.DataFrame data: List of column names in the DataFrame corresponding to shape names.
#     :returns Dict[Union[str, Tuple[str], Union[int, float, List[int]]]]: Dict with the following keys and values:
#             'pct_alternation': Percent alternation computed as spontaneous alternation cnt /(total number of arm entries − (number of arms - 1))} × 100
#             'alternation_cnt': The sliding count of ROI entry sequences of len(shape_names) that are all unique.
#             'same_arm_returns_cnt': Aggregate count of sequantial visits to the same ROI.
#             'alternate_arm_returns_cnt': Aggregate count of errors which are not same-arm-return errors.
#             'error_cnt': Aggregate error count (same_arm_returns_cnt + alternate_arm_returns_cnt),
#             'same_arm_returns_dict': Dictionary with the keys being the name of the ROI and values are a list of frames when the same-arm-return errors where committed.
#             'alternate_arm_returns_cnt': Dictionary with the keys being the name of the ROI and values are a list of frames when the alternate-arm-return errors where committed.
#             'alternations_dict': Dictionary with the keys being unique ROI name tuple sequences of length len(shape_names) and values are a list of frames when the sequence was completed.
#
#     :example:
#     >>> data = np.zeros((100, 4), dtype=int)
#     >>> random_indices = np.random.randint(0, 4, size=100)
#     >>> for i in range(100): data[i, random_indices[i]] = 1
#     >>> df = pd.DataFrame(data, columns=['left', 'top', 'right', 'bottom'])
#     >>> spontanous_alternations = spontaneous_alternations(data=df, shape_names=['left', 'top', 'right', 'bottom'])
#     """
#
#     def get_sliding_alternation(data: np.ndarray) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[Tuple[int], List[int]]]:
#         alt_cnt, stride = 0, data.shape[1]-1
#         arm_visits = np.full((data.shape[0]), -1)
#         same_arm_returns, alternations, alternate_arm_returns = {}, {}, {}
#         for i in range(data.shape[1]-1):  alternate_arm_returns[i], same_arm_returns[i] = [], []
#         for i in list(itertools.permutations(list(range(0, data.shape[1]-1)))): alternations[i] = []
#         for i in range(data.shape[0]): arm_visits[i] = np.argwhere(data[i, 1:] == 1).flatten()[0]
#         for i in range(stride-1, arm_visits.shape[0]):
#             current, priors = arm_visits[i], arm_visits[i-(stride-1):i]
#             sequence = np.append(priors, [current])
#             if np.unique(sequence).shape[0] == stride:
#                 alternations[tuple(sequence)].append(data[i, 0])
#             else:
#                 if current == priors[-1]: same_arm_returns[current].append(data[i, 0])
#                 else: alternate_arm_returns[current].append(data[i, 0])
#         return same_arm_returns, alternate_arm_returns, alternations
#
#     check_instance(source=spontaneous_alternations.__name__, instance=data, accepted_types=(pd.DataFrame,))
#     check_valid_lst(data=roi_names, source=spontaneous_alternations.__name__, valid_dtypes=(str,))
#     for shape_name in roi_names: check_that_column_exist(df=data, column_name=shape_name, file_name='')
#     data = data[roi_names]
#     additional_vals = list(set(np.unique(data.values.flatten())) - {0, 1})
#     if len(additional_vals) > 0:
#         raise CountError(msg=f'When computing spontaneous alternation, ROI fields can only be 0 or 1. Found {additional_vals}', source=spontaneous_alternations.__name__)
#     above_1_idx = np.argwhere(np.sum(data.values, axis=1) > 1)
#     if above_1_idx.shape[0] > 0:
#         raise CountError(msg=f'When computing spontaneous alternation, animals should only exist in <=1 ROIs in any one frame. In {above_1_idx.shape[0]} frames, the animal exist in more than one ROI.', source=spontaneous_alternations.__name__)
#     shape_map = {}
#     for i in range(len(roi_names)): shape_map[i] = roi_names[i]
#     data = np.hstack((np.arange(0, data.shape[0]).reshape(-1, 1), data.values))
#     data = data[np.sum(data[:, 1:], axis=1) != 0]
#     data = data[np.concatenate(([0], np.where(~(data[:, 1:][1:] == data[:, 1:][:-1]).all(axis=1))[0] + 1))]
#     # same_arm, alternate_arm, alt = get_sliding_alternation(data=data)
#     # same_arm_returns, alternate_arm_returns = {}, {}
#     # for k, v in same_arm.items(): same_arm_returns[shape_map[k]] = v
#     # for k, v in alternate_arm.items(): alternate_arm_returns[shape_map[k]] = v
#     # alternations = {}
#     # for k, v in alt.items():
#     #     new_k = []
#     #     for i in k: new_k.append(shape_map[i])
#     #     alternations[tuple(new_k)] = v
#     #
#     # same_arm_returns_cnt, alternation_cnt, alternate_arm_returns_cnt = 0, 0, 0
#     # for v in same_arm_returns.values():
#     #     same_arm_returns_cnt += len(v)
#     # for v in alternate_arm_returns.values():
#     #     alternate_arm_returns_cnt += len(v)
#     # for v in alternations.values(): alternation_cnt += len(v)
#     # pct_alternation = alternation_cnt / (data.shape[0] - (data.shape[1] -1))
#     #
#     # return {'pct_alternation': pct_alternation * 100,
#     #        'alternation_cnt': alternation_cnt,
#     #        'error_cnt': same_arm_returns_cnt + alternate_arm_returns_cnt,
#     #        'same_arm_returns_cnt': same_arm_returns_cnt,
#     #        'alternate_arm_returns_cnt': alternate_arm_returns_cnt,
#     #        'same_arm_returns_dict': same_arm_returns,
#     #        'alternate_arm_returns_dict': alternate_arm_returns,
#     #        'alternations_dict': alternations}


def __spontaneous_alternations(data: Dict[str, List[int]]) -> Tuple[int, Dict[str, np.ndarray]]:

    d = {}
    roi_names = data.keys()
    for shape_name, shape_data in data.items():
        for entry_frm in shape_data: d[entry_frm] = shape_name
    d = {k: d[k] for k in sorted(d)}
    print(d)
    pass


# def filter_low_p_bps_for_shapes(x: np.ndarray, p: np.ndarray, threshold: float):
#     """
#     Filter body-part data for geometry construction while maintaining valid geometry arrays.
#
#     Having a 3D array representing body-parts across time, and a second 3D array representing probabilities of those
#     body-parts across time, we want to "remove" body-parts with low detection probabilities whilst also keeping the array sizes
#     intact and suitable for geometry construction. To do this, we find body-parts with detection probabilities below the threshold, and replace these with a body-part
#     that doesn't fall below the detection probability threshold within the same frame. However, to construct a geometry, we need >= 3 unique key-point locations.
#     Thus, no substitution can be made to when there are less than three unique body-part locations within a frame that falls above the threshold.
#
#     :example:
#     >>> x = np.random.randint(0, 500, (18000, 7, 2))
#     >>> p = np.random.random(size=(18000, 7, 1))
#     >>> x = filter_low_p_bps_for_shapes(x=x, p=p, threshold=0.1)
#     >>> x = x.reshape(x.shape[0], int(x.shape[1] * 2))
#     """
#
#     results = np.copy(x)
#     for i in range(x.shape[0]):
#         below_p_idx = np.argwhere(p[i].flatten() < threshold).flatten()
#         above_p_idx = np.argwhere(p[i].flatten() >= threshold).flatten()
#         if (below_p_idx.shape[0] > 0) and (above_p_idx.shape[0] >= 3):
#             for j in below_p_idx:
#                 new_val = x[i][above_p_idx[0]]
#                 results[i][j] = new_val
#     return results



def spontaneous_alternations(config_path: Union[str, os.PathLike],
                             roi_names: List[str],
                             animal_area: Optional[int] = 80,
                             threshold: Optional[float] = 0.1,
                             data_dir: Optional[Union[str, os.PathLike]] = None):

    check_file_exist_and_readable(file_path=config_path)
    config = ConfigReader(config_path=config_path)
    if data_dir is None:
        data_dir = config.outlier_corrected_dir
    check_if_dir_exists(in_dir=data_dir)
    file_paths = get_file_name_info_in_directory(directory=data_dir, file_type=config.file_type)
    config.read_roi_data()
    files_w_missing_rois = list(set(file_paths.keys()) - set(config.video_names_w_rois))
    files_w_rois = [x for x in list(file_paths.keys()) if x in config.video_names_w_rois]
    if len(files_w_rois) == 0: raise NoFilesFoundError(msg=f'No ROI definitions found for any of the data files in {data_dir}', source=spontaneous_alternations.__name__)
    if len(files_w_missing_rois) > 0: NoFileFoundWarning(msg=f'{len(files_w_missing_rois)} file(s) in {data_dir} are missing ROI definitions and will be skipped when performing spontaneous alternation calculations: {files_w_missing_rois}', source=spontaneous_alternations.__name__)
    for video_name in files_w_rois:
        video_rectangles = config.roi_dict[Keys.ROI_RECTANGLES.value][config.roi_dict[Keys.ROI_RECTANGLES.value]['Video'] == video_name]
        video_circles = config.roi_dict[Keys.ROI_CIRCLES.value][config.roi_dict[Keys.ROI_CIRCLES.value]['Video'] == video_name]
        video_polygons = config.roi_dict[Keys.ROI_POLYGONS.value][config.roi_dict[Keys.ROI_POLYGONS.value]['Video'] == video_name]
        video_shape_names = list(video_circles['Name']) + list(video_rectangles['Name']) + list(video_polygons['Name'])
        missing_rois = list(set(roi_names) - set(video_shape_names))
        if len(missing_rois) > 0:
            raise NoROIDataError(msg=f'{len(missing_rois)} ROI(s) are missing from {video_name}: {missing_rois}', source=spontaneous_alternations.__name__)
    roi_geometries = GeometryMixin.simba_roi_to_geometries(rectangles_df=config.rectangles_df, circles_df=config.circles_df, polygons_df=config.polygon_df)
    roi_geometries = {k: v for k, v in roi_geometries.items() if k in files_w_rois}
    print(roi_geometries)
    # file_paths = list({file_paths[k] for k in files_w_rois if k in file_paths})
    # for file_path in file_paths:
    #     data_df = read_df(file_path=file_path, file_type=config.file_type)
    #     x = data_df[[x for x in config.bp_headers if not x.endswith('_p') and not 'tail_end' in x.lower()]]
    #     p = data_df[[x for x in config.bp_headers if x.endswith('_p') and not 'tail_end' in x.lower()]]
    #     x = x.values.reshape(len(x), int(len(x.columns)/2) , 2).astype(np.int64)
    #     p = p.values.reshape(len(p), len(p.columns) , 1)
    #     x = GeometryMixin.filter_low_p_bps_for_shapes(x=x, p=p, threshold=threshold)
    #     x = x.reshape(x.shape[0], -1, 2)
    #     print(x.shape)
    #     polygons = GeometryMixin.bodyparts_to_polygon(data=x)






    #
    #
    # settings = {'body_parts': {'Animal_1': body_part}, 'threshold': threshold}





    # for file_path in file_paths:
    #     roi_analyzer = ROIAnalyzer(ini_path=config_path, file_path=file_path, settings=settings, detailed_bout_data=True)
    #     roi_analyzer.run()
    #     data_dict = roi_analyzer.entries_exit_dict[get_fn_ext(filepath=file_path)[1]]['Animal_1']
    #     data_dict_cleaned = {}
    #     for k, v in data_dict.items(): data_dict_cleaned[k] = v['Entry_times']
    #     __spontaneous_alternations(data=data_dict_cleaned)
        # for shape in roi_names:
        #     data_df[shape] = 0
        #     roi_shape_df = roi_analyzer.detailed_df[roi_analyzer.detailed_df['SHAPE'] == shape]
        #     inside_roi_idx = list(roi_shape_df.apply(lambda x: list(range(int(x["ENTRY FRAMES"]), int(x["EXIT FRAMES"]) + 1)), 1,))
        #     inside_roi_idx = [x for xs in inside_roi_idx for x in xs]
        #     data_df.loc[inside_roi_idx, shape] = 1
        # alternation = __spontaneous_alternations(data=data_df, roi_names=roi_names)
        #print(data_df)


spontaneous_alternations(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/spontenous_alternation/project_folder/project_config.ini',
                         roi_names=['A', 'B', 'C'], body_part='Center')



# data = np.zeros((50, 4), dtype=int)
# random_indices = np.random.randint(0, 4, size=50)
# for i in range(50): data[i, random_indices[i]] = 1
# df = pd.DataFrame(data, columns=['left', 'top', 'right', 'bottom'])
# results = spontanous_alternation = spontaneous_alternations(data=df, shape_names=['left', 'top', 'right', 'bottom'])

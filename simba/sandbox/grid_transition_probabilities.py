import functools
import multiprocessing
from typing import Dict, Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon

from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.checks import check_int, check_valid_array, check_valid_dict
from simba.utils.enums import Defaults, Formats
from simba.utils.printing import SimbaTimer
from simba.utils.read_write import find_core_cnt, get_video_meta_data, read_df


def _compute_framewise_geometry_idx(data: np.ndarray,
                                    grid: Dict[Tuple[int, int], Polygon],
                                    verbose: bool):

    frm_idxs, cords = data[:, 0], data[:, 1:]
    results = np.full(shape=(data.shape[0], 3), dtype=np.int32, fill_value=-1)
    for frm_idx in range(frm_idxs.shape[0]):
        frm_id, frm_point = frm_idxs[frm_idx], Point(cords[frm_idx])
        if verbose:
            print(f'Processing frame {frm_id}...')
        for grid_idx, grid_geometry in grid.items():
            if grid_geometry.contains(frm_point) or grid_geometry.touches(frm_point):
                results[frm_idx] = np.array([frm_id, grid_idx[0], grid_idx[1]])

    return results




def geometry_transition_probabilities(data: np.ndarray,
                                      grid: Dict[Tuple[int, int], Polygon],
                                      core_cnt: Optional[int] = -1,
                                      verbose: Optional[bool] = False) -> (Dict[Tuple[int, int], float], Dict[Tuple[int, int], int]):
    """
    Calculate geometry transition probabilities based on spatial transitions between grid cells.

    Computes transition probabilities between pairs of spatial grid cells, represented as polygons. For each cell, it calculates the likelihood of transitioning to other cells.

    :param np.ndarray data: A 2D array where each row represents a point in space with two coordinates [x, y].
    :param Dict[Tuple[int, int], Polygon] grid: A dictionary mapping grid cell identifiers (tuple of int, int) to their corresponding polygon objects.
                                                Each grid cell is represented by a tuple key (e.g., (row, col)) and its spatial boundaries as a `Polygon`. Can be computed with E.g., created by :func:`simba.mixins.geometry_mixin.GeometryMixin.bucket_img_into_grid_square` or :func:`simba.mixins.geometry_mixin.GeometryMixin.bucket_img_into_grid_hexagon`.
    :param Optional[int] core_cnt: The number of cores to use for parallel processing. Default is -1, which uses the maximum available cores.
    :param Optional[bool] verbose: If True, the function will print additional information, including the elapsed time for processing.
    :return: A tuple containing two dictionaries:
             - A dictionary of transition probabilities between grid cells, where each key is a grid cell tuple (row, col),
               and each value is another dictionary representing the transition probabilities to other cells.
             - A dictionary of transition counts between grid cells, where each key is a grid cell tuple (row, col),
               and each value is another dictionary representing the transition counts to other cells.
    :rtype: Tuple[Dict[Tuple[int, int], Dict[Tuple[int, int], float]], Dict[Tuple[int, int], Dict[Tuple[int, int], int]]]

    :example:
    >>> video_meta_data = get_video_meta_data(video_path=r"C:\troubleshooting\mitra\project_folder\videos\708_MA149_Gq_CNO_0515.mp4")
    >>> w, h = video_meta_data['width'], video_meta_data['height']
    >>> grid = GeometryMixin().bucket_img_into_grid_square(bucket_grid_size=(5, 5), bucket_grid_size_mm=None, img_size=(h, w), verbose=False)[0]
    >>> data = read_df(file_path=r'C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\708_MA149_Gq_CNO_0515.csv', file_type='csv')[['Nose_x', 'Nose_y']].values
    >>> transition_probabilities, _ = geometry_transition_probabilities(data=data, grid=grid)
    """

    timer = SimbaTimer(start=True)
    check_valid_array(data=data, source=geometry_transition_probabilities.__name__, accepted_ndims=(2,), accepted_axis_1_shape=[2,], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_dict(x=grid, valid_key_dtypes=(tuple,), valid_values_dtypes=(Polygon,))
    check_int(name="core_cnt", value=core_cnt, min_value=-1, unaccepted_vals=[0])
    if core_cnt == -1 or core_cnt > find_core_cnt()[0]: core_cnt = find_core_cnt()[0]
    frm_id = np.arange(0, data.shape[0]).reshape(-1, 1)
    data = np.hstack((frm_id, data)).reshape(-1, 3).astype(np.int32)
    data, results = np.array_split(data, core_cnt), []
    with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
        constants = functools.partial(_compute_framewise_geometry_idx, grid=grid, verbose=verbose)
        for cnt, result in enumerate(pool.imap(constants, data, chunksize=1)):
            results.append(result)
    pool.join(); pool.terminate(); del data

    results = np.vstack(results)[:, 1:].astype(np.int32)
    out_transition_probabilities, out_transition_cnts = {}, {}
    unique_grids = np.unique(results, axis=0)
    for unique_grid in unique_grids:
        in_grid_idx = np.where(np.all(results == unique_grid, axis=1))[0]
        in_grid_idx = np.split(in_grid_idx, np.where(np.diff(in_grid_idx) > 1)[0] + 1)
        transition_idx = [np.max(x)+1 for x in in_grid_idx if np.max(x)+1 < results.shape[0]]
        transition_geometries = results[transition_idx, :]
        unique_rows, counts = np.unique(transition_geometries, axis=0, return_counts=True)
        grid_dict = {tuple(row): count for row, count in zip(unique_rows, counts)}
        non_transition_grids = [tuple(x) for x in unique_grids if tuple(x) not in grid_dict.keys()]
        non_transition_grids = {k: 0 for k in non_transition_grids}
        grid_dict.update(non_transition_grids)
        transition_cnt = sum(grid_dict.values())
        out_transition_probabilities[tuple(unique_grid)] = {k: v/transition_cnt for k,v in grid_dict.items()}
        out_transition_cnts[tuple(unique_grid)] = grid_dict
    timer.stop_timer()
    if verbose:
        print(f'Geometry transition probabilities complete. Elapsed time: {timer.elapsed_time_str}')
    return (out_transition_probabilities, out_transition_cnts)


# if __name__=="__main__":
#     video_meta_data = get_video_meta_data(video_path=r"C:\troubleshooting\mitra\project_folder\videos\708_MA149_Gq_CNO_0515.mp4")
#     w, h = video_meta_data['width'], video_meta_data['height']
#     grid = GeometryMixin().bucket_img_into_grid_square(bucket_grid_size=(5, 5), bucket_grid_size_mm=None, img_size=(h, w), verbose=False)[0]
#     data = read_df(file_path=r'C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\708_MA149_Gq_CNO_0515.csv', file_type='csv')[['Nose_x', 'Nose_y']].values
#     transition_probabilities, _ = geometry_transition_probabilities(data=data, grid=grid)
#     print(transition_probabilities)


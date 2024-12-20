import time
from typing import Tuple

import numpy as np
from numba import jit, njit, prange

from simba.utils.read_write import read_df


@njit("(int32[:, :, :], int64, int64, int64, int32[:])")
def egocentrically_align_pose_numba(data: np.ndarray,
                                    anchor_1_idx: int,
                                    anchor_2_idx: int,
                                    direction: int,
                                    anchor_location: np.ndarray,
                                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Aligns a set of 2D points egocentrically based on two anchor points and a target direction.

    Rotates and translates a 3D array of 2D points (e.g., time-series of frame-wise data) such that
    one anchor point is aligned to a specified location, and the direction between the two anchors is aligned
    to a target angle.


    .. video:: _static/img/EgocentricalAligner.webm
       :width: 600
       :autoplay:
       :loop:

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/egocentrically_align_pose_numba.csv
       :widths: 12, 22, 22, 22, 22
       :align: center
       :class: simba-table
       :header-rows: 1

    :param np.ndarray data: A 3D array of shape `(num_frames, num_points, 2)` containing 2D points for each frame. Each frame is represented as a 2D array of shape `(num_points, 2)`, where each row corresponds to a point's (x, y) coordinates.
    :param int anchor_1_idx: The index of the first anchor point in `data` used as the center of alignment. This body-part will be placed in the center of the image.
    :param int anchor_2_idx: The index of the second anchor point in `data` used to calculate the direction vector. This bosy-part will be located `direction` degrees from the anchor_1 body-part.
    :param int direction: The target direction in degrees to which the vector between the two anchors will be aligned.
    :param np.ndarray anchor_location: A 1D array of shape `(2,)` specifying the target (x, y) location for `anchor_1_idx` after alignment.
    :return: A tuple containing the rotated data, and variables required for also rotating the video using the same rules:
             - `aligned_data`: A 3D array of shape `(num_frames, num_points, 2)` with the aligned 2D points.
             - `centers`: A 2D array of shape `(num_frames, 2)` containing the original locations of `anchor_1_idx` in each frame before alignment.
             - `rotation_vectors`: A 3D array of shape `(num_frames, 2, 2)` containing the rotation matrices applied to each frame.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]

    :example:
    >>> data = np.random.randint(0, 500, (100, 7, 2))
    >>> anchor_1_idx = 5 # E.g., the animal tail-base is the 5th body-part
    >>> anchor_2_idx = 7 # E.g., the animal nose is the 7th row in the data
    >>> anchor_location = np.array([250, 250]) # the tail-base (index 5) is placed at x=250, y=250 in the image.
    >>> direction = 90 # The nose (index 7) will be placed in direction 90 degrees (S) relative to the tailbase.
    >>> results, centers, rotation_vectors = egocentrically_align_pose_numba(data=data, anchor_1_idx=anchor_1_idx, anchor_2_idx=anchor_2_idx, direction=direction)
    """

    target_angle = np.deg2rad(direction)
    centers = np.full((data.shape[0], 2), fill_value=-1, dtype=np.int32)
    rotation_vectors = np.full((data.shape[0], 2, 2), fill_value=-1, dtype=np.float32)
    results = np.zeros_like(data, dtype=np.int32)
    for frm_idx in prange(data.shape[0]):
        frm_points = data[frm_idx]
        frm_anchor_1, frm_anchor_2 = frm_points[anchor_1_idx], frm_points[anchor_2_idx]
        centers[frm_idx] = frm_anchor_1
        delta_x, delta_y = frm_anchor_2[0] - frm_anchor_1[0], frm_anchor_2[1] - frm_anchor_1[1]
        frm_angle = np.arctan2(delta_y, delta_x)
        frm_rotation_angle = target_angle - frm_angle
        frm_cos_theta, frm_sin_theta = np.cos(frm_rotation_angle), np.sin(frm_rotation_angle)
        R = np.array([[frm_cos_theta, -frm_sin_theta], [frm_sin_theta, frm_cos_theta]])
        rotation_vectors[frm_idx] = R
        keypoints_rotated = np.dot(frm_points.astype(np.float64) - frm_anchor_1.astype(np.float64), R.T)
        anchor_1_position_after_rotation = keypoints_rotated[anchor_1_idx]
        translation_to_target = anchor_location - anchor_1_position_after_rotation
        results[frm_idx] = keypoints_rotated + translation_to_target

    return results, centers, rotation_vectors



# data_path = r"C:\projects\simba\simba\tests\data\test_projects\mouse_open_field\project_folder\csv\outlier_corrected_movement_location\Video1.csv"
# data_df = read_df(file_path=data_path, file_type='csv')
# bp_cols = [x for x in data_df.columns if not x.endswith('_p')]
# data_arr = data_df[bp_cols].values.astype(np.int32).reshape(len(data_df), int(len(bp_cols)/2), 2).astype(np.int32)
#
# for i in [1000000, 2000000, 4000000, 8000000, 16000000, 32000000, 64000000]:
#     times = []
#     data_arr = np.random.randint(0, 500, (i, 6, 2))
#     for j in range(3):
#         start = time.time()
#         results, centers, rotation_vectors = egocentrically_align_pose_numba(data=data_arr, anchor_1_idx=2, anchor_2_idx=3, anchor_location=np.array([250, 250]).astype(np.int32), direction=90)
#         run_time = time.time() - start
#         times.append(run_time)
#     print(i, '\t' * 3, np.mean(times), '\t' *3, np.std(times))

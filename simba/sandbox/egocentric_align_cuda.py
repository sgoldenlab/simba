import math
import time

from simba.utils.checks import check_int, check_valid_array
import numpy as np
from typing import Tuple
from simba.utils.errors import InvalidInputError
from simba.utils.enums import Formats
from simba.utils.read_write import read_df
from numba import cuda
from simba.data_processors.cuda.utils import _cuda_matrix_multiplication, _cuda_2d_transpose, _cuda_subtract_2d, _cuda_add_2d

THREADS_PER_BLOCK = 1024


@cuda.jit()
def _egocentric_align_kernel(data, centers, rotation_vectors, results, target_angle, anchor_idx, transposed_rotation_vectors, matrix_multiplier_arr, anchor_loc):
    frm_idx = cuda.grid(1)
    if frm_idx >= data.shape[0]:
        return
    else:
        frm_points = data[frm_idx]
        frm_anchor_1, frm_anchor_2 = frm_points[anchor_idx[0]], frm_points[anchor_idx[1]]
        centers[frm_idx][0], centers[frm_idx][1] = frm_anchor_1[0], frm_anchor_1[1]
        delta_x, delta_y = frm_anchor_2[0] - frm_anchor_1[0], frm_anchor_2[1] - frm_anchor_1[1]
        frm_angle = math.atan2(delta_y, delta_x)
        frm_rotation_angle = target_angle[0] - frm_angle
        frm_cos_theta, frm_sin_theta = math.cos(frm_rotation_angle), math.sin(frm_rotation_angle)
        rotation_vectors[frm_idx][0][0], rotation_vectors[frm_idx][0][1] = frm_cos_theta, -frm_sin_theta
        rotation_vectors[frm_idx][1][0], rotation_vectors[frm_idx][1][1] = frm_sin_theta, frm_cos_theta
        keypoints_rotated = _cuda_subtract_2d(frm_points, frm_anchor_1)
        r_transposed = _cuda_2d_transpose(rotation_vectors[frm_idx], transposed_rotation_vectors[frm_idx])
        keypoints_rotated = _cuda_matrix_multiplication(keypoints_rotated, r_transposed, matrix_multiplier_arr[frm_idx])
        anchor_1_position_after_rotation = keypoints_rotated[anchor_idx[0]]
        anchor_1_position_after_rotation[0] = anchor_loc[0] - anchor_1_position_after_rotation[0]
        anchor_1_position_after_rotation[1] = anchor_loc[1] - anchor_1_position_after_rotation[1]

        frm_results = _cuda_add_2d(keypoints_rotated, anchor_1_position_after_rotation)
        for i in range(frm_results.shape[0]):
            for j in range(frm_results.shape[1]):
                results[frm_idx][i][j] = frm_results[i][j]



def egocentrically_align_pose_cuda(data: np.ndarray,
                                   anchor_1_idx: int,
                                   anchor_2_idx: int,
                                   anchor_location: np.ndarray,
                                   direction: int,
                                   batch_size: int = int(10e+5)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """

    :example:
    >>> DATA_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/data/501_MA142_Gi_Saline_0513.csv"
    >>> VIDEO_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/videos/501_MA142_Gi_Saline_0513.mp4"
    >>> SAVE_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/videos/501_MA142_Gi_Saline_0513_rotated.mp4"
    >>> ANCHOR_LOC = np.array([300, 300])
    >>>
    >>> df = read_df(file_path=DATA_PATH, file_type='csv')
    >>> bp_cols = [x for x in df.columns if not x.endswith('_p')]
    >>> data = df[bp_cols].values.reshape(len(df), int(len(bp_cols)/2), 2).astype(np.int64)
    >>> data, centers, rotation_matrices = egocentrically_align_pose_cuda(data=data, anchor_1_idx=6, anchor_2_idx=2, anchor_location=ANCHOR_LOC, direction=180,batch_size=36000000)
    """

    check_valid_array(data=data, source=egocentrically_align_pose_cuda.__name__, accepted_ndims=(3,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name=f'{egocentrically_align_pose_cuda.__name__} anchor_1_idx', min_value=0, max_value=data.shape[1], value=anchor_1_idx)
    check_int(name=f'{egocentrically_align_pose_cuda.__name__} anchor_2_idx', min_value=0, max_value=data.shape[1], value=anchor_2_idx)
    if anchor_1_idx == anchor_2_idx: raise InvalidInputError(msg=f'Anchor 1 index ({anchor_1_idx}) cannot be the same as Anchor 2 index ({anchor_2_idx})', source=egocentrically_align_pose_cuda.__name__)
    check_int(name=f'{egocentrically_align_pose_cuda.__name__} direction', value=direction, min_value=0, max_value=360)
    check_valid_array(data=anchor_location, source=egocentrically_align_pose_cuda.__name__, accepted_ndims=(1,), accepted_axis_0_shape=[2,], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name=f'{egocentrically_align_pose_cuda.__name__} batch_size', value=batch_size, min_value=1)
    results = np.full_like(a=data, fill_value=-1, dtype=np.int64)
    results_centers = np.full((data.shape[0], 2), fill_value=-1, dtype=np.int64)
    results_rotation_vectors = np.full((data.shape[0], 2, 2), fill_value=-1, dtype=np.float64)
    transposed_results_rotation_vectors = np.full((data.shape[0], 2, 2), fill_value=np.nan, dtype=np.float64)
    matrix_multiplier_arr = np.full((data.shape[0], data.shape[1], 2), fill_value=-1, dtype=np.int64)
    target_angle = np.deg2rad(direction)
    target_angle_dev = cuda.to_device(np.array([target_angle]))
    anchor_idx_dev = cuda.to_device(np.array([anchor_1_idx, anchor_2_idx]))
    anchor_loc_dev = cuda.to_device(anchor_location)

    for l in range(0, data.shape[0], batch_size):
        r = l + batch_size
        sample_data = np.ascontiguousarray(data[l:r]).astype(np.float64)
        sample_centers = np.ascontiguousarray(results_centers[l:r]).astype(np.int64)
        sample_rotation_vectors = np.ascontiguousarray(results_rotation_vectors[l:r].astype(np.float64))
        sample_transposed_rotation_vectors = np.ascontiguousarray(transposed_results_rotation_vectors[l:r])
        sample_matrix_multiplier_arr = np.ascontiguousarray(matrix_multiplier_arr[l:r])
        sample_results = np.ascontiguousarray(results[l:r].astype(np.float64))
        sample_data_dev = cuda.to_device(sample_data)
        sample_centers_dev = cuda.to_device(sample_centers)
        sample_matrix_multiplier_arr_dev = cuda.to_device(sample_matrix_multiplier_arr)
        sample_transposed_rotation_vectors_dev = cuda.to_device(sample_transposed_rotation_vectors)
        sample_rotation_vectors_dev = cuda.to_device(sample_rotation_vectors)
        sample_results_dev = cuda.to_device(sample_results)
        bpg = (sample_data.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
        _egocentric_align_kernel[bpg, THREADS_PER_BLOCK](sample_data_dev,
                                                         sample_centers_dev,
                                                         sample_rotation_vectors_dev,
                                                         sample_results_dev,
                                                         target_angle_dev,
                                                         anchor_idx_dev,
                                                         sample_transposed_rotation_vectors_dev,
                                                         sample_matrix_multiplier_arr_dev,
                                                         anchor_loc_dev)
        results[l:r] = sample_results_dev.copy_to_host()
        results_centers[l:r] = sample_centers_dev.copy_to_host()
        results_rotation_vectors[l:r] = sample_rotation_vectors_dev.copy_to_host()

    return results, results_centers, results_rotation_vectors


DATA_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/data/501_MA142_Gi_Saline_0513.csv"
VIDEO_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/videos/501_MA142_Gi_Saline_0513.mp4"
SAVE_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/videos/501_MA142_Gi_Saline_0513_rotated.mp4"
ANCHOR_LOC = np.array([300, 300])

df = read_df(file_path=DATA_PATH, file_type='csv')
bp_cols = [x for x in df.columns if not x.endswith('_p')]
data = df[bp_cols].values.reshape(len(df), int(len(bp_cols)/2), 2).astype(np.int64)
data, centers, rotation_matrices = egocentrically_align_pose_cuda(data=data, anchor_1_idx=6, anchor_2_idx=2, anchor_location=ANCHOR_LOC, direction=180,batch_size=36000000)

for i in [2000000, 4000000, 8000000, 16000000, 32000000, 64000000]:
    data = np.random.randint(0, 500, (i, 6, 2))
    times = []
    for j in range(3):
        start_t = time.perf_counter()
        data, centers, rotation_matrices = egocentrically_align_pose_cuda(data=data, anchor_1_idx=6, anchor_2_idx=2, anchor_location=ANCHOR_LOC, direction=180, batch_size=36000000)
        times.append(time.perf_counter() - start_t)
    print(i, '\t' * 4, np.mean(times), '\t' * 4, np.std(times))


# from simba.video_processors.egocentric_video_rotator import EgocentricVideoRotator
#
# runner = EgocentricVideoRotator(video_path=VIDEO_PATH, centers=centers, rotation_vectors=rotation_matrices, anchor_location=(300, 300))
# runner.run()

#_, centers, rotation_vectors = egocentrically_align_pose(data=data, anchor_1_idx=6, anchor_2_idx=2, anchor_location=ANCHOR_LOC, direction=0)
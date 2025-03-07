import cv2
import numpy as np
from numba import jit

from simba.utils.data import egocentrically_align_pose
from simba.utils.read_write import read_df, read_img_batch_from_video_gpu


@jit(nopython=True)
def center_rotation_warpaffine_vectors(rotation_vectors: np.ndarray, centers: np.ndarray):
    """
    Create WarpAffine vectors for rotating a video around the center. These are used for egocentric alignment of video.

    .. note::
       `rotation_vectors` and `centers` are returned by :func:`simba.utils.data.egocentrically_align_pose`, or :func:`simba.utils.data.egocentrically_align_pose_numba`
    """
    results = np.full((rotation_vectors.shape[0], 2, 3), fill_value=np.nan, dtype=np.float64)
    for idx in range(rotation_vectors.shape[0]):
        R, center = rotation_vectors[idx], centers[idx]
        top = np.hstack((R[0, :], np.array([-center[0] * R[0, 0] - center[1] * R[0, 1] + center[0]])))
        bottom = np.hstack((R[1, :], np.array([-center[0] * R[1, 0] - center[1] * R[1, 1] + center[1]])))
        results[idx] = np.vstack((top, bottom))
    return results

@jit(nopython=True)
def align_target_warpaffine_vectors(centers: np.ndarray, target: np.ndarray):
    """
    Create WarpAffine for placing original center at new target position. These are used for egocentric alignment of video.
    .. note::
       `centers` are returned by :func:`simba.utils.data.egocentrically_align_pose`, or :func:`simba.utils.data.egocentrically_align_pose_numba`
       `target` in the location in the image where the anchor body-part should be placed.
    """
    results = np.full((centers.shape[0], 2, 3), fill_value=np.nan, dtype=np.float64)
    for idx in range(centers.shape[0]):
        translation_x = target[0] - centers[idx][0]
        translation_y = target[1] - centers[idx][1]
        results[idx] = np.array([[1, 0, translation_x], [0, 1, translation_y]])
    return results


@jit(nopython=True, cache=True)
def _bilinear_interpolate(image: np.ndarray, x: int, y: int):
    """
    Helper called by :func:`simba.sandbox.warp_numba.egocentric_frm_rotator` Perform bilinear interpolation on an image at fractional coordinates (x, y). Assumes coordinates (x, y) are within the bounds of the image.
    """
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    dx, dy = x - x0, y - y0
    if x0 < 0 or x0 + 1 >= image.shape[1] or y0 < 0 or y0 + 1 >= image.shape[0]:
        return 0
    I00, I01 = image[y0, x0], image[y0, x0+1]
    I10, I11 = image[y0+1, x0], image[y0+1, x0+1]
    return (I00 * (1 - dx) * (1 - dy) + I01 * dx * (1 - dy) + I10 * (1 - dx) * dy + I11 * dx * dy)

@jit(nopython=True)
def egocentric_frm_rotator(frames: np.ndarray, rotation_matrices: np.ndarray) -> np.ndarray:
    """
    Rotates a sequence of frames using the provided rotation matrices in an egocentric manner.

    Applies a geometric transformation to each frame in the input sequence based on
    its corresponding rotation matrix. The transformation includes rotation and translation,
    followed by bilinear interpolation to map pixel values from the source frame to the output frame.

    .. note::
       To create rotation matrices, see :func:`simba.utils.data.center_rotation_warpaffine_vectors` and :func:`simba.utils.data.align_target_warpaffine_vectors`

    :param np.ndarray frames: A 4D array of shape (N, H, W, C)
    :param np.ndarray rotation_matrices: A 3D array of shape (N, 3, 3), where each 3x3 matrix represents an affine transformation for a corresponding frame. The matrix should include rotation and translation components.
    :return: A 4D array of shape (N, H, W, C), representing the warped frames after applying the transformations. The shape matches the input `frames`.
    :rtype: np.ndarray

    :example:
    >>> DATA_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/data/501_MA142_Gi_Saline_0513.csv"
    >>> VIDEO_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/videos/501_MA142_Gi_Saline_0513.mp4"
    >>> SAVE_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/videos/501_MA142_Gi_Saline_0513_rotated.mp4"
    >>> ANCHOR_LOC = np.array([300, 300])
    >>>
    >>> df = read_df(file_path=DATA_PATH, file_type='csv')
    >>> bp_cols = [x for x in df.columns if not x.endswith('_p')]
    >>> data = df[bp_cols].values.reshape(len(df), int(len(bp_cols)/2), 2).astype(np.int64)
    >>> data, centers, rotation_matrices = egocentrically_align_pose(data=data, anchor_1_idx=6, anchor_2_idx=2, anchor_location=ANCHOR_LOC, direction=180)
    >>> imgs = read_img_batch_from_video_gpu(video_path=VIDEO_PATH, start_frm=0, end_frm=100)
    >>> imgs = np.stack(list(imgs.values()), axis=0)
    >>>
    >>> rot_matrices_center = center_rotation_warpaffine_vectors(rotation_vectors=rotation_matrices, centers=centers)
    >>> rot_matrices_align = align_target_warpaffine_vectors(centers=centers, target=ANCHOR_LOC)
    >>>
    >>> imgs_centered = egocentric_frm_rotator(frames=imgs, rotation_matrices=rot_matrices_center)
    >>> imgs_out = egocentric_frm_rotator(frames=imgs_centered, rotation_matrices=rot_matrices_align)
    """

    N, H, W, C = frames.shape
    warped_frames = np.zeros_like(frames)
    for i in range(N):
        frame = frames[i]
        rotation_matrix = rotation_matrices[i]
        affine_matrix = rotation_matrix[:2, :2]
        translation = np.ascontiguousarray(rotation_matrix[:2, 2])
        inverse_affine_matrix = np.ascontiguousarray(np.linalg.inv(affine_matrix))
        inverse_translation = -np.dot(inverse_affine_matrix, translation)
        for r in range(H):
            for c in range(W):
                src_x = inverse_affine_matrix[0, 0] * c + inverse_affine_matrix[0, 1] * r + inverse_translation[0]
                src_y = inverse_affine_matrix[1, 0] * c + inverse_affine_matrix[1, 1] * r + inverse_translation[1]
                for ch in range(C):
                    warped_frames[i, r, c, ch] = _bilinear_interpolate(frame[:, :, ch], src_x, src_y)
    return warped_frames


DATA_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/data/501_MA142_Gi_Saline_0513.csv"
VIDEO_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/videos/501_MA142_Gi_Saline_0513.mp4"
SAVE_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/videos/501_MA142_Gi_Saline_0513_rotated.mp4"
ANCHOR_LOC = np.array([300, 300])

df = read_df(file_path=DATA_PATH, file_type='csv')
bp_cols = [x for x in df.columns if not x.endswith('_p')]
data = df[bp_cols].values.reshape(len(df), int(len(bp_cols)/2), 2).astype(np.int64)
data, centers, rotation_matrices = egocentrically_align_pose(data=data, anchor_1_idx=6, anchor_2_idx=2, anchor_location=ANCHOR_LOC, direction=180)
imgs = read_img_batch_from_video_gpu(video_path=VIDEO_PATH, start_frm=0, end_frm=100)
imgs = np.stack(list(imgs.values()), axis=0)

rot_matrices_center = center_rotation_warpaffine_vectors(rotation_vectors=rotation_matrices, centers=centers)
rot_matrices_align = align_target_warpaffine_vectors(centers=centers, target=ANCHOR_LOC)

imgs_centered = egocentric_frm_rotator(frames=imgs, rotation_matrices=rot_matrices_center)
imgs_out = egocentric_frm_rotator(frames=imgs_centered, rotation_matrices=rot_matrices_align)

for i in range(imgs_out.shape[0]):


    cv2.imshow('sadasdas', imgs_out[i])
    cv2.waitKey(60)




# DATA_PATH = r"C:\Users\sroni\OneDrive\Desktop\rotate_ex\data\501_MA142_Gi_Saline_0513.csv"
# VIDEO_PATH = r"C:\Users\sroni\OneDrive\Desktop\rotate_ex\videos\501_MA142_Gi_Saline_0513.mp4"
# SAVE_PATH = r"C:\Users\sroni\OneDrive\Desktop\rotate_ex\videos\501_MA142_Gi_Saline_0513_rotated.mp4"
# ANCHOR_LOC = np.array([250, 250])
#
# df = read_df(file_path=DATA_PATH, file_type='csv')
# bp_cols = [x for x in df.columns if not x.endswith('_p')]
# data = df[bp_cols].values.reshape(len(df), int(len(bp_cols)/2), 2).astype(np.int32)
#
_, centers, rotation_vectors = egocentrically_align_pose(data=data, anchor_1_idx=6, anchor_2_idx=2, anchor_location=ANCHOR_LOC, direction=0)
# p = center_rotation_warpaffine_vectors(rotation_vectors=rotation_vectors, centers=centers)
# k = align_target_warpaffine_vectors(centers=centers, target=ANCHOR_LOC)
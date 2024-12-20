import cv2
import numba
import numpy as np



@numba.jit(nopython=True, cache=True)
def _bilinear_interpolate(image: np.ndarray, x: int, y: int):
    """
    Perform bilinear interpolation on an image at fractional coordinates (x, y). Assumes coordinates (x, y) are within the bounds of the image.
    """

    x0, y0 = int(np.floor(x)), int(np.floor(y))
    dx, dy = x - x0, y - y0
    if x0 < 0 or x0 + 1 >= image.shape[1] or y0 < 0 or y0 + 1 >= image.shape[0]:
        return 0

    I00, I01 = image[y0, x0], image[y0, x0+1]
    I10, I11 = image[y0+1, x0], image[y0+1, x0+1]
    interpolated_value = (I00 * (1 - dx) * (1 - dy) + I01 * dx * (1 - dy) + I10 * (1 - dx) * dy + I11 * dx * dy)
    return interpolated_value

@numba.jit(nopython=True)
def warp_affine_numpy(frames: np.ndarray, rotation_matrices: np.ndarray):
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


from simba.utils.read_write import read_df, read_img_batch_from_video_gpu
from simba.utils.data import egocentrically_align_pose
from simba.sandbox.warp_numba import center_rotation_warpaffine_vectors, align_target_warpaffine_vectors
import cv2

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

imgs_centered = warp_affine_numpy(frames=imgs, rotation_matrices=rot_matrices_center)
imgs_out = warp_affine_numpy(frames=imgs_centered, rotation_matrices=rot_matrices_align)

for i in range(imgs_out.shape[0]):


    cv2.imshow('sadasdas', imgs_out[i])
    cv2.waitKey(60)

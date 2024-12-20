import math
import os
import time
from typing import Optional, Union

import cv2
import numpy as np


from simba.utils.read_write import read_df, read_img_batch_from_video_gpu, get_video_meta_data, get_fn_ext
from simba.utils.data import egocentrically_align_pose, align_target_warpaffine_vectors, center_rotation_warpaffine_vectors
from simba.utils.checks import check_valid_array, check_int, check_if_dir_exists
from simba.utils.enums import Formats
from simba.utils.errors import FrameRangeError
from numba import cuda, njit, prange, jit

THREADS_PER_BLOCK = 256

#@njit("(int32[:, :, :], int64, int64, int64, int64[:])")
@njit("(float64[:, :, :]),")
def _get_inverse_affine_matrices_inverse_translation(matrices: np.ndarray) -> np.ndarray:
    """ Helper for egodentric rotation of videos. This function is used by """
    matrices = np.ascontiguousarray(matrices)
    inverse_affine_matrix = np.ascontiguousarray(np.full((matrices.shape[0], 2, 2), fill_value=np.nan, dtype=np.float64))
    inverse_translation_matrix = np.full((matrices.shape[0], 2), fill_value=np.nan, dtype=np.float64)
    for i in prange(matrices.shape[0]):
        inverse_affine_matrix[i] = np.linalg.inv(matrices[i][:2, :2])
        inverse_translation_matrix[i] = -np.dot(inverse_affine_matrix[i], matrices[i][:2, 2])
    return inverse_affine_matrix, inverse_translation_matrix


@jit()
def get_target_translations(targets):
    rotation_matrix = np.eye(3)
    results = np.full((targets.shape[0], 3, 3), fill_value=np.nan, dtype=np.float64)
    for i in range(targets.shape[0]):
        transform_matrix = np.dot(rotation_matrix, targets[i])
        results[i] = np.linalg.inv(transform_matrix)
    return results


@cuda.jit()
def _egocentric_rotator_kernel_1(imgs, centers, target, rotation_matrices, results, video_width, video_height):
    frm_idx = cuda.grid(1)  # Get thread index

    # Ensure thread index is within bounds
    if frm_idx >= imgs.shape[0]:
        return

    img = imgs[frm_idx]  # The image for this frame
    center = centers[frm_idx]  # The center for this frame
    rotation_matrix = rotation_matrices[frm_idx]  # The rotation matrix for this frame

    # Allocate transformation matrices in local memory
    T_origin = cuda.local.array((3, 3), dtype=np.float32)
    R = cuda.local.array((3, 3), dtype=np.float32)
    T_target = cuda.local.array((3, 3), dtype=np.float32)
    T_final = cuda.local.array((3, 3), dtype=np.float32)

    # Step 1: Translate to origin (center -> (0,0))
    T_origin[0, 0] = 1
    T_origin[0, 1] = 0
    T_origin[0, 2] = -center[0]
    T_origin[1, 0] = 0
    T_origin[1, 1] = 1
    T_origin[1, 2] = -center[1]
    T_origin[2, 0] = 0
    T_origin[2, 1] = 0
    T_origin[2, 2] = 1

    # Step 2: Apply rotation matrix
    R[0, 0] = rotation_matrix[0, 0]
    R[0, 1] = rotation_matrix[0, 1]
    R[1, 0] = rotation_matrix[1, 0]
    R[1, 1] = rotation_matrix[1, 1]
    R[2, 2] = 1  # Ensure homogeneous coordinate is included

    # Step 3: Translate back to target position
    T_target[0, 0] = 1
    T_target[0, 1] = 0
    T_target[0, 2] = target[0]
    T_target[1, 0] = 0
    T_target[1, 1] = 1
    T_target[1, 2] = target[1]
    T_target[2, 0] = 0
    T_target[2, 1] = 0
    T_target[2, 2] = 1

    # Combine the transformations: T_final = T_target * R * T_origin
    for i in range(3):
        for j in range(3):
            T_final[i, j] = 0
            for k in range(3):
                T_final[i, j] += T_target[i, k] * R[k, j]

    for i in range(3):
        for j in range(3):
            T_final[i, j] += T_final[i, j] * T_origin[i, j]

    # Apply the final transformation to every pixel in the image
    video_height = video_height[0]
    video_width = video_width[0]
    for r in range(video_height):
        for c in range(video_width):
            # Transform coordinates
            coords_x = c * T_final[0, 0] + r * T_final[0, 1] + T_final[0, 2]
            coords_y = c * T_final[1, 0] + r * T_final[1, 1] + T_final[1, 2]
            print(coords_x, coords_y)

            # Check if transformed coordinates are within bounds
            if 0 <= coords_x < video_width and 0 <= coords_y < video_height:
                for ch in range(3):  # Assuming RGB images
                    results[frm_idx, r, c, ch] = img[int(coords_y), int(coords_x), ch]


@cuda.jit()
def _egocentric_rotator_kernel(imgs,
                               inverse_affines_center,
                               inverse_translations_center,
                               inverse_affines_target,
                               inverse_translations_target,
                               target_rotations,
                               results):
    frm_idx = cuda.grid(1)
    if frm_idx >= imgs.shape[0]:
        return
    else:
        img, inverse_affine_center, inverse_translation_center = imgs[frm_idx], inverse_affines_center[frm_idx], inverse_translations_center[frm_idx]
        inverse_affine_target, inverse_translation_target = inverse_affines_target[frm_idx], inverse_translations_target[frm_idx]
        target_rotation = target_rotations[frm_idx]
        H, W, C = img.shape
        for r in range(H):
            for c in range(W):
                src_x = inverse_affine_center[0, 0] * c + inverse_affine_center[0, 1] * r + inverse_translation_center[0]
                src_y = inverse_affine_center[1, 0] * c + inverse_affine_center[1, 1] * r + inverse_translation_center[1]
                x0 = int(math.floor(src_x))
                y0 = int(math.floor(src_y))
                dx, dy = int(src_x - x0), int(src_y - y0)
                for ch in range(C):
                    if x0 < 0 or x0 + 1 >= img.shape[1] or y0 < 0 or y0 + 1 >= img.shape[0]:
                        results[frm_idx, r, c, ch] = 0
                    else:
                        I00, I01 = img[y0, x0][0], img[y0, x0 + 1][0]
                        I10, I11 = img[y0 + 1, x0][0], img[y0 + 1, x0 + 1][0]
                        val = I00 * (1 - dx) * (1 - dy) + I01 * dx * (1 - dy) + I10 * (1 - dx) * dy + I11 * dx * dy
                        results[frm_idx, r, c, ch] = val

        cuda.syncthreads()

        #img = results[frm_idx]
        for r in range(H):
            for c in range(W):
                src_x = int(c - target_rotations[frm_idx, 0, 2])
                src_y = int(r - target_rotations[frm_idx, 1, 2])
                if 0 <= src_x < W and 0 <= src_y < H:
                    for ch in range(C):
                        results[frm_idx, r, c, ch] = results[frm_idx, src_y, src_x, ch]
                else:
                    print(src_x, src_y)
                    for ch in range(C):
                        results[frm_idx, r, c, ch] = 255
        #
        #
        #         src_x = inverse_affine_target[0, 0] * c + inverse_affine_target[0, 1] * r + inverse_translation_target[0]
        #         src_y = inverse_affine_target[1, 0] * c + inverse_affine_target[1, 1] * r + inverse_translation_target[1]
        #         x0 = int(math.floor(src_x))
        #         y0 = int(math.floor(src_y))
        #         dx, dy = int(src_x - x0), int(src_y - y0)
        #         for ch in range(C):
        #             if x0 < 0 or x0 + 1 >= img.shape[1] or y0 < 0 or y0 + 1 >= img.shape[0]:
        #                 results[frm_idx, r, c, ch] = 0
        #             else:
        #                 I00, I01 = img[y0, x0][0], img[y0, x0 + 1][0]
        #                 I10, I11 = img[y0 + 1, x0][0], img[y0 + 1, x0 + 1][0]
        #                 results[frm_idx, r, c, ch] = I00 * (1 - dx) * (1 - dy) + I01 * dx * (1 - dy) + I10 * (1 - dx) * dy + I11 * dx * dy
        #

def egocentric_video_rotator_cuda(video_path: np.ndarray,
                                  rotation_matrix: np.ndarray,
                                  center_matrix: np.ndarray,
                                  anchor_loc: np.ndarray,
                                  batch_size: Optional[int] = 100,
                                  save_path: Optional[Union[str, os.PathLike]] = None):

    video_meta_data = get_video_meta_data(video_path=video_path)
    if video_meta_data['frame_count'] != rotation_matrix.shape[0]:
        raise FrameRangeError(msg=f'The video {video_path} contains {video_meta_data["frame_count"]} frames while the the rotation_matrix has data for {rotation_matrix.shape[0]} frames', source=egocentric_video_rotator_cuda.__name__)
    if video_meta_data['frame_count'] != center_matrix.shape[0]:
        raise FrameRangeError(msg=f'The video {video_path} contains {video_meta_data["frame_count"]} frames while the the center_matrix has data for {center_matrix.shape[0]} frames', source=egocentric_video_rotator_cuda.__name__)
    if rotation_matrix.shape[0] != center_matrix.shape[0]:
        raise FrameRangeError(msg=f'The center_matrix has data for {center_matrix.shape[0]} frames while the rotation_matrix has data for {rotation_matrix.shape[0]} frames', source=egocentric_video_rotator_cuda.__name__)
    check_int(name=f'{egocentric_video_rotator_cuda.__name__} batch_size', value=batch_size, min_value=1)
    check_valid_array(data=anchor_loc, source=f'{egocentric_video_rotator_cuda.__name__} anchor_loc', accepted_axis_0_shape=[2,], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    video_dir, video_name, video_ext = get_fn_ext(filepath=video_path)
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=egocentric_video_rotator_cuda.__name__)
    else:
        save_path = os.path.join(video_dir, f'{video_name}_rotated.mp4')
    fourcc = cv2.VideoWriter_fourcc(*f'{Formats.MP4_CODEC.value}')
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'], (video_meta_data['width'], video_meta_data['height']))
    for batch_cnt, l in enumerate(range(0, video_meta_data['frame_count'], batch_size)):
        r = min(l + batch_size -1, video_meta_data['frame_count'])
        print(f'Reading frames {l}-{r} (video: {video_name}, frames: {video_meta_data["frame_count"]})...')
        batch_imgs = read_img_batch_from_video_gpu(video_path=VIDEO_PATH, start_frm=l, end_frm=r)
        batch_imgs = np.ascontiguousarray(np.stack(list(batch_imgs.values()), axis=0)).astype(np.uint8)
        batch_rot = np.ascontiguousarray(rotation_matrix[l:r+1]).astype(np.float64)
        batch_results = np.full_like(batch_imgs, fill_value=0, dtype=np.uint8)
        batch_centers = np.ascontiguousarray(center_matrix[l:r+1])
        batch_center_rotations = center_rotation_warpaffine_vectors(rotation_vectors=batch_rot, centers=batch_centers)
        batch_target_rotations = align_target_warpaffine_vectors(centers=batch_centers, target=anchor_loc)
        inverse_affine_center, inverse_translation_center = _get_inverse_affine_matrices_inverse_translation(matrices=batch_center_rotations)
        inverse_affine_target, inverse_translation_target = _get_inverse_affine_matrices_inverse_translation(matrices=batch_target_rotations)
        batch_imgs_dev = cuda.to_device(batch_imgs)
        batch_results_dev = cuda.to_device(batch_results)
        inverse_affine_center_dev = cuda.to_device(inverse_affine_center)
        inverse_translation_center_dev = cuda.to_device(inverse_translation_center)
        inverse_affine_target_dev = cuda.to_device(inverse_affine_target)
        inverse_translation_target_dev = cuda.to_device(inverse_translation_target)
        batch_target_rotations_dev = cuda.to_device(batch_target_rotations)
        bpg = (batch_imgs_dev.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK

        batch_centers_dev = cuda.to_device(batch_centers)
        target_dev = cuda.to_device(anchor_loc)
        batch_rot_dev = cuda.to_device(batch_rot)
        w_dev = cuda.to_device(np.array([video_meta_data['width']]))
        h_dev = cuda.to_device(np.array([video_meta_data['height']]))

        _egocentric_rotator_kernel_1[bpg, THREADS_PER_BLOCK](batch_imgs_dev,
                                     batch_centers_dev,
                                     target_dev,
                                     batch_rot_dev,
                                     batch_results_dev,
                                     w_dev,
                                     h_dev)
        # _egocentric_rotator_kernel[bpg, THREADS_PER_BLOCK](batch_imgs_dev,
        #                                                    inverse_affine_center_dev,
        #                                                    inverse_translation_center_dev,
        #                                                    inverse_affine_target_dev,
        #                                                    inverse_translation_target_dev,
        #                                                    batch_target_rotations_dev,
        #                                                    batch_results_dev)
        out_results = batch_results_dev.copy_to_host()
        errors, correct = [], []
        for frm in range(out_results.shape[0]):
            f = out_results[frm]
            # if len(np.unique(f)) == 1:
            #     errors.append(batch_target_rotations[frm])
            # else:
            #     correct.append(batch_target_rotations[frm])
            print(f.shape)
            writer.write(cv2.cvtColor(out_results[frm], cv2.COLOR_BGR2RGB))
        #     writer.write(f)
        print(errors[0:5], '\n', correct[0:5])
        if batch_cnt > 1:
           break
            #writer.write(cv2.cvtColor(batch_results[frm], cv2.COLOR_BGR2RGB))
    writer.release()


        #results[l:r] =
    #
    # return results
    # #

DATA_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/data/501_MA142_Gi_Saline_0513.csv"
VIDEO_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/videos/501_MA142_Gi_Saline_0513.mp4"
SAVE_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/videos/501_MA142_Gi_Saline_0513_rotated.mp4"
ANCHOR_LOC = np.array([250, 250])

df = read_df(file_path=DATA_PATH, file_type='csv')
bp_cols = [x for x in df.columns if not x.endswith('_p')]
data = df[bp_cols].values.reshape(len(df), int(len(bp_cols)/2), 2).astype(np.int32).astype(np.int32)
rotated_pose, centers, rotation_vectors = egocentrically_align_pose(data=data, anchor_1_idx=5, anchor_2_idx=2, anchor_location=ANCHOR_LOC, direction=0)

# imgs = read_img_batch_from_video_gpu(video_path=VIDEO_PATH, start_frm=0, end_frm=500)
# imgs = np.stack(list(imgs.values()), axis=0)
start = time.perf_counter()
results = egocentric_video_rotator_cuda(video_path=VIDEO_PATH,
                                        rotation_matrix=rotation_vectors,
                                        anchor_loc=ANCHOR_LOC,
                                        center_matrix=centers,
                                        batch_size=1000)
print(time.perf_counter() - start)
# #
# for i in range(results.shape[0]):
#     print(i)
#     cv2.imshow('asdasd', results[i])
#     cv2.waitKey(60)


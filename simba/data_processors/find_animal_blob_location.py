import functools
import gc
import multiprocessing
import os
import time
from copy import copy, deepcopy
from typing import Dict, Optional, Tuple, Union
from scipy.spatial.distance import cdist

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Polygon

from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.checks import (check_float, check_instance, check_int,
                                check_nvidea_gpu_available,
                                check_valid_boolean, is_img_bw)
from simba.utils.data import resample_geometry_vertices
from simba.utils.enums import Defaults
from simba.utils.errors import FFMPEGCodecGPUError, SimBAGPUError
from simba.utils.lookups import get_available_ram
from simba.utils.read_write import (find_core_cnt, get_fn_ext,
                                    get_memory_usage_array,
                                    get_video_meta_data, img_stack_to_bw, read_frm_of_video,
                                    read_img_batch_from_video,
                                    read_img_batch_from_video_gpu)


def stabilize_body_parts(bp_1: np.ndarray,
                         bp_2: np.ndarray,
                         center_positions: np.ndarray,
                         max_jump_distance: int = 20,
                         smoothing_factor: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:

    d1 = np.linalg.norm(bp_1[0] - center_positions[0])
    d2 = np.linalg.norm(bp_2[0] - center_positions[0])
    if d1 < d2:
        stable_nose, stable_tail = bp_1.copy(), bp_2.copy()
    else:
        stable_nose, stable_tail = bp_2.copy(), bp_1.copy()
    prev_velocity = np.zeros_like(center_positions[0])
    for i in range(1, len(bp_1)):
        dC = center_positions[i] - center_positions[i - 1]
        velocity = dC / np.linalg.norm(dC) if np.linalg.norm(dC) > 0 else np.zeros_like(dC)
        smoothed_velocity = smoothing_factor * prev_velocity + (1 - smoothing_factor) * velocity
        prev_velocity = smoothed_velocity
        dist_nose = np.linalg.norm(stable_nose[i - 1] - center_positions[i - 1])
        dist_tail = np.linalg.norm(stable_tail[i - 1] - center_positions[i - 1])
        if dist_nose > dist_tail:
            stable_nose[i] = bp_1[i] if np.linalg.norm(bp_1[i] - center_positions[i]) < np.linalg.norm(
                bp_2[i] - center_positions[i]) else bp_2[i]
            stable_tail[i] = bp_2[i] if stable_nose[i] is bp_1[i] else bp_1[i]
        else:
            stable_nose[i] = bp_1[i] if np.linalg.norm(bp_1[i] - center_positions[i]) < np.linalg.norm(
                bp_2[i] - center_positions[i]) else bp_2[i]
            stable_tail[i] = bp_2[i] if stable_nose[i] is bp_1[i] else bp_1[i]
        nose_jump = np.linalg.norm(stable_nose[i] - stable_nose[i - 1])
        tail_jump = np.linalg.norm(stable_tail[i] - stable_tail[i - 1])
        if nose_jump > max_jump_distance:
            stable_nose[i] = stable_nose[i - 1] + (stable_nose[i] - stable_nose[i - 1]) * (max_jump_distance / nose_jump)
        if tail_jump > max_jump_distance:
            stable_tail[i] = stable_tail[i - 1] + (stable_tail[i] - stable_tail[i - 1]) * (max_jump_distance / tail_jump)

    return stable_nose, stable_tail


def get_hull_from_vertices(vertices: np.ndarray) -> Tuple[bool, np.ndarray]:

    vertices = np.unique(vertices, axis=0).astype(int)
    if vertices.shape[0] < 3:
        return False, np.full((vertices.shape[0], 2), fill_value=0, dtype=np.int32)
    for i in range(1, vertices.shape[0]):
        if (vertices[i] != vertices[0]).all():
            try:
                return (True, vertices[ConvexHull(vertices).vertices])
            except QhullError:
                return False, np.full((vertices.shape[0], 2), fill_value=0, dtype=np.int32)
        else:
            pass
    return False, np.full((vertices.shape[0], 2), fill_value=0, dtype=np.int32)

def get_nose_tail_from_vertices(vertices: np.ndarray, smooth_factor=5):
    T, N, _ = vertices.shape
    anterior = np.full((T, 2), -1, dtype=np.float32)
    posterior = np.full((T, 2), -1, dtype=np.float32)

    centroids = np.mean(vertices, axis=1)  # Compute centroids (T, 2)
    cumulative_motion = centroids - centroids[0]
    pairwise_dists = np.array([cdist(frame, frame) for frame in vertices])
    max_indices = np.array([np.unravel_index(np.argmax(dists), dists.shape) for dists in pairwise_dists])

    # Initialize head and tail based on first frame
    first_farthest_pts = vertices[0][max_indices[0]]
    anterior[0], posterior[0] = first_farthest_pts
    head_history = [anterior[0]]

    for idx in range(1, T):
        farthest_two_pts = vertices[idx][max_indices[idx]]
        motion_vector = cumulative_motion[idx]
        projections = np.dot(farthest_two_pts - centroids[idx], motion_vector)
        head_idx = np.argmax(projections)
        tail_idx = 1 - head_idx
        candidate_head = farthest_two_pts[head_idx]
        candidate_tail = farthest_two_pts[tail_idx]


        anterior[idx] = candidate_head
        posterior[idx] = candidate_tail
        head_history.append(candidate_head)

        if len(head_history) > smooth_factor:
            head_history.pop(0)

    return anterior, posterior



def get_left_right_points(hull_vertices: np.ndarray,
                           anterior: np.ndarray,
                           center: np.ndarray,
                           posterior: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Ensure inputs are 3D for consistency (n_images, n_points, 2)
    if hull_vertices.ndim != 3 or anterior.ndim != 2 or center.ndim != 2 or posterior.ndim != 2:
        raise ValueError("hull_vertices must be 3D, while anterior, center, and posterior must be 2D.")

    direction = posterior[:, None, :] - anterior[:, None, :]
    perp_vector = np.stack([-direction[..., 1], direction[..., 0]], axis=-1)

    projections = np.einsum('ijk,ijk->ij', hull_vertices - center[:, None, :], perp_vector)
    left_mask = projections < 0
    right_mask = projections > 0

    dist = np.linalg.norm(hull_vertices - center[:, None, :], axis=-1)

    left_points = np.where(left_mask, dist, np.inf).argmin(axis=1)
    right_points = np.where(right_mask, dist, np.inf).argmin(axis=1)

    left_points = hull_vertices[np.arange(hull_vertices.shape[0]), left_points]
    right_points = hull_vertices[np.arange(hull_vertices.shape[0]), right_points]

    return left_points, right_points




def get_blob_vertices_from_imgs(imgs: Dict[int, np.ndarray],
                                verbose: bool = False,
                                video_name: Optional[str] = None,
                                inclusion_zone: Optional[Union[Polygon, MultiPolygon,]] = None,
                                window_size: Optional[int] = None,
                                convex_hull: bool = False,
                                vertice_cnt: int = 50) -> Dict[int, Dict[str, Union[int, np.ndarray]]]:
    """
    Helper to find the largest connected component in binary image. E.g., Use to find a "blob" (i.e., animal) within a background subtracted image.

    .. seealso::
       To create background subtracted videos, use e.g., :func:`simba.video_processors.video_processing.video_bg_subtraction_mp`, or :func:`~simba.video_processors.video_processing.video_bg_subtraction`.
       To get ``img`` dict, use :func:`~simba.utils.read_write.read_img_batch_from_video_gpu` or :func:`~simba.mixins.image_mixin.ImageMixin.read_img_batch_from_video`.
       For relevant notebook, see `BACKGROUND REMOVAL <https://simba-uw-tf-dev.readthedocs.io/en/latest/nb/bg_remove.html>`__.

    .. important::
       Pass black and white [0, 255] pixel values only, where the foreground is 255 and background is 0.

    :param Dict[int, np.ndarray] imgs: Dictionary of images where the key is the frame id and the value is an image in np.ndarray format.
    :param bool verbose: If True, prints progress. Default: False.
    :param video_name video_name: The name of the video being processed for interpretable progress msg if ``verbose``.
    :param Optional[Union[Polygon, MultiPolygon]] inclusion_zone: Optional shapely polygon, or multipolygon, restricting where to search for the largest blob. Default: None.
    :param Optional[int] window_size: If not None, then integer representing the size multiplier of the animal geometry in previous frame. If not None, the animal geometry will only be searched for within this geometry.
    :return: Dictionary where the key is the frame id and the value is a 2D array with x and y coordinates.
    :rtype: Dict[int, np.ndarray]

    :example:
    >>> imgs = read_img_batch_from_video_gpu(video_path=r"C:\troubleshooting\mitra\test\temp\501_MA142_Gi_Saline_0515.mp4", start_frm=0, end_frm=0, black_and_white=True)
    >>> data = get_blob_vertices_from_imgs(imgs=imgs, window_size=3)
    >>> data = pd.DataFrame.from_dict(data, orient='index')
    """


    check_valid_boolean(value=[verbose], source=f'{get_blob_vertices_from_imgs.__name__} verbose', raise_error=True)
    if inclusion_zone is not None:
        check_instance(source=f'{get_blob_vertices_from_imgs.__name__} inclusion_zone', instance=inclusion_zone, accepted_types=(MultiPolygon, Polygon,), raise_error=True)
    if window_size is not None:
        check_float(name='window_size', value=window_size, min_value=1.0, raise_error=True)
    check_int(name=f'{get_blob_vertices_from_imgs.__name__} vertice_cnt', value=vertice_cnt, min_value=3, raise_error=True)
    results, prior_window = {}, None
    for frm_idx, img in imgs.items():
        if verbose:
            if video_name is None: print(f'Finding animal in frame {frm_idx}...')
            else: print(f'Finding animal in frame {frm_idx} ({video_name})...')
        is_img_bw(img=img, raise_error=True, source=f'{get_blob_vertices_from_imgs.__name__} {frm_idx}')
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        contours = [cnt.reshape(1, -1, 2) for cnt in contours if len(cnt) >= 3]
        geometries = GeometryMixin().contours_to_geometries(contours=contours, force_rectangles=False, convex_hull=convex_hull)
        geometries = [g for g in geometries if g.is_valid]
        if len(geometries) == 0:
            results[frm_idx] = np.full(shape=(vertice_cnt, 2), fill_value=np.nan, dtype=np.float32)
        else:
            if inclusion_zone is not None:
                geo_idx = [inclusion_zone.intersects(x) for x in geometries]
                selected_polygons = [geom for geom, is_inside in zip(geometries, geo_idx) if is_inside]
                geometries = deepcopy(selected_polygons)
            if prior_window is not None:
                geo_idx = [prior_window.intersects(x) for x in geometries]
                selected_polygons = [geom for geom, is_inside in zip(geometries, geo_idx) if is_inside]
                geometries = deepcopy(selected_polygons)
            if len(geometries) == 0:
                results[frm_idx] = np.full(shape=(vertice_cnt, 2), fill_value=np.nan, dtype=np.float32)
            else:
                geometry_stats = GeometryMixin().get_shape_statistics(shapes=geometries)
                geometry = geometries[np.argmax(np.array(geometry_stats['areas']))]
                if window_size is not None:
                    window_geometry = GeometryMixin.minimum_rotated_rectangle(shape=geometry)
                    prior_window = scale(window_geometry, xfact=window_size, yfact=window_size, origin=window_geometry.centroid)
                vertices = np.array(geometry.exterior.coords).astype(np.int32)
                results[frm_idx] = resample_geometry_vertices(vertices=vertices.reshape(-1, vertices.shape[0], vertices.shape[1]), vertice_cnt=vertice_cnt)[0]

    return results

def get_blob_vertices_from_video(video_path: Union[str, os.PathLike],
                                 gpu: bool = False,
                                 core_cnt: int = -1,
                                 verbose: bool = True,
                                 inclusion_zone: Optional[Union[Polygon, MultiPolygon]] = None,
                                 window_size: Optional[float] = None,
                                 batch_size: Optional[int] = None,
                                 convex_hull: bool = False,
                                 vertice_cnt: int = 50) -> np.ndarray:

    """
    Detects the location of the largest blob in each frame of a video. Processes frames in batches and optionally uses GPU for acceleration. Results can be saved to a specified path or returned as a NumPy array.

    .. seealso::
       For visualization of results, see :func:`simba.plotting.blob_plotter.BlobPlotter` and :func:`simba.mixins.plotting_mixin.PlottingMixin._plot_blobs`
       Background subtraction can be performed using :func:`~simba.video_processors.video_processing.video_bg_subtraction_mp` or :func:`~simba.video_processors.video_processing.video_bg_subtraction`.

    .. note::
       In ``inclusion_zones`` is not None, then the largest blob will be searches for **inside** the passed vertices.

    :param Union[str, os.PathLike] video_path: Path to the video file from which to extract frames. Often, a background subtracted video, which can be created with e.g., :func:`simba.video_processors.video_processing.video_bg_subtraction_mp`.
    :param Optional[int] batch_size: Number of frames to process in each batch. Default is 3k.
    :param Optional[bool] gpu: Whether to use GPU acceleration for processing. Default is False.
    :param Optional[bool] verbose: Whether to print progress and status messages. Default is True.
    :param Optional[Union[Polygon, MultiPolygon]] inclusion_zones: Optional shapely polygon, or multipolygon, restricting where to search for the largest blob. Default: None.
    :param Optional[int] window_size: If not None, then integer representing the size multiplier of the animal geometry in previous frame. If not None, the animal geometry will only be searched for within this geometry.
    :param bool convex_hull:  If True, creates the convex hull of the shape, which is the smallest convex polygon that encloses the shape. Default True.
    :return: A dataframe shape (N, 4) where N is the number of frames, containing the X and Y coordinates of the centroid of the largest blob in each frame and the vertices representing the hull. If `save_path` is provided, returns None.
    :rtype: Union[None, pd.DataFrame]

    :example:
    >>> x = get_blob_vertices_from_video(video_path=r"/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/2022-06-20_NOB_DOT_4_downsampled_bg_subtracted.mp4", gpu=True)
    >>> y = get_blob_vertices_from_video(video_path=r"C:\troubleshooting\RAT_NOR\project_folder\videos\2022-06-20_NOB_IOT_1_bg_subtracted.mp4", gpu=True)
    """

    video_meta = get_video_meta_data(video_path=video_path)
    _, video_name, _ = get_fn_ext(filepath=video_path)
    if batch_size is None:
        img = read_frm_of_video(video_path=video_path, frame_index=0)
        img_size = max(1, int(get_memory_usage_array(x=img)["megabytes"]))
        available_ram = get_available_ram()['available_mb']
        safe_ram = available_ram * 0.25
        batch_size = min(max(1, int(safe_ram / img_size)), video_meta['frame_count'])
    else:
        check_int(name=f'{get_blob_vertices_from_video.__name__} batch_size', value=batch_size, min_value=1)
        if batch_size > video_meta['frame_count']: batch_size = video_meta['frame_count']
    _, video_name, _ = get_fn_ext(filepath=video_path)
    check_valid_boolean(value=gpu, source=f'{get_blob_vertices_from_video.__name__} gpu')
    check_valid_boolean(value=verbose, source=f'{get_blob_vertices_from_video.__name__} verbose')
    check_int(name=f'{get_blob_vertices_from_video.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0], raise_error=True)
    check_int(name=f'{get_blob_vertices_from_video.__name__} vertice_cnt', value=vertice_cnt, min_value=3, raise_error=True)
    core_cnt = find_core_cnt()[0] if core_cnt == -1 else core_cnt
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg='No GPU detected, try to set GPU to False', source=get_blob_vertices_from_video.__name__)
    if inclusion_zone is not None:
        check_instance(source=f'{get_blob_vertices_from_video} inclusion_zone', instance=inclusion_zone, accepted_types=(MultiPolygon, Polygon,), raise_error=True)
    if window_size is not None:
        check_float(name='window_size', value=window_size, min_value=1.0, raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise SimBAGPUError(msg='GPU is set to True, but SImBA could not find a GPU on the machine', source=get_blob_vertices_from_video.__name__)
    frame_ids = list(range(0, video_meta['frame_count']))
    frame_ids = [frame_ids[i:i + batch_size] for i in range(0, len(frame_ids), batch_size)]
    results = {}
    if verbose:
        print(f'Starting animal location detection for video {video_meta["video_name"]}...')
    with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
        constants = functools.partial(get_blob_vertices_from_imgs, verbose=verbose, video_name=video_name, inclusion_zone=inclusion_zone, convex_hull=convex_hull, vertice_cnt=vertice_cnt)
        for frame_batch in range(len(frame_ids)):
            print(f"Processing frame batch {frame_batch + 1}/{len(frame_ids)} (Video: {video_meta['video_name']}, batch_size: {batch_size}, available_ram: {available_ram})")
            time.sleep(3)
            start_frm, end_frm = frame_ids[frame_batch][0], frame_ids[frame_batch][-1]
            if gpu:
                imgs = read_img_batch_from_video_gpu(video_path=video_path, start_frm=start_frm, end_frm=end_frm, verbose=False, greyscale=False, black_and_white=False, out_format='dict')
            else:
                imgs = read_img_batch_from_video(video_path=video_path, start_frm=start_frm, end_frm=end_frm, verbose=False, black_and_white=False, greyscale=False, core_cnt=core_cnt)
            imgs = img_stack_to_bw(imgs=np.stack(list(imgs.values()), axis=0))
            imgs = {key: imgs[i] for i, key in enumerate(np.arange(start_frm, end_frm + 1) )}
            imgs = [dict(subset) for subset in np.array_split(list(imgs.items()), core_cnt)]
            for cnt, result in enumerate(pool.map(constants, imgs, chunksize=1)):
                results.update(result)

    pool.join()
    pool.terminate()
    gc.collect()
    results = dict(sorted(results.items()))

    return np.stack(list(results.values()), axis=0).astype(np.int32)




# video_path = r"D:\EPM\large_sample\out\2025-02-24 08-25-56.mp4"
#
# #video_path = r"D:\open_field_3\sample\1.mp4"
#
#
# if __name__ == "__main__":
#     new_start = time.time()
#     y = get_blob_vertices_from_video(video_path=video_path, gpu=True, verbose=True, batch_size=None)
#     new_time = time.time() - new_start
#     print(new_time)
#loop = asyncio.get_event_loop()
#loop.run_until_complete(get_blob_vertices_from_video(video_path=r"D:\open_field_3\sample\1.mp4", gpu=True, verbose=False))

#asyncio.run(get_blob_vertices_from_video(video_path=r"D:\open_field_3\sample\1.mp4", gpu=True, verbose=False))

#get_blob_locations(video_path=r"D:\EPM\sampled\.temp\1.mp4", gpu=False)


# if __name__ == "__main__":
#     blob_location = get_blob_locations(video_path=r"D:\open_field_3\sample\.temp\10164671.mp4", gpu=False)


# from shapely.ops import unary_union
# from scipy.spatial import ConvexHull
# #
#imgs = read_img_batch_from_video(video_path=r"D:\open_field_3\sample\.temp\10164671.mp4", start_frm=0, end_frm=5, black_and_white=True)
# imgs = read_img_batch_from_video_gpu(video_path=r"D:\open_field_3\sample\.temp\10164671.mp4", start_frm=0, end_frm=1, black_and_white=True)
# x = find_animal_blob_location(imgs=imgs)
#
#
#
# frame_1_data = x[0]
# center = GeometryMixin.bodyparts_to_points(data=np.array([[frame_1_data['center_x'], frame_1_data['center_y']]]))
# nose = GeometryMixin.bodyparts_to_points(data=np.array([[frame_1_data['nose_x'], frame_1_data['nose_y']]]))
# left = GeometryMixin.bodyparts_to_points(data=np.array([[frame_1_data['left_x'], frame_1_data['left_y']]]))
# right = GeometryMixin.bodyparts_to_points(data=np.array([[frame_1_data['right_x'], frame_1_data['right_y']]]))
# tail = GeometryMixin.bodyparts_to_points(data=np.array([[frame_1_data['tail_x'], frame_1_data['tail_y']]]))
#
#
# img = GeometryMixin.view_shapes(shapes=[center[0], nose[0], left[0], right[0], tail[0]], bg_img=imgs[0], circle_size=10)
# # #
# cv2.imshow('dsfsdfd', img)
# cv2.waitKey(50000)

#
# #get_blob_locations(video_path=r"D:\open_field_3\sample\.temp\10164671.mp4", gpu=False)
#
# # if __name__ == "__main__":
# #     get_blob_locations(video_path=r"D:\open_field_3\sample\.temp\10164671.mp4", gpu=False)
#
#
#
# #
# # imgs = read_img_batch_from_video_gpu(video_path=r"C:\troubleshooting\mitra\test\temp\501_MA142_Gi_Saline_0515.mp4", start_frm=0, end_frm=0, black_and_white=True)
# # data = find_animal_blob_location(imgs=imgs, window_size=3)
# # data = pd.DataFrame.from_dict(data, orient='index')
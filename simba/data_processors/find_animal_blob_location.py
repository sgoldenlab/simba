from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import scale
from typing import Dict, Optional, Union
from copy import copy
import numpy as np
import cv2
import os
import pandas as pd
from simba.utils.read_write import read_img_batch_from_video_gpu, get_video_meta_data, find_core_cnt, get_fn_ext, read_img_batch_from_video
from simba.utils.checks import (check_float, check_instance, check_valid_boolean, is_img_bw, check_int, check_nvidea_gpu_available)
from simba.utils.printing import SimbaTimer
from simba.utils.errors import NotDirectoryError, FFMPEGCodecGPUError
import functools
import multiprocessing
from simba.utils.enums import Defaults

def find_animal_blob_location(imgs: Dict[int, np.ndarray],
                              verbose: bool = False,
                              video_name: Optional[str] = None,
                              inclusion_zone: Optional[Union[Polygon, MultiPolygon,]] = None,
                              window_size: Optional[int] = None) -> Dict[int, Dict[str, Union[int, np.ndarray]]]:
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
    >>> data = find_animal_blob_location(imgs=imgs, window_size=3)
    >>> data = pd.DataFrame.from_dict(data, orient='index')
    """

    from simba.mixins.geometry_mixin import GeometryMixin
    check_valid_boolean(value=[verbose], source=f'{find_animal_blob_location.__name__} verbose', raise_error=True)
    if inclusion_zone is not None:
        check_instance(source=f'{find_animal_blob_location.__name__} inclusion_zone', instance=inclusion_zone, accepted_types=(MultiPolygon, Polygon,), raise_error=True)
    if window_size is not None:
        check_float(name='window_size', value=window_size, min_value=1.0, raise_error=True)
    results, prior_window = {}, None
    for frm_idx, img in imgs.items():
        if verbose:
            if video_name is None: print(f'Finding blob in image {frm_idx}...')
            else: print(f'Finding blob in image {frm_idx} (Video {video_name})...')
        is_img_bw(img=img, raise_error=True, source=f'{find_animal_blob_location.__name__} {frm_idx}')
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        contours = [cnt.reshape(1, -1, 2) for cnt in contours if len(cnt) >= 3]
        geometries = GeometryMixin().contours_to_geometries(contours=contours, force_rectangles=False)
        if inclusion_zone is not None:
            geo_idx = [inclusion_zone.contains(x.centroid) for x in geometries]
            selected_polygons = [geometries[i] for i in geo_idx]
            geometries = copy(selected_polygons)
        if prior_window is not None:
            geo_idx = [prior_window.contains(x.centroid) for x in geometries]
            selected_polygons = [geometries[i] for i in geo_idx]
            geometries = copy(selected_polygons)
        geometry_stats = GeometryMixin().get_shape_statistics(shapes=geometries)
        geometry = geometries[np.argmax(np.array(geometry_stats['areas']))].convex_hull.simplify(tolerance=5)
        if window_size is not None:
            window_geometry = GeometryMixin.minimum_rotated_rectangle(shape=geometry)
            prior_window = scale(window_geometry, xfact=window_size, yfact=window_size, origin=window_geometry.centroid)
        center = np.array(geometry.centroid.coords)[0].astype(np.int32)
        vertices = np.array(geometry.exterior.coords).astype(np.int32)
        results[frm_idx] = {'x': center[0], 'y': center[1], 'vertices': vertices}
    return results


def get_blob_locations(video_path: Union[str, os.PathLike],
                       batch_size: int = 3000,
                       gpu: bool = False,
                       verbose: bool = True,
                       save_path: Optional[Union[str, os.PathLike]] = None,
                       inclusion_zone: Optional[Union[Polygon, MultiPolygon]] = None,
                       window_size: Optional[float] = None) -> Union[None, pd.DataFrame]:
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
    :param Optional[Union[str, os.PathLike]] save_path: Path to save the results as a CSV file. If None, results are not saved.
    :param Optional[Union[Polygon, MultiPolygon]] inclusion_zones: Optional shapely polygon, or multipolygon, restricting where to search for the largest blob. Default: None.
    :param Optional[int] window_size: If not None, then integer representing the size multiplier of the animal geometry in previous frame. If not None, the animal geometry will only be searched for within this geometry.
    :return: A dataframe shape (N, 4) where N is the number of frames, containing the X and Y coordinates of the centroid of the largest blob in each frame and the vertices representing the hull. If `save_path` is provided, returns None.
    :rtype: Union[None, pd.DataFrame]
    :example:
    >>> x = ImageMixin.get_blob_locations(video_path=r"/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/2022-06-20_NOB_DOT_4_downsampled_bg_subtracted.mp4", gpu=True)
    >>> y = ImageMixin.get_blob_locations(video_path=r"C:\troubleshooting\RAT_NOR\project_folder\videos\2022-06-20_NOB_IOT_1_bg_subtracted.mp4", gpu=True)
    """


    timer = SimbaTimer(start=True)
    video_meta = get_video_meta_data(video_path=video_path)
    _, video_name, _ = get_fn_ext(filepath=video_path)
    check_int(name=f'{get_blob_locations.__name__} batch_size', value=batch_size, min_value=1)
    if batch_size > video_meta['frame_count']: batch_size = video_meta['frame_count']
    if save_path is not None:
        if not os.path.isdir(os.path.dirname(save_path)):
            raise NotDirectoryError(msg=f'Directory path {os.path.dirname(save_path)} is not a valid directory', source=get_blob_locations.__name__)
    check_valid_boolean(value=gpu, source=f'{get_blob_locations.__name__} gpu')
    check_valid_boolean(value=verbose, source=f'{get_blob_locations.__name__} verbose')
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg='No GPU detected, try to set GPU to False', source=get_blob_locations.__name__)
    if inclusion_zone is not None:
        check_instance(source=f'{get_blob_locations} inclusion_zone', instance=inclusion_zone, accepted_types=(MultiPolygon, Polygon,), raise_error=True)
    if window_size is not None:
        check_float(name='window_size', value=window_size, min_value=1.0, raise_error=True)
    frame_ids = list(range(0, video_meta['frame_count']))
    frame_ids = [frame_ids[i:i + batch_size] for i in range(0, len(frame_ids), batch_size)]
    core_cnt = find_core_cnt()[0]
    results = {}
    if verbose:
        print('Starting blob location detection...')
    for frame_batch in range(len(frame_ids)):
        start_frm, end_frm = frame_ids[frame_batch][0], frame_ids[frame_batch][-1]
        if gpu:
            imgs = read_img_batch_from_video_gpu(video_path=video_path, start_frm=start_frm, end_frm=end_frm,  verbose=False, out_format='array', black_and_white=True)
        else:
            imgs = read_img_batch_from_video(video_path=video_path, start_frm=start_frm, end_frm=end_frm, verbose=False, black_and_white=True)
        keys = np.arange(start_frm, end_frm+1)
        img_dict = {}
        for cnt, img_id in enumerate(range(start_frm, end_frm+1)):
            img_dict[img_id] = imgs[cnt]
        batch_results = {}
        img_dict = [{k: img_dict[k] for k in subset} for subset in np.array_split(keys, core_cnt)]
        del imgs
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(find_animal_blob_location,
                                          verbose=verbose,
                                          video_name=video_name,
                                          inclusion_zone=inclusion_zone)
            for cnt, result in enumerate(pool.imap(constants, img_dict, chunksize=1)):
                batch_results.update(result)
        results.update(batch_results)
        pool.join()
        pool.terminate()
    results = dict(sorted(results.items()))
    results = pd.DataFrame.from_dict(data=results, orient='index', columns=['x', 'y', 'vertices'])
    if save_path is not None:
        results.to_csv(save_path)
    else:
        return results




if __name__ == "__main__":
    get_blob_locations(video_path=r"C:\troubleshooting\mitra\test\temp\501_MA142_Gi_Saline_0515.mp4", gpu=False, save_path=r'C:\troubleshooting\mitra\test\blob_data\501_MA142_Gi_Saline_0515.csv')



#
# imgs = read_img_batch_from_video_gpu(video_path=r"C:\troubleshooting\mitra\test\temp\501_MA142_Gi_Saline_0515.mp4", start_frm=0, end_frm=0, black_and_white=True)
# data = find_animal_blob_location(imgs=imgs, window_size=3)
# data = pd.DataFrame.from_dict(data, orient='index')
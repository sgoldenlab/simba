import os
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from shapely.geometry import Polygon

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.checks import (check_if_dir_exists, check_int,
                                check_valid_boolean, check_valid_dict)
from simba.utils.enums import Keys
from simba.utils.errors import InvalidInputError
from simba.utils.read_write import (find_video_of_file, get_video_meta_data,
                                    read_frm_of_video, read_roi_data)


def simba_rois_to_yolo(config_path: Optional[Union[str, os.PathLike]] = None,
                       roi_path: Optional[Union[str, os.PathLike]] = None,
                       video_dir: Optional[Union[str, os.PathLike]] = None,
                       save_dir: Optional[Union[str, os.PathLike]] = None,
                       roi_frm_cnt: Optional[int] = 10,
                       obb: Optional[bool] = False,
                       greyscale: Optional[bool] = True) -> None:

    """
    Converts SimBA roi definitions into annotations and images for training yolo network.

    :param Optional[Union[str, os.PathLike]] config_path: Optional path to the project config file in SimBA project.
    :param Optional[Union[str, os.PathLike]] roi_path: Path to the SimBA roi definitions .h5 file. If None, then the ``roi_coordinates_path`` of the project.
    :param Optional[Union[str, os.PathLike]] video_dir: Directory where to find the videos. If None, then the videos folder of the project.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where to save the labels and images. If None, then the logs folder of the project.
    :param Optional[int] roi_frm_cnt: Number of frames for each video to create bounding boxes for.
    :param Optional[bool] obb: If True, created object-oriented yolo bounding boxes. Else, axis aligned yolo bounding boxes. Default False.
    :param Optional[bool] greyscale: If True, converts the images to greyscale if rgb. Default: True.
    :return: None

    :example I:
    >>> simba_rois_to_yolo(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")

    :example II:
    >>> simba_rois_to_yolo(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini", save_dir=r"C:\troubleshooting\RAT_NOR\project_folder\logs\yolo", video_dir=r"C:\troubleshooting\RAT_NOR\project_folder\videos", roi_path=r"C:\troubleshooting\RAT_NOR\project_folder\logs\measures\ROI_definitions.h5")
    """

    if roi_path is None or video_dir is None or save_dir is None:
        config = ConfigReader(config_path=config_path)
        roi_path = config.roi_coordinates_path
        video_dir = config.video_dir
        save_dir = config.logs_path
    check_int(name=f'{simba_rois_to_yolo.__name__} roi_frm_cnt', value=roi_frm_cnt, min_value=1)
    check_valid_boolean(value=[obb,greyscale], source=f'{simba_rois_to_yolo.__name__} obb')
    roi_data = read_roi_data(roi_path=roi_path)
    roi_geometries = GeometryMixin.simba_roi_to_geometries(rectangles_df=roi_data[0], circles_df=roi_data[1], polygons_df=roi_data[2])[0]
    roi_geometries_rectangles = {}
    roi_ids, roi_cnt = {}, 0
    save_img_dir = os.path.join(save_dir, 'images')
    save_labels_dir = os.path.join(save_dir, 'labels')
    if not os.path.isdir(save_img_dir): os.makedirs(save_img_dir)
    if not os.path.isdir(save_labels_dir): os.makedirs(save_labels_dir)
    for video_name, roi_data in roi_geometries.items():
        roi_geometries_rectangles[video_name] = {}
        for roi_name, roi in roi_data.items():
            if obb:
                roi_geometries_rectangles[video_name][roi_name] = GeometryMixin.minimum_rotated_rectangle(shape=roi)
            else:
                keypoints = np.array(roi.exterior.coords).astype(np.int32).reshape(1, -1, 2)
                roi_geometries_rectangles[video_name][roi_name] = Polygon(GeometryMixin.keypoints_to_axis_aligned_bounding_box(keypoints=keypoints)[0])
            if roi_name not in roi_ids.keys():
                roi_ids[roi_name] = roi_cnt
                roi_cnt += 1

    roi_results = {}
    img_results = {}
    for video_name, roi_data in roi_geometries.items():
        roi_results[video_name] = {}
        img_results[video_name] = []
        video_path = find_video_of_file(video_dir=video_dir, filename=video_name)
        video_meta_data = get_video_meta_data(video_path)
        if roi_frm_cnt > video_meta_data['frame_count']:
            roi_frm_cnt = video_meta_data['frame_count']
        cap = cv2.VideoCapture(video_path)
        frm_idx = np.sort(np.random.choice(np.arange(0, video_meta_data['frame_count']), size=roi_frm_cnt))
        for idx in frm_idx:
            img_results[video_name].append(read_frm_of_video(video_path=cap, frame_index=idx, greyscale=greyscale))
        w, h = video_meta_data['width'], video_meta_data['height']
        for roi_name, roi in roi_data.items():
            roi_id = roi_ids[roi_name]
            if not obb:
                shape_stats = GeometryMixin.get_shape_statistics(shapes=roi)
                x_center = shape_stats['centers'][0][0] / w
                y_center = shape_stats['centers'][0][1] / h
                width = shape_stats['widths'][0] / w
                height = shape_stats['lengths'][0] / h
                roi_str = ' '.join([str(roi_id), str(x_center), str(y_center), str(width), str(height)])
            else:
                img_geometry = np.array(roi.exterior.coords).astype(np.int32)[1:]
                x1, y1 = img_geometry[0][0] / w, img_geometry[0][1] / h
                x2, y2 = img_geometry[1][0] / w, img_geometry[1][1] / h
                x3, y3 = img_geometry[2][0] / w, img_geometry[2][1] / h
                x4, y4 = img_geometry[3][0] / w, img_geometry[3][1] / h
                roi_str = ' '.join([str(roi_id), str(x1), str(y1), str(x2), str(y2), str(x3), str(y3), str(x4), str(y4)])
            roi_results[video_name][roi_name] = roi_str

    for video_name, imgs in img_results.items():
        for img_cnt, img in enumerate(imgs):
            img_save_path = os.path.join(save_img_dir, f'{video_name}_{img_cnt}.png')
            cv2.imwrite(img_save_path, img)
            label_save_path = os.path.join(save_labels_dir, f'{video_name}_{img_cnt}.txt')
            x = list(roi_results[video_name].values())
            with open(label_save_path, mode='wt', encoding='utf-8') as f:
                f.write('\n'.join(x))

simba_rois_to_yolo(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini",
                   save_dir=r"C:\troubleshooting\RAT_NOR\project_folder\logs\yolo",
                   video_dir=r"C:\troubleshooting\RAT_NOR\project_folder\videos",
                   roi_path=r"C:\troubleshooting\RAT_NOR\project_folder\logs\measures\ROI_definitions.h5")
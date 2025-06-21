import os
import random
from datetime import datetime
from typing import Optional, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2
import numpy as np
from shapely.geometry import Polygon

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.third_party_label_appenders.transform.utils import \
    create_yolo_keypoint_yaml
from simba.utils.checks import (check_float, check_if_dir_exists, check_int,
                                check_valid_boolean)
from simba.utils.enums import Formats, Options
from simba.utils.errors import InvalidInputError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (create_directory,
                                    find_files_of_filetypes_in_directory,
                                    find_video_of_file, get_video_meta_data,
                                    read_frm_of_video, read_roi_data)


class SimBAROI2Yolo:

    """
    Converts SimBA roi definitions into annotations and images for training yolo network.

    :param Optional[Union[str, os.PathLike]] config_path: Optional path to the project config file in SimBA project.
    :param Optional[Union[str, os.PathLike]] roi_path: Path to the SimBA roi definitions .h5 file. If None, then the ``roi_coordinates_path`` of the project.
    :param Optional[Union[str, os.PathLike]] video_dir: Directory where to find the videos. If None, then the videos folder of the project.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where to save the labels and images. If None, then the logs folder of the project.
    :param Optional[int] roi_frm_cnt: Number of frames for each video to create bounding boxes for.
    :param float train_size: Proportion of frames randomly assigned to the training dataset. Value must be between 0.1 and 0.99. Default: 0.7.
    :param Optional[bool] obb: If True, created object-oriented yolo bounding boxes. Else, axis aligned yolo bounding boxes. Default False.
    :param Optional[bool] greyscale: If True, converts the images to greyscale if rgb. Default: True.
    :param Optional[bool] verbose: If True, prints progress. Default: False.
    :return: None

    :example I:
    >>> SimBAROI2Yolo(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini").run()

    :example II:
    >>> SimBAROI2Yolo(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini", save_dir=r"C:\troubleshooting\RAT_NOR\project_folder\logs\yolo", video_dir=r"C:\troubleshooting\RAT_NOR\project_folder\videos", roi_path=r"C:\troubleshooting\RAT_NOR\project_folder\logs\measures\ROI_definitions.h5").run()

    :example III:
    >>> SimBAROI2Yolo(video_dir=r"C:\troubleshooting\RAT_NOR\project_folder\videos", roi_path=r"C:\troubleshooting\RAT_NOR\project_folder\logs\measures\ROI_definitions.h5", save_dir=r'C:\troubleshooting\RAT_NOR\project_folder\yolo', verbose=True, roi_frm_cnt=20, obb=True).run()
    """


    def __init__(self,
                 config_path: Optional[Union[str, os.PathLike]] = None,
                 roi_path: Optional[Union[str, os.PathLike]] = None,
                 video_dir: Optional[Union[str, os.PathLike]] = None,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 roi_frm_cnt: int = 10,
                 train_size: float = 0.7,
                 obb: bool = False,
                 greyscale: bool = False,
                 clahe: bool = False,
                 verbose: bool = True):


        if video_dir is not None: check_if_dir_exists(in_dir=video_dir, source=self.__class__.__name__)
        if roi_path is not None: check_if_dir_exists(in_dir=roi_path, source=self.__class__.__name__)
        if save_dir is not None: check_if_dir_exists(in_dir=save_dir, source=self.__class__.__name__)
        if video_dir is None and config_path is None:
            raise InvalidInputError(msg='If not passing config_path, then pass video_dir', source=self.__class__.__name__)
        if save_dir is None and config_path is None:
            raise InvalidInputError(msg='If not passing config_path, then pass save_dir', source=self.__class__.__name__)
        if roi_path is None and config_path is None:
            raise InvalidInputError(msg='If not passing config_path, then pass roi_path', source=self.__class__.__name__)
        self.timer = SimbaTimer(start=True)
        config = None
        if roi_path is None or video_dir is None or save_dir is None:
            config = ConfigReader(config_path=config_path)

        roi_path = roi_path if roi_path is not None else config.roi_coordinates_path
        save_dir = save_dir if save_dir is not None else os.path.join(config.logs_path, f'yolo_{datetime.now().strftime("%Y%m%d%H%M%S")}')
        video_dir = video_dir if video_dir is not None else config.video_dir
        if not os.path.isdir(save_dir): os.makedirs(save_dir)

        check_int(name=f'{self.__class__.__name__} roi_frm_cnt', value=roi_frm_cnt, min_value=1)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale')
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe')
        check_valid_boolean(value=obb, source=f'{self.__class__.__name__} obb')
        check_float(name=f'{self.__class__.__name__} train_size', min_value=10e-6, max_value=0.99, value=train_size, raise_error=True)
        check_if_dir_exists(in_dir=video_dir)
        roi_data = read_roi_data(roi_path=roi_path)
        roi_geometries = GeometryMixin.simba_roi_to_geometries(rectangles_df=roi_data[0], circles_df=roi_data[1], polygons_df=roi_data[2])[0]
        video_files = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True, raise_warning=False, as_dict=True)
        self.sliced_roi_geometries = {k: v for k, v in roi_geometries.items() if k in video_files.keys()}
        if len(self.sliced_roi_geometries.keys()) == 0:
            raise NoFilesFoundError(msg=f'No video files for in {video_dir} directory for the videos represented in the {roi_path} file: {roi_geometries.keys()}',source=self.__class__.__name__)
        self.roi_geometries_rectangles = {}
        self.roi_ids, self.roi_cnt = {}, 0
        self.map_path = os.path.join(save_dir, 'map.yaml')
        self.img_dir, lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
        self.img_train_dir, self.img_val_dir = os.path.join(self.img_dir, 'train'), os.path.join(self.img_dir, 'val')
        self.lbl_train_dir, self.lb_val_dir = os.path.join(lbl_dir, 'train'), os.path.join(lbl_dir, 'val')
        create_directory(paths=[self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lb_val_dir], overwrite=False)
        self.verbose, self.video_dir, self.roi_frm_cnt, self.greyscale = verbose, video_dir, roi_frm_cnt, greyscale
        self.train_size, self.save_dir, self.obb, self.clahe = train_size, save_dir, obb, clahe
        self.video_cnt = len(list(self.sliced_roi_geometries.keys()))

    def run(self):
        self.timer = SimbaTimer(start=True)
        if self.verbose: print('Reading geometries...')
        for video_cnt, (video_name, roi_data) in enumerate(self.sliced_roi_geometries.items()):
            if self.verbose: print(
                f'Reading ROI geometries for video {video_name}... ({video_cnt + 1}/{self.video_cnt})')
            self.roi_geometries_rectangles[video_name] = {}
            for roi_name, roi in roi_data.items():
                if self.obb:
                    self.roi_geometries_rectangles[video_name][roi_name] = GeometryMixin.minimum_rotated_rectangle(shape=roi)
                else:
                    keypoints = np.array(roi.exterior.coords).astype(np.int32).reshape(1, -1, 2)
                    self.roi_geometries_rectangles[video_name][roi_name] = Polygon(
                        GeometryMixin.keypoints_to_axis_aligned_bounding_box(keypoints=keypoints)[0])
                    print(self.roi_geometries_rectangles[video_name][roi_name])
                if roi_name not in self.roi_ids.keys():
                    self.roi_ids[roi_name] = self.roi_cnt
                    self.roi_cnt += 1

        roi_results, img_results = {}, {}
        if self.verbose: print('Reading coordinates ...')
        for video_cnt, (video_name, roi_data) in enumerate(self.roi_geometries_rectangles.items()):
            if self.verbose: print(
                f'Reading ROI coordinates for video {video_name}... ({video_cnt + 1}/{len(list(self.roi_geometries_rectangles.keys()))})')
            roi_results[video_name] = {}
            img_results[video_name] = []
            video_path = find_video_of_file(video_dir=self.video_dir, filename=video_name)
            video_meta_data = get_video_meta_data(video_path)
            if self.roi_frm_cnt > video_meta_data['frame_count']:
                self.roi_frm_cnt = video_meta_data['frame_count']
            cap = cv2.VideoCapture(video_path)
            frm_idx = np.sort(np.random.choice(np.arange(0, video_meta_data['frame_count']), size=self.roi_frm_cnt))
            for idx in frm_idx:
                img_results[video_name].append(read_frm_of_video(video_path=cap, frame_index=idx, greyscale=self.greyscale, clahe=self.clahe))
            w, h = video_meta_data['width'], video_meta_data['height']
            for roi_name, roi in roi_data.items():
                roi_id = self.roi_ids[roi_name]
                if not self.obb:
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
                    roi_str = ' '.join([str(roi_id), str(x1), str(y1), str(x2), str(y2), str(x3), str(y3), str(x4), str(y4), '\n'])
                roi_results[video_name][roi_name] = roi_str

        total_img_cnt = sum(len(v) for v in img_results.values())
        train_idx = random.sample(list(range(0, total_img_cnt)), int(total_img_cnt * self.train_size))

        if self.verbose: print('Reading images ...')
        cnt = 0
        for video_cnt, (video_name, imgs) in enumerate(img_results.items()):
            if self.verbose: print(
                f'Reading ROI images for video {video_name}... ({video_cnt + 1}/{len(list(img_results.keys()))})')
            for img_cnt, img in enumerate(imgs):
                if cnt in train_idx:
                    img_save_path = os.path.join(self.img_train_dir, f'{video_name}_{img_cnt}.png')
                    lbl_save_path = os.path.join(self.lbl_train_dir, f'{video_name}_{img_cnt}.txt')
                else:
                    img_save_path = os.path.join(self.img_val_dir, f'{video_name}_{img_cnt}.png')
                    lbl_save_path = os.path.join(self.lb_val_dir, f'{video_name}_{img_cnt}.txt')
                cv2.imwrite(img_save_path, img)
                # circle = roi_geometries_rectangles[video_name]['circle']
                # pts = np.array(circle.exterior.coords, dtype=np.int32)
                # pts = pts.reshape((-1, 1, 2))
                # cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                # cv2.imshow('sadasdasd', img)
                x = list(roi_results[video_name].values())
                with open(lbl_save_path, mode='wt', encoding='utf-8') as f:
                    f.write(''.join(x))
                cnt += 1

        roi_ids = {v: k for k, v in self.roi_ids.items()}

        create_yolo_keypoint_yaml(path=self.save_dir, train_path=self.img_train_dir, val_path=self.img_val_dir, names=roi_ids, save_path=self.map_path)
        self.timer.stop_timer()
        if self.verbose:
            stdout_success(msg=f'YOLO ROI data saved in {self.save_dir}', elapsed_time=self.timer.elapsed_time_str)


# runner = SimBAROI2Yolo(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")
# runner.run()
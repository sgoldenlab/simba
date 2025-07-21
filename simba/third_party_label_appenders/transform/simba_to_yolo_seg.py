import os
import random
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.third_party_label_appenders.converters import create_yolo_yaml
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int, check_str,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_tuple,
                                check_video_and_data_frm_count_align)
from simba.utils.enums import Formats, Options
from simba.utils.errors import NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (create_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_video_meta_data, read_df,
                                    read_frm_of_video)
from simba.utils.warnings import NoDataFoundWarning


class SimBA2YoloSegmentation(ConfigReader):
    """
    Convert pose estimation data from a SimBA project into the YOLO keypoint format, including frame sampling,
    image-label pair creation, bounding box computation, and train/validation splitting.

    .. note::
       For creating the ``flip_idx``, see :func:`simba.third_party_label_appenders.converters.get_yolo_keypoint_flip_idx`.
       For creating the ``bp_id_idx``, see :func:`simba.third_party_label_appenders.converters.get_yolo_keypoint_bp_id_idx`

    :param Union[str, os.PathLike] config_path: Path to the SimBA project `.ini` configuration file.
    :param Union[str, os.PathLike] save_dir: Directory where YOLO-formatted data will be saved. Subdirectories for images/labels (train/val) are created.
    :param Optional[Union[str, os.PathLike] data_dir: Optional directory containing outlier-corrected SimBA pose estimation data. If None, uses path from config.
    :param float train_size: Proportion of samples to allocate to the training set (range 0.1–0.99). Remaining samples go to validation.
    :param bool verbose: If True, prints progress updates to the console.
    :param bool greyscale: If True, saves extracted video frames in greyscale. Otherwise, saves in color.
    :param float padding: Padding added around the bounding box (as a proportion of image dimensions, range 0.0–1.0). Useful if animal body-parts are in a "line".
    :param Tuple[int, ...] flip_idx: Tuple defining symmetric keypoint indices for horizontal flipping. Used to write the `map.yaml` file. If None, then attempt to infer.
    :param Dict[int, str] names: Dictionary mapping instance IDs to class names. Used in annotation labels and `map.yaml`.
    :param Optional[int] sample_size: If specified, limits the number of randomly sampled frames per video. If None, all frames are used.
    :param Optional[Dict[int, Union[Tuple[int], List[int]]]] bp_id_idx: Optional mapping of instance IDs to keypoint index groups, allowing support for multiple animals per frame. Must match keys in `map_dict`.
    :param Optional[str] single_id: If the data contains pose-estimation for multiple indivisuals, but you want to treat it as examples of a single individual, pass the name of the single individual. Defaults to None, and the YOLO data will be formatted to the number of objects which the H5 data contains.
    :return: None. Saves YOLO-formatted images and annotations to disk in the `save_dir` location.

    :example:
    >>> SAVE_DIR = r'D:\troubleshooting\mitra\mitra_yolo'
    >>> CONFIG_PATH = r"C:\troubleshooting\mitra\project_folder\project_config.ini"
    >>> runner = SimBA2Yolo(config_path=CONFIG_PATH, save_dir=SAVE_DIR, sample_size=10, verbose=True)
    >>> runner.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 train_size: float = 0.7,
                 verbose: bool = False,
                 greyscale: bool = False,
                 clahe: bool = False,
                 padding: int = 0,
                 threshold: float = 0.00,
                 sample_size: Optional[int] = None,
                 single_id: Optional[str] = None) -> None:


        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale')
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe')
        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        if padding is not None: check_int(name=f'{self.__class__.__name__} padding', value=padding, min_value=0, raise_error=True)
        check_float(name=f'{self.__class__.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, max_value=1.0, min_value=0.0)
        check_if_dir_exists(in_dir=save_dir)
        self.img_dir, self.lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
        self.img_train_dir, self.img_val_dir = os.path.join(self.img_dir, 'train'), os.path.join(self.img_dir, 'val')
        self.lbl_train_dir, self.lb_val_dir = os.path.join(self.lbl_dir, 'train'), os.path.join(self.lbl_dir, 'val')
        create_directory(paths=[self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lb_val_dir], overwrite=False)
        self.map_path = os.path.join(save_dir, 'map.yaml')
        if single_id is not None: check_str(name=f'{self.__class__.__name__} single_id', value=single_id, raise_error=True)
        if sample_size is not None: check_int(name=f'{self.__class__.__name__} sample', value=sample_size, min_value=1)
        self.config = ConfigReader(config_path=config_path)
        if data_dir is not None:
            check_if_dir_exists(in_dir=data_dir, source=f'{self.__class__.__name__} data_dir')
        else:
            data_dir = self.outlier_corrected_dir
        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=[f'.{self.file_type}'], raise_error=True, as_dict=True)
        self.video_paths = find_files_of_filetypes_in_directory(directory=self.video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True, as_dict=True)
        missing_videos = [x for x in self.data_paths.keys() if x not in self.video_paths.keys()]
        if len(missing_videos) > 0:
            NoDataFoundWarning(msg=f'Data files {missing_videos} do not have corresponding videos in the {self.video_dir} directory', source=self.__class__.__name__)
        self.data_w_video = [x for x in self.data_paths.keys() if x in self.video_paths.keys()]
        if len(self.data_w_video) == 0:
            raise NoFilesFoundError(msg=f'None of the data files in {data_dir} have matching videos in the {self.video_dir} directory', source=self.__class__.__name__)
        self.sample_size, self.train_size, self.verbose, self.save_dir = sample_size, train_size, verbose, save_dir
        self.greyscale, self.padding = greyscale, padding
        self.clahe, self.threshold, self.single_id = clahe, threshold, single_id
        if self.single_id is None:
            self.map_ids = {v: k for k, v in enumerate(self.animal_bp_dict.keys())}
        else:
            self.map_ids = {self.single_id: '0'}

    def run(self):
        timer, annotations = SimbaTimer(start=True), []
        for file_cnt, video_name in enumerate(self.data_w_video):
            data = read_df(file_path=self.data_paths[video_name], file_type=self.config.file_type)
            check_valid_dataframe(df=data, source=f'{self.__class__.__name__} {self.data_paths[video_name]}', valid_dtypes=Formats.NUMERIC_DTYPES.value)
            video_path = self.video_paths[video_name]
            check_video_and_data_frm_count_align(video=video_path, data=data, name=self.data_paths[video_name], raise_error=True)
            p_data = data[data.columns[list(data.columns.str.endswith('_p'))]]
            data = data.loc[:, ~data.columns.str.endswith('_p')].reset_index(drop=True)
            data = data.iloc[(p_data[(p_data > self.threshold).all(axis=1)].index)]
            data['video'], frm_cnt = video_name, len(data)
            if self.sample_size is None:
                video_sample_idx = list(range(0, frm_cnt))
            else:
                video_sample_idx = list(range(0, frm_cnt)) if self.sample_size > frm_cnt else random.sample(list(range(0, frm_cnt)), self.sample_size)
            annotations.append(data.iloc[video_sample_idx].reset_index(drop=False))

        annotations = pd.concat(annotations, axis=0).reset_index(drop=True)
        video_names = annotations.pop('video').reset_index(drop=True).values
        train_idx = random.sample(list(annotations['index']), int(len(annotations) * self.train_size))


        for cnt, (idx, idx_data) in enumerate(annotations.iterrows()):
            vid_path = self.video_paths[video_names[cnt]]
            video_meta = get_video_meta_data(video_path=vid_path)
            if self.verbose:
                print(f'Processing annotation {cnt + 1}/{len(annotations)} from SimBA to YOLO segmentation.. ({video_meta["video_name"]})...')
            img = read_frm_of_video(video_path=vid_path, frame_index=idx_data['index'], greyscale=self.greyscale, clahe=self.clahe)
            img_lbl = ''
            img_name = f'{video_meta["video_name"]}.{idx}'
            if idx_data['index'] in train_idx:
                label_save_path, img_save_path = os.path.join(self.lbl_train_dir, f'{img_name}.txt'), os.path.join(self.img_train_dir, f'{img_name}.png')
            else:
                label_save_path, img_save_path = os.path.join(self.lb_val_dir, f'{img_name}.txt'), os.path.join(self.img_val_dir, f'{img_name}.png')
            for animal_cnt, (animal_name, bps_data) in enumerate(self.animal_bp_dict.items()):
                cols = [item for group in zip(bps_data['X_bps'], bps_data['Y_bps']) for item in group]
                keypoints = annotations.loc[idx, cols].values.reshape(-1, 2).astype(np.int32)
                if self.padding is not None and self.padding > 0:
                    keypoints = keypoints.reshape(-1, keypoints.shape[0], 2).astype(np.int32)
                    keypoints = GeometryMixin().bodyparts_to_polygon(data=keypoints, parallel_offset=self.padding)[0]
                    keypoints = np.array(keypoints.exterior.coords).astype(np.int32)
                mask = ~(np.all(keypoints == 0, axis=1) | np.all(keypoints < 0, axis=1) | np.any(~np.isfinite(keypoints), axis=1))
                keypoints = keypoints[mask]
                if keypoints.shape[0] > 2:
                    instance_str = f'{animal_cnt} ' if self.single_id is None else '0 '
                    pts = np.unique(keypoints, axis=0)
                    center = np.mean(pts, axis=0)
                    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
                    seg_arr = pts[np.argsort(angles)]
                    seg_arr_x, seg_arr_y = np.clip(seg_arr[:, 0].flatten() / video_meta['width'], 0, 1), np.clip(seg_arr[:, 1].flatten() / video_meta['height'], 0, 1)
                    kps = list(np.column_stack((seg_arr_x, seg_arr_y)).flatten())
                    instance_str += ' '.join(str(x) for x in kps).strip() + '\n'
                    img_lbl += instance_str
            if img_lbl:
                with open(label_save_path, mode='wt', encoding='utf-8') as f:
                    f.write(img_lbl)
                cv2.imwrite(img_save_path, img)

        create_yolo_yaml(path=self.save_dir, train_path=self.img_train_dir, val_path=self.img_val_dir, names=self.map_ids, save_path=self.map_path)
        timer.stop_timer()
        if self.verbose: stdout_success(msg=f'Labelme to YOLO conversion complete. Data saved in directory {self.save_dir}.', elapsed_time=timer.elapsed_time_str)

# SAVE_DIR = r'D:\troubleshooting\mitra\mitra_yolo_seg'
# CONFIG_PATH = r"C:\troubleshooting\mitra\project_folder\project_config.ini"
# runner = SimBA2YoloSegmentation(config_path=CONFIG_PATH, save_dir=SAVE_DIR, sample_size=250, verbose=True, padding=None)
# runner.run()
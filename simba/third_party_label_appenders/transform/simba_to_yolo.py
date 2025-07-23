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
from simba.third_party_label_appenders.transform.utils import (
    create_yolo_keypoint_yaml, get_yolo_keypoint_flip_idx)
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
from simba.utils.yolo import keypoint_array_to_yolo_annotation_str


class SimBA2Yolo:
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
                 padding: float = 0.00,
                 threshold: float = 0.00,
                 flip_idx: Optional[Tuple[int, ...]] = None,
                 names: Tuple[str, ...] = ('animal_1',),
                 sample_size: Optional[int] = None,
                 bp_id_idx: Optional[Dict[int, Union[Tuple[int], List[int]]]] = None,
                 single_id: Optional[str] = None) -> None:

        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale')
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe')
        check_file_exist_and_readable(file_path=config_path)
        check_float(name=f'{self.__class__.__name__} padding', value=padding, max_value=1.0, min_value=0.0, raise_error=True)
        check_float(name=f'{self.__class__.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, max_value=1.0, min_value=0.0)
        check_valid_tuple(x=names, source=f'{self.__class__.__name__} names', valid_dtypes=(str,), minimum_length=1)
        check_if_dir_exists(in_dir=save_dir)
        if flip_idx is not None: check_valid_tuple(x=flip_idx, source=self.__class__.__name__, valid_dtypes=(int,), minimum_length=1)
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
            data_dir = self.config.outlier_corrected_dir
        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=[f'.{self.config.file_type}'], raise_error=True, as_dict=True)
        self.video_paths = find_files_of_filetypes_in_directory(directory=self.config.video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True, as_dict=True)
        missing_videos = [x for x in self.data_paths.keys() if x not in self.video_paths.keys()]
        if len(missing_videos) > 0:
            NoDataFoundWarning(msg=f'Data files {missing_videos} do not have corresponding videos in the {self.config.video_dir} directory', source=self.__class__.__name__)
        self.data_w_video = [x for x in self.data_paths.keys() if x in self.video_paths.keys()]
        if len(self.data_w_video) == 0:
            raise NoFilesFoundError(msg=f'None of the data files in {data_dir} have matching videos in the {self.config.video_dir} directory', source=self.__class__.__name__)
        self.sample_size, self.train_size, self.verbose, self.save_dir = sample_size, train_size, verbose, save_dir
        self.greyscale, self.bp_id_idx, self.padding, self.flip_idx = greyscale, bp_id_idx, padding, flip_idx
        self.clahe, self.threshold, self.single_id = clahe, threshold, single_id
        self.names = {0: self.single_id} if self.single_id is not None else {k:v for k, v in enumerate(names)}


    def run(self):
        annotations, timer, body_part_headers = [], SimbaTimer(start=True), []
        for file_cnt, video_name in enumerate(self.data_w_video):
            data = read_df(file_path=self.data_paths[video_name], file_type=self.config.file_type)
            check_valid_dataframe(df=data, source=f'{self.__class__.__name__} {self.data_paths[video_name]}', valid_dtypes=Formats.NUMERIC_DTYPES.value)
            video_path = self.video_paths[video_name]
            check_video_and_data_frm_count_align(video=video_path, data=data, name=self.data_paths[video_name], raise_error=True)
            p_data = data[data.columns[list(data.columns.str.endswith('_p'))]]
            data = data.loc[:, ~data.columns.str.endswith('_p')].reset_index(drop=True)
            data = data.iloc[(p_data[(p_data > self.threshold).all(axis=1)].index)]
            body_part_headers = data.columns
            data['video'], frm_cnt = video_name, len(data)
            if self.sample_size is None:
                video_sample_idx = list(range(0, frm_cnt))
            else:
                video_sample_idx = list(range(0, frm_cnt)) if self.sample_size > frm_cnt else random.sample(list(range(0, frm_cnt)), self.sample_size)
            annotations.append(data.iloc[video_sample_idx].reset_index(drop=False))

        if self.flip_idx is None:
            self.flip_idx = get_yolo_keypoint_flip_idx(x=list(dict.fromkeys([x[:-2] for x in body_part_headers])))

        annotations = pd.concat(annotations, axis=0).reset_index(drop=True)
        video_names = annotations.pop('video').reset_index(drop=True).values
        train_idx = random.sample(list(annotations['index']), int(len(annotations) * self.train_size))
        bp_id_idx = np.array_split(np.array(range(0, int(len(body_part_headers) / 2))), len(self.names.keys()))
        bp_id_idx = [list(x) for x in bp_id_idx]

        for cnt, (idx, idx_data) in enumerate(annotations.iterrows()):
            vid_path = self.video_paths[video_names[cnt]]
            video_meta = get_video_meta_data(video_path=vid_path)
            frm_idx, keypoints = idx_data[0], idx_data.values[1:].reshape(-1, 2)
            mask = (keypoints[:, 0] == 0.0) & (keypoints[:, 1] == 0.0)
            keypoints[mask] = np.nan
            if np.all(np.isnan(keypoints)) or np.all(keypoints == 0.0) or np.all(np.isnan(keypoints) | (keypoints == 0.0)):
                continue
            img_lbl = ''
            if self.verbose:
                print(f'Processing image {cnt + 1}/{len(annotations)}...')
            file_name = f'{video_meta["video_name"]}.{frm_idx}'
            if frm_idx in train_idx:
                img_save_path, lbl_save_path = os.path.join(self.img_train_dir, f'{file_name}.png'), os.path.join(self.lbl_train_dir, f'{file_name}.txt')
            else:
                img_save_path, lbl_save_path = os.path.join(self.img_train_dir, f'{file_name}.png'), os.path.join(self.lb_val_dir, f'{file_name}.txt')
            img = read_frm_of_video(video_path=vid_path, frame_index=frm_idx, greyscale=self.greyscale, clahe=self.clahe)
            img_h, img_w = img.shape[0], img.shape[1]
            keypoints_with_id = {}
            for k, idx in enumerate(bp_id_idx):
                keypoints_with_id[k] = keypoints[idx, :]
            for id, keypoints in keypoints_with_id.items():
                if np.all(np.isnan(keypoints)) or np.all(keypoints == 0.0) or np.all(np.isnan(keypoints) | (keypoints == 0.0)):
                    continue
                visability_col = np.full((keypoints.shape[0], 1), fill_value=2).flatten()
                keypoints = np.insert(keypoints, 2, visability_col, axis=1)
                both_zero = (keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)
                has_nan_or_inf = ~np.isfinite(keypoints[:, 0]) | ~np.isfinite(keypoints[:, 1])
                mask = both_zero | has_nan_or_inf
                keypoints[mask, 2] = 0
                instance_str = f'{id} ' if self.single_id is None else '0 '
                instance_str += keypoint_array_to_yolo_annotation_str(x=keypoints, img_w=img_w, img_h=img_h, padding=self.padding)
                img_lbl += instance_str.strip() + '\n'
                with open(lbl_save_path, mode='wt', encoding='utf-8') as f:
                    f.write(img_lbl)
                cv2.imwrite(img_save_path, img)

        create_yolo_keypoint_yaml(path=self.save_dir, train_path=self.img_train_dir, val_path=self.img_val_dir, names=self.names, save_path=self.map_path, kpt_shape=(len(self.flip_idx), 3), flip_idx=self.flip_idx)
        timer.stop_timer()
        stdout_success(msg=f'YOLO formated data saved in {self.save_dir} directory', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)


# SAVE_DIR = r'D:\troubleshooting\mitra\mitra_yolo'
# CONFIG_PATH = r"C:\troubleshooting\mitra\project_folder\project_config.ini"
# runner = SimBA2Yolo(config_path=CONFIG_PATH, save_dir=SAVE_DIR, sample_size=10, verbose=True, names=('animal_1',))
# runner.run()
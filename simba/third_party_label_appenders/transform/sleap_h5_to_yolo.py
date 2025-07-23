import os
import random
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.third_party_label_appenders.converters import \
    create_yolo_keypoint_yaml
from simba.utils.checks import (check_float, check_if_dir_exists, check_int,
                                check_str, check_valid_boolean,
                                check_valid_tuple)
from simba.utils.enums import Options
from simba.utils.errors import InvalidInputError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (create_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_video_meta_data, read_frm_of_video,
                                    read_sleap_h5)
from simba.utils.warnings import FrameRangeWarning
from simba.utils.yolo import keypoint_array_to_yolo_annotation_str


class SleapH52Yolo:
    """
    Convert SLEAP `.h5` pose estimation annotations to YOLO keypoint annotation format.

    Reads SLEAP `.h5` files and associated videos, samples frames based on a confidence
    threshold, extracts keypoints for one or more animals, and saves image-label pairs in a format
    compatible with YOLOv8 keypoint training.

    :param Union[str, os.PathLike] data_dir: Directory containing SLEAP `.h5` files.
    :param Union[str, os.PathLike] video_dir: Directory containing the videos associated with `.h5` files.
    :param Union[str, os.PathLike] save_dir: Directory to save YOLO-formatted images, labels, and metadata.
    :param Optional[int] frms_cnt: Number of frames to sample per video. If `None`, all valid frames are used.
    :param bool verbose: If True, print progress during processing.
    :param float threshold: Likelihood threshold below which poses are discarded.
    :param float train_size: Proportion of frames to assign to the training set (rest go to validation).
    :param Tuple[int, ...] flip_idx: Tuple indicating how to flip body-parts for augmentation. Length must match keypoint count.
    :param int animal_cnt: Number of animals tracked per frame.
    :param bool greyscale: If True, convert images to grayscale.
    :param bool clahe: If True, apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    :param float padding: Relative padding to apply around the bounding box of keypoints (range 0.0 to 1.0).
    :param Optional[str] single_id: Optional custom ID to assign all annotations the same class (used in single-animal datasets).


    :example:
    >>>DATA_DIR = r'D:\ares\data\termite_1\data'
    >>>VIDEO_DIR = r'D:\ares\data\termite_1\video'
    >>>SAVE_DIR = r"D:\imgs\sleap_h5"
    >>>runner = SleapH52Yolo(data_dir=DATA_DIR, video_dir=VIDEO_DIR, save_dir=SAVE_DIR, threshold=0.9, frms_cnt=50, single_id='termite')
    >>>runner.run()
    """

    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 video_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 frms_cnt: Optional[int] = None,
                 verbose: bool = True,
                 threshold: float = 0,
                 train_size: float = 0.7,
                 flip_idx: Tuple[int, ...] = None,
                 animal_cnt: int = 2,
                 greyscale: bool = False,
                 clahe: bool = False,
                 padding: float = 0.00,
                 single_id: Optional[str] = None):

        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.H5', '.h5'], as_dict=True, raise_error=True)
        self.video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, as_dict=True, raise_error=True)
        missing_video_paths = [x for x in self.video_paths.keys() if x not in self.data_paths.keys()]
        missing_data_paths = [x for x in self.data_paths.keys() if x not in self.video_paths.keys()]
        check_if_dir_exists(in_dir=save_dir)
        self.img_dir, self.lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
        self.img_train_dir, self.img_val_dir = os.path.join(save_dir, 'images', 'train'), os.path.join(save_dir, 'images', 'val')
        self.lbl_train_dir, self.lb_val_dir = os.path.join(save_dir, 'labels', 'train'), os.path.join(save_dir, 'labels', 'val')
        if flip_idx is not None: check_valid_tuple(x=flip_idx, source=f'{self.__class__.__name__} flip_idx', valid_dtypes=(int,), minimum_length=1)
        if single_id is not None: check_str(name=f'{self.__class__.__name__} single_id', value=single_id, raise_error=True)
        check_int(name=f'{self.__class__.__name__} animal_cnt', value=animal_cnt, min_value=1)
        create_directory(paths=[self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lb_val_dir])
        check_float(name=f'{self.__class__.__name__} instance_threshold', min_value=0.0, max_value=1.0,  raise_error=True, value=threshold)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale', raise_error=True)
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe', raise_error=True)
        check_float(name=f'{self.__class__.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
        check_float(name=f'{self.__class__.__name__} padding', value=padding, max_value=1.0, min_value=0.0, raise_error=True)
        self.map_path = os.path.join(save_dir, 'map.yaml')
        if frms_cnt is not None:
            check_int(name=f'{self.__class__.__name__} frms_cnt', value=frms_cnt, min_value=1, raise_error=True)
        if len(missing_video_paths) > 0:
            raise NoFilesFoundError(msg=f'Video(s) {missing_video_paths} could not be found in {video_dir} directory', source=self.__class__.__name__)
        if len(missing_data_paths) > 0:
            raise NoFilesFoundError(msg=f'CSV data for {missing_data_paths} could not be found in {data_dir} directory', source=self.__class__.__name__)
        self.map_dict = {v: f'animal_{k + 1}' for k, v in enumerate(range(animal_cnt))} if single_id is None else {0: single_id}
        self.threshold, self.frms_cnt, self.clahe, self.greyscale = threshold, frms_cnt, clahe, greyscale
        self.train_size, self.verbose, self.animal_cnt = train_size, verbose, animal_cnt
        self.padding, self.save_dir, self.flip_idx, self.single_id = padding, save_dir, flip_idx, single_id


    def run(self):
        dfs = []
        for file_cnt, (file_name, file_path) in enumerate(self.data_paths.items()):
            df = read_sleap_h5(file_path=file_path)
            p_cols = df.iloc[:, 2::3]
            df = df.iloc[p_cols[df.gt(self.threshold).all(axis=1)].index]
            df['frm_idx'] = df.index
            selected_frms = random.sample(list(df['frm_idx']), self.frms_cnt) if self.frms_cnt is not None else list(
                df['frm_idx'].unique())
            df = df[df['frm_idx'].isin(selected_frms)]
            df['video'] = self.video_paths[file_name]
            dfs.append(df)

        dfs, timer = pd.concat(dfs, axis=0), SimbaTimer(start=True)
        dfs['id'] = dfs['frm_idx'].astype(str) + dfs['video'].astype(str)
        train_idx = random.sample(list(dfs['id'].unique()), int(len(dfs['frm_idx'].unique()) * self.train_size))
        for frm_cnt, frm_id in enumerate(dfs['id'].unique()):
            frm_data = dfs[dfs['id'] == frm_id]
            video_path = list(frm_data['video'])[0]
            frm_idx = list(frm_data['frm_idx'])[0]
            video_meta = get_video_meta_data(video_path=video_path)
            if self.verbose:
                print(f'Processing frame: {frm_cnt + 1}/{len(dfs)} ...')
            if frm_idx > video_meta['frame_count']:
                FrameRangeWarning(msg=f'Frame {frm_idx} could not be read from video {video_path}. The video {video_meta["video_name"]} has {video_meta["frame_count"]} frames', source=self.__class__.__name__)
                continue
            img = read_frm_of_video(video_path=video_path, frame_index=frm_idx, greyscale=self.greyscale, clahe=self.clahe)
            img_h, img_w = img.shape[0], img.shape[1]
            if list(frm_data['id'])[0] in train_idx:
                img_save_path = os.path.join(self.img_train_dir, f'{video_meta["video_name"]}_{frm_idx}.png')
                lbl_save_path = os.path.join(self.lbl_train_dir, f'{video_meta["video_name"]}_{frm_idx}.txt')
            else:
                img_save_path = os.path.join(self.img_val_dir, f'{video_meta["video_name"]}_{frm_idx}.png')
                lbl_save_path = os.path.join(self.lb_val_dir, f'{video_meta["video_name"]}_{frm_idx}.txt')
            img_lbl = ''
            frm_data = frm_data.drop(['video', 'id', 'frm_idx'], axis=1).T.iloc[:, 0]
            animal_idxs = np.array_split(list(range(0, len(frm_data))), self.animal_cnt)
            for track_id, animal_idx in enumerate(animal_idxs):
                keypoints = frm_data[frm_data.index.isin(animal_idx)].values.reshape(-1, 3)
                keypoints[keypoints[:, 2] != 0.0, 2] = 2
                if frm_cnt == 0 and track_id == 0:
                    if self.flip_idx is not None and keypoints.shape[0] != len(self.flip_idx):
                        raise InvalidInputError(msg=f'The SLEAP data contains data for {keypoints.shape[0]} body-parts, but passed flip_idx suggests {len(self.flip_idx)} body-parts', source=self.__class__.__name__)
                    elif self.flip_idx is None:
                        self.flip_idx = tuple(list(range(0, keypoints.shape[0])))
                instance_str = f'{track_id} ' if self.single_id is None else f'0 '
                instance_str += keypoint_array_to_yolo_annotation_str(x=keypoints, img_w=img_w, img_h=img_h, padding=self.padding)
                img_lbl += instance_str
            with open(lbl_save_path, mode='wt', encoding='utf-8') as f:
                f.write(img_lbl)
            cv2.imwrite(img_save_path, img)
        create_yolo_keypoint_yaml(path=self.save_dir, train_path=self.img_train_dir, val_path=self.img_val_dir, names=self.map_dict, save_path=self.map_path, kpt_shape=(len(self.flip_idx), 3), flip_idx=self.flip_idx)
        timer.stop_timer()
        stdout_success(msg=f'YOLO formated data saved in {self.save_dir} directory', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)


# DATA_DIR = r'D:\ares\data\termite_1\data'
# VIDEO_DIR = r'D:\ares\data\termite_1\video'
# SAVE_DIR = r"D:\imgs\sleap_h5"
#
# runner = SleapH52Yolo(data_dir=DATA_DIR, video_dir=VIDEO_DIR, save_dir=SAVE_DIR, threshold=0.9, frms_cnt=50, single_id='termite')
# runner.run()
#df = read_sleap_h5(file_path=r"D:\ares\data\termite_1\termite.h5")


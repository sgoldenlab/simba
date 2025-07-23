import os
import random
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2

from simba.third_party_label_appenders.transform.utils import (
    create_yolo_keypoint_yaml, get_yolo_keypoint_flip_idx)
from simba.utils.checks import (check_float, check_if_dir_exists, check_int,
                                check_str, check_valid_boolean,
                                check_valid_dataframe, check_valid_tuple)
from simba.utils.enums import Options
from simba.utils.errors import NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (create_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_video_meta_data, read_frm_of_video)
from simba.utils.yolo import keypoint_array_to_yolo_annotation_str


class Sleap2Yolo:
    """
    Convert SLEAP pose estimation CSV data and corresponding videos into YOLO keypoint dataset format.

    .. note::
       This converts SLEAP **inference** data to YOLO keypoints (not SLEAP annotations).

    :param Union[str, os.PathLike] data_dir: Directory path containing SLEAP-generated CSV files with inferred keypoints.
    :param Union[str, os.PathLike] video_dir: Directory path containing corresponding videos from which frames are to be extracted.
    :param Union[str, os.PathLike] save_dir: Output directory where YOLO-formatted images, labels, and map YAML file will be saved. Subdirectories `images/train`, `images/val`, `labels/train`, `labels/val` will be created.
    :param Optional[int] frms_cnt: Number of frames to randomly sample from each video for conversion. If None, all frames are used.
    :param float instance_threshold: Minimum confidence score threshold to filter out low-confidence pose instances. Only instances with `instance.score` >= this threshold are used.
    :param float train_size: Proportion of frames randomly assigned to the training dataset. Value must be between 0.1 and 0.99. Default: 0.7.
    :param bool verbose: If True, prints progress. Default: True.
    :param Tuple[int, ...] flip_idx: Tuple of keypoint indices used for horizontal flip augmentation during training. The tuple defines the order of keypoints after flipping.
    :param Dict[str, int] map_dict: Dictionary mapping class indices to class names. Used for creating the YAML class names mapping file.
    :param float padding: Fractional padding to add around the bounding boxes (relative to image dimensions). Helps to slightly enlarge bounding boxes by this percentage. Default 0.05. E.g., Useful when all body-parts are along animal length.
    :param Optional[str] single_id: If the data contains pose-estimation for multiple individuals, but you want to treat it as examples of a single individual, pass the name of the single individual. Defaults to None, and the YOLO data will be formatted to the number of objects which the H5 data contains.

    :return: None. Results saved in ``save_dir``.

    :example:
    >>> DATA_DIR = r'D:\ares\data\ant\sleap_csv'
    >>> VIDEO_DIR = r'D:\ares\data\ant\sleap_video'
    >>> SAVE_DIR = r"D:\imgs\sleap_csv"
    >>> runner = Sleap2Yolo(data_dir=DATA_DIR, video_dir=VIDEO_DIR, frms_cnt=50, train_size=0.8, instance_threshold=0.9, save_dir=SAVE_DIR, single_id='ant')
    >>> runner.run()

    """
    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 video_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 frms_cnt: Optional[int] = None,
                 verbose: bool = True,
                 instance_threshold: float = 0,
                 train_size: float = 0.7,
                 flip_idx: Optional[Tuple[int, ...]] = None,
                 names: Optional[Tuple[str, ...]] = None,
                 greyscale: bool = False,
                 clahe: bool = False,
                 padding: float = 0.00,
                 single_id: Optional[str] = None):

        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'], as_dict=True, raise_error=True)
        self.video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, as_dict=True, raise_error=True)
        missing_video_paths = [x for x in self.video_paths.keys() if x not in self.data_paths.keys()]
        missing_data_paths = [x for x in self.data_paths.keys() if x not in self.video_paths.keys()]
        check_if_dir_exists(in_dir=save_dir)
        self.img_dir, self.lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
        self.img_train_dir, self.img_val_dir = os.path.join(save_dir, 'images', 'train'), os.path.join(save_dir, 'images', 'val')
        self.lbl_train_dir, self.lb_val_dir = os.path.join(save_dir, 'labels', 'train'), os.path.join(save_dir, 'labels', 'val')
        if flip_idx is not None: check_valid_tuple(x=flip_idx, source=f'{self.__class__.__name__} flip_idx', valid_dtypes=(int,), minimum_length=1)
        if names is not None: check_valid_tuple(x=names, source=f'{self.__class__.__name__} names', valid_dtypes=(str,), minimum_length=1)
        if single_id is not None: check_str(name=f'{self.__class__.__name__} single_id', value=single_id, raise_error=True)
        create_directory(paths=[self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lb_val_dir], overwrite=True)
        check_float(name=f'{self.__class__.__name__} instance_threshold', min_value=0.0, max_value=1.0, raise_error=True, value=instance_threshold)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale', raise_error=True)
        check_float(name=f'{self.__class__.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
        check_float(name=f'{self.__class__.__name__} padding', value=padding, max_value=1.0, min_value=0.0, raise_error=True)

        self.map_path = os.path.join(save_dir, 'map.yaml')
        if frms_cnt is not None:
            check_int(name=f'{self.__class__.__name__} frms_cnt', value=frms_cnt, min_value=1, raise_error=True)
        if len(missing_video_paths) > 0:
            raise NoFilesFoundError(msg=f'Video(s) {missing_video_paths} could not be found in {video_dir} directory', source=self.__class__.__name__)
        if len(missing_data_paths) > 0:
            raise NoFilesFoundError(msg=f'CSV data for {missing_data_paths} could not be found in {data_dir} directory', source=self.__class__.__name__)
        self.verbose, self.instance_threshold, self.frms_cnt = verbose, instance_threshold, frms_cnt
        self.names, self.greyscale, self.train_size, self.clahe = names, greyscale, train_size, clahe
        self.padding, self.flip_idx, self.save_dir, self.single_id = padding, flip_idx, save_dir, single_id



    def run(self):
        dfs, timer, bp_cols = [], SimbaTimer(start=True), []
        for file_cnt, (file_name, file_path) in enumerate(self.data_paths.items()):
            df = pd.read_csv(filepath_or_buffer=file_path)
            check_valid_dataframe(df=df, source=self.__class__.__name__, required_fields=['track', 'frame_idx', 'instance.score'])
            df = df if self.instance_threshold is None else df[df['instance.score'] >= self.instance_threshold]
            cord_cols, frame_idx = df.drop(['track', 'frame_idx', 'instance.score'], axis=1), df['frame_idx']
            bp_cols = [x for x in cord_cols if not '.score' in x]
            selected_frms = random.sample(list(frame_idx.unique()), self.frms_cnt) if self.frms_cnt is not None else list(frame_idx.unique())
            df = df[df['frame_idx'].isin(selected_frms)]
            df['video'] = self.video_paths[file_name]
            dfs.append(df)

        dfs = pd.concat(dfs, axis=0)
        unique_tracks_lk = {v: k for k, v in enumerate(dfs['track'].unique())}
        if self.names is not None:
            check_valid_tuple(x=self.names, source=f'{self.__class__.__name__} names', valid_dtypes=(str,), accepted_lengths=(len(list(unique_tracks_lk.keys())),))
        else:
            self.names = tuple([f'animal_{k + 1}' for k in range(len(list(unique_tracks_lk.keys())), )])
        map_dict = {k: v for k, v in enumerate(self.names)} if self.single_id is None else {0: self.single_id}
        dfs['id'] = dfs['frame_idx'].astype(str) + dfs['video'].astype(str)
        train_idx = random.sample(list(dfs['id'].unique()), int(len(dfs['frame_idx'].unique()) * self.train_size))
        if self.flip_idx is None:
            self.flip_idx = get_yolo_keypoint_flip_idx(x=list(dict.fromkeys([x[:-2] for x in bp_cols])))

        for frm_cnt, frm_id in enumerate(dfs['id'].unique()):
            frm_data = dfs[dfs['id'] == frm_id]
            video_path = list(frm_data['video'])[0]
            frm_idx = list(frm_data['frame_idx'])[0]
            video_meta = get_video_meta_data(video_path=video_path)
            if self.verbose:
                print(f'Processing frame: {frm_cnt + 1}/{len(list(dfs["id"].unique()))} ...')
            img = read_frm_of_video(video_path=video_path, frame_index=frm_idx, greyscale=self.greyscale, clahe=self.clahe)
            img_h, img_w = img.shape[0], img.shape[1]
            if list(frm_data['id'])[0] in train_idx:
                img_save_path = os.path.join(self.img_train_dir, f'{video_meta["video_name"]}_{frm_idx}.png')
                lbl_save_path = os.path.join(self.lbl_train_dir, f'{video_meta["video_name"]}_{frm_idx}.txt')
            else:
                img_save_path = os.path.join(self.img_val_dir, f'{video_meta["video_name"]}_{frm_idx}.png')
                lbl_save_path = os.path.join(self.lb_val_dir, f'{video_meta["video_name"]}_{frm_idx}.txt')
            img_lbl = ''
            for track_cnt, (_, track_data) in enumerate(frm_data.iterrows()):
                track_id, keypoints = unique_tracks_lk[track_data['track']], track_data.drop(['track', 'frame_idx', 'instance.score', 'video', 'id']),
                keypoints = keypoints.values.reshape(-1, 3).astype(np.float32)
                keypoints[keypoints[:, 2] != 0.0, 2] = 2
                instance_str = f'{track_id} ' if self.single_id is None else f'0 '
                instance_str += keypoint_array_to_yolo_annotation_str(x=keypoints, img_h=img_h, img_w=img_w, padding=self.padding)
                img_lbl += instance_str
            with open(lbl_save_path, mode='wt', encoding='utf-8') as f:
                f.write(img_lbl)
            cv2.imwrite(img_save_path, img)

        create_yolo_keypoint_yaml(path=self.save_dir, train_path=self.img_train_dir, val_path=self.img_val_dir, names=map_dict, save_path=self.map_path, kpt_shape=(len(self.flip_idx), 3), flip_idx=tuple(self.flip_idx))
        timer.stop_timer()
        stdout_success(msg=f'YOLO formated data saved in {self.save_dir} directory', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)


# DATA_DIR = r'D:\ares\data\ant\sleap_csv'
# VIDEO_DIR = r'D:\ares\data\ant\sleap_video'
# SAVE_DIR = r"D:\imgs\sleap_csv"
#
# runner = Sleap2Yolo(data_dir=DATA_DIR, video_dir=VIDEO_DIR, frms_cnt=50, train_size=0.8, instance_threshold=0.9, save_dir=SAVE_DIR, single_id='ant')
# runner.run()
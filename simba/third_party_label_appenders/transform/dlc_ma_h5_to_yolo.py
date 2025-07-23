import os
import random
from typing import Optional, Tuple, Union

import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2
import numpy as np

from simba.mixins.pose_importer_mixin import PoseImporterMixin
from simba.third_party_label_appenders.transform.utils import (
    create_yolo_keypoint_yaml, get_yolo_keypoint_flip_idx)
from simba.utils.checks import (check_float, check_if_dir_exists, check_int,
                                check_str, check_valid_boolean,
                                check_valid_tuple)
from simba.utils.enums import Formats, Options
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (create_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_video_meta_data, read_frm_of_video)
from simba.utils.warnings import FrameRangeWarning
from simba.utils.yolo import keypoint_array_to_yolo_annotation_str

H5_EXT = ['.H5', '.h5']

class MADLCH52Yolo:
    """
    Convert multi-animal DeepLabCut pose estimation H5 data and corresponding videos into YOLO keypoint dataset format.

    .. note::
       This converts DeepLabCut **inference** data to YOLO keypoints (not DeepLabcut annotations).

    :param Union[str, os.PathLike] data_dir: Directory path containing DLC-generated H5 files with inferred keypoints.
    :param Union[str, os.PathLike] video_dir: Directory path containing corresponding videos from which frames are to be extracted.
    :param Union[str, os.PathLike] save_dir: Output directory where YOLO-formatted images, labels, and map YAML file will be saved. Subdirectories `images/train`, `images/val`, `labels/train`, `labels/val` will be created.
    :param Optional[int] frms_cnt: Number of frames to randomly sample from each video for conversion. If None, all frames are used.
    :param float threshold: Minimum confidence score threshold to filter out low-confidence pose instances. Only instances with `instance.score` >= this threshold are used.
    :param float train_size: Proportion of frames randomly assigned to the training dataset. Value must be between 0.1 and 0.99. Default: 0.7.
    :param bool verbose: If True, prints progress. Default: True.
    :param Tuple[int, ...] flip_idx: Tuple of keypoint indices used for horizontal flip augmentation during training. The tuple defines the order of keypoints after flipping. If None, it will be inferred.
    :param float padding: Fractional padding to add around the bounding boxes (relative to image dimensions). Helps to slightly enlarge bounding boxes by this percentage. Default 0.05. E.g., Useful when all body-parts are along animal length.
    :param Optional[str] single_id: If the data contains pose-estimation for multiple indivisuals, but you want to treat it as examples of a single individual, pass the name of the single individual. Defaults to None, and the YOLO data will be formatted to the number of objects which the H5 data contains.
    :return: None. Results saved in ``save_dir``.

    :example:
    >>> DATA_DIR = r'D:\troubleshooting\dlc_h5_multianimal_to_yolo\data'
    >>> VIDEO_DIR = r'D:\troubleshooting\dlc_h5_multianimal_to_yolo\videos'
    >>> SAVE_DIR = r"D:\imgs\madlc"
    >>> runner = MADLCH52Yolo(data_dir=DATA_DIR, video_dir=VIDEO_DIR, save_dir=SAVE_DIR, clahe=True, single_id='animal_1')
    >>> runner.run()
    """

    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 video_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 frms_cnt: Optional[Union[int, None]] = None,
                 verbose: bool = True,
                 threshold: float = 0,
                 train_size: float = 0.7,
                 flip_idx: Tuple[int, ...] = None,
                 greyscale: bool = False,
                 clahe: bool = False,
                 padding: float = 0.00,
                 single_id: Optional[str] = None):


        data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=H5_EXT, as_dict=True, raise_error=True)
        video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, as_dict=True, raise_error=True)
        check_if_dir_exists(in_dir=save_dir)
        self.img_dir, self.lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
        self.img_train_dir, self.img_val_dir = os.path.join(save_dir, 'images', 'train'), os.path.join(save_dir, 'images', 'val')
        self.lbl_train_dir, self.lb_val_dir = os.path.join(save_dir, 'labels', 'train'), os.path.join(save_dir, 'labels', 'val')
        if flip_idx is not None: check_valid_tuple(x=flip_idx, source=f'{self.__class__.__name__} flip_idx', valid_dtypes=(int,), minimum_length=1)
        if single_id is not None: check_str(name=f'{self.__class__.__name__} single_id', value=single_id, raise_error=True)
        check_float(name=f'{self.__class__.__name__} threshold', min_value=0.0, max_value=1.0, raise_error=True, value=threshold)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale', raise_error=True)
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe', raise_error=True)
        check_float(name=f'{self.__class__.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
        check_float(name=f'{self.__class__.__name__} padding', value=padding, max_value=1.0, min_value=0.0, raise_error=True)
        self.data_and_videos_lk = PoseImporterMixin().link_video_paths_to_data_paths(data_paths=list(data_paths.values()), video_paths=list(video_paths.values()), str_splits=Formats.DLC_NETWORK_FILE_NAMES.value)
        self.map_path = os.path.join(save_dir, 'map.yaml')
        if frms_cnt is not None: check_int(name=f'{self.__class__ .__name__} frms_cnt', value=frms_cnt, min_value=1, raise_error=True)
        create_directory(paths=[self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lb_val_dir])
        self.threshold, self.frms_cnt, self.train_size, self.save_dir = threshold, frms_cnt, train_size, save_dir
        self.verbose, self.greyscale, self.flip_idx, self.padding = verbose, greyscale, flip_idx, padding
        self.clahe, self.single_id = clahe, single_id


    def run(self):
        dfs, timer, bp_header = [], SimbaTimer(start=True), []
        for cnt, (video_name, video_data) in enumerate(self.data_and_videos_lk.items()):
            data = pd.read_hdf(video_data["DATA"], header=[0, 1, 2, 3]).replace([np.inf, -np.inf], np.nan).fillna(-1)
            new_cols = [(x[1], x[2], x[3]) for x in list(data.columns)]
            if cnt == 0:
                self.animal_names = list(dict.fromkeys([x[1] for x in list(data.columns)]))
            data.columns = new_cols
            p_data = data[[x for x in data.columns if 'likelihood' in x]]
            data = data.iloc[(p_data[(p_data > self.threshold).all(axis=1)].index)]
            bp_header = [f'{x[1]}_{x[2]}' for x in data.columns]
            bp_header = [x for x in bp_header if 'likelihood' not in x]
            data['frm_idx'] = data.index
            data['video'] = video_data['VIDEO']
            selected_frms = random.sample(list(data['frm_idx'].unique()), self.frms_cnt) if self.frms_cnt is not None else list(data['frm_idx'].unique())
            data = data[data['frm_idx'].isin(selected_frms)]
            dfs.append(data)

        dfs = pd.concat(dfs, axis=0)
        dfs['id'] = dfs['frm_idx'].astype(str) + dfs['video'].astype(str)
        train_idx = random.sample(list(dfs['id'].unique()), int(len(dfs['frm_idx'].unique()) * self.train_size))
        map_dict = {k: v for k, v in enumerate(self.animal_names)} if self.single_id is None else {0: self.single_id}
        id_dict = {v: k for k, v in enumerate(self.animal_names)}
        if self.flip_idx is None:
            self.flip_idx = get_yolo_keypoint_flip_idx(x=list(dict.fromkeys([x[:-2] for x in bp_header])))

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
            for track_cnt, animal_name in enumerate(self.animal_names):
                keypoints = frm_data.T[[idx[0] == animal_name for idx in frm_data.index]].values.reshape(-1, 3)
                keypoints[keypoints[:, 2] != 0.0, 2] = 2
                instance_str = f'{id_dict[animal_name]} ' if self.single_id is None else f'0 '
                instance_str += keypoint_array_to_yolo_annotation_str(x=keypoints, img_h=img_h, img_w=img_w, padding=self.padding)
                img_lbl += instance_str
            with open(lbl_save_path, mode='wt', encoding='utf-8') as f:
                f.write(img_lbl)
            cv2.imwrite(img_save_path, img)
        create_yolo_keypoint_yaml(path=self.save_dir, train_path=self.img_train_dir, val_path=self.img_val_dir, names=map_dict, save_path=self.map_path, kpt_shape=(len(self.flip_idx), 3), flip_idx=self.flip_idx)
        timer.stop_timer()
        stdout_success(msg=f'YOLO formated data saved in {self.save_dir} directory', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)


# DATA_DIR = r'D:\troubleshooting\dlc_h5_multianimal_to_yolo\data'
# VIDEO_DIR = r'D:\troubleshooting\dlc_h5_multianimal_to_yolo\videos'
# SAVE_DIR = r"D:\imgs\madlc"
# runner = MADLCH52Yolo(data_dir=DATA_DIR, video_dir=VIDEO_DIR, save_dir=SAVE_DIR, clahe=True, single_id='animal_1')
# runner.run()
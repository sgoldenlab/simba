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

from simba.third_party_label_appenders.transform.utils import (
    create_yolo_keypoint_yaml, get_yolo_keypoint_flip_idx)
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_valid_boolean,
                                check_valid_dataframe, check_valid_tuple)
from simba.utils.enums import Formats
from simba.utils.errors import InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (create_directory, get_fn_ext, read_img,
                                    recursive_file_search)
from simba.utils.yolo import keypoint_array_to_yolo_annotation_str

COLLECTED_DATA = 'CollectedData'


class MultiDLC2Yolo:
    """

    :example:
    >>> DLC_DIR = r'E:\deeplabcut_projects\resident_intruder_white_black-SN-2025-09-30\labeled-data'
    >>> SAVE_DIR = r'E:\yolo_resident_intruder'
    >>> runner = MultiDLC2Yolo(dlc_dir=DLC_DIR, save_dir=SAVE_DIR, verbose=True, clahe=True, names=('resident', 'intruder'))
    >>> runner.run()
    """

    def __init__(self,
                 dlc_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 train_size: float = 0.7,
                 verbose: bool = False,
                 padding: float = 0.00,
                 flip_idx: Optional[Tuple[int, ...]] = None,
                 names: Tuple[str, ...] = ('resident', 'intruder'),
                 greyscale: bool = False,
                 clahe: bool = False) -> None:

        check_if_dir_exists(in_dir=dlc_dir, source=f'{self.__class__.__name__} dlc_dir')
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale')
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe')
        check_float(name=f'{self.__class__.__name__} padding', value=padding, max_value=1.0, min_value=0.0,
                    raise_error=True)
        check_float(name=f'{self.__class__.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
        check_valid_tuple(x=names, source=f'{self.__class__.__name__} names', minimum_length=1, valid_dtypes=(str,))
        check_if_dir_exists(in_dir=save_dir)
        self.annotation_paths = recursive_file_search(directory=dlc_dir, substrings=[COLLECTED_DATA],
                                                      extensions=['csv'], case_sensitive=False, raise_error=True)
        if flip_idx is not None: check_valid_tuple(x=flip_idx, source=f'{self.__class__.__name__} flip_idx',
                                                   valid_dtypes=(int,), minimum_length=1)
        self.img_dir, self.lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
        self.img_train_dir, self.img_val_dir = os.path.join(self.img_dir, 'train'), os.path.join(self.img_dir, 'val')
        self.lbl_train_dir, self.lb_val_dir = os.path.join(self.lbl_dir, 'train'), os.path.join(self.lbl_dir, 'val')
        create_directory(paths=[self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lb_val_dir],
                         overwrite=False)
        self.names = {k: v for k, v in enumerate(names)}
        self.map_path = os.path.join(save_dir, 'map.yaml')
        self.verbose, self.greyscale, self.train_size, self.clahe = verbose, greyscale, train_size, clahe
        self.padding, self.flip_idx, self.save_dir = padding, flip_idx, save_dir

    def run(self):
        annotations, body_part_headers, timer = [], [], SimbaTimer(start=True)
        unique_individuals = None
        for file_cnt, annotation_path in enumerate(self.annotation_paths):
            file_timer = SimbaTimer(start=True)
            annotation_data = pd.read_csv(annotation_path, header=[0, 1, 2, 3])
            img_dirs = annotation_data.pop(annotation_data.columns[1]).reset_index(drop=True).values
            img_names = annotation_data.pop(annotation_data.columns[1]).reset_index(drop=True).values
            annotation_data.columns.names = ["scorer", "individuals", "bodyparts", "coords"]
            unique_bodyparts = annotation_data.columns.get_level_values("bodyparts").unique().dropna().tolist()[1:]
            if file_cnt == 0 and self.flip_idx is None:
                self.flip_idx = get_yolo_keypoint_flip_idx(x=unique_bodyparts)
            if file_cnt == 0:
                unique_individuals = annotation_data.columns.get_level_values("individuals").unique().dropna().tolist()[
                                     1:]
                if len(unique_individuals) != len(self.names.keys()):
                    raise InvalidInputError(
                        msg=f'{len(self.names.keys())} names were passed but data in {annotation_path} contains {len(unique_individuals)} IDs',
                        source=self.__class__.__name__)
            annotation_data = annotation_data.reset_index(drop=True)
            annotation_data['img_path'] = [os.path.join(os.path.dirname(annotation_path), x) for y, x in
                                           zip(img_dirs, img_names)]
            annotations.append(annotation_data)
            file_timer.stop_timer()
            print(
                f'Read annotations file {file_cnt + 1}/{len(self.annotation_paths)} (elapsed time: {file_timer.elapsed_time_str}s)...')

        annotations = pd.concat(annotations, axis=0).reset_index(drop=True)
        train_idx = random.sample(list(range(0, len(annotations))), int(len(annotations) * self.train_size))
        for cnt, (idx, idx_data) in enumerate(annotations.iterrows()):
            img_lbl = ''
            if self.verbose:
                print(f'Processing image {cnt + 1}/{len(annotations)} (DLC file {cnt + 1}/{len(annotations)})...')
            img_path = idx_data['img_path'].values[0]
            check_file_exist_and_readable(img_path)
            _, file_name, _ = get_fn_ext(img_path)
            if idx in train_idx:
                img_save_path, lbl_save_path = os.path.join(self.img_train_dir, f'{file_name}.png'), os.path.join(
                    self.lbl_train_dir, f'{file_name}.txt')
            else:
                img_save_path, lbl_save_path = os.path.join(self.img_val_dir, f'{file_name}.png'), os.path.join(
                    self.lb_val_dir, f'{file_name}.txt')
            img = read_img(img_path=img_path, greyscale=self.greyscale, clahe=self.clahe)
            img_h, img_w = img.shape[0], img.shape[1]
            for individual_id, individual in enumerate(unique_individuals):
                keypoints = idx_data.xs(individual, level='individuals').values.reshape(-1, 2).astype(np.float32)
                if np.all(np.isnan(keypoints)) or np.all(keypoints == 0.0) or np.all(
                        np.isnan(keypoints) | (keypoints == 0.0)):
                    continue
                visability_col = np.full((keypoints.shape[0], 1), fill_value=2).flatten()
                keypoints = np.insert(keypoints, 2, visability_col, axis=1)
                both_zero = (keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)
                has_nan_or_inf = ~np.isfinite(keypoints[:, 0]) | ~np.isfinite(keypoints[:, 1])
                mask = both_zero | has_nan_or_inf
                keypoints[mask, 2] = 0
                keypoints[~np.isfinite(keypoints)] = 0
                instance_str = f'{individual_id} '
                instance_str += keypoint_array_to_yolo_annotation_str(x=keypoints, img_w=img_w, img_h=img_h,
                                                                      padding=self.padding)
                img_lbl += instance_str
            with open(lbl_save_path, mode='wt', encoding='utf-8') as f:
                f.write(img_lbl)
            cv2.imwrite(img_save_path, img)

        create_yolo_keypoint_yaml(path=self.save_dir, train_path=self.img_train_dir, val_path=self.img_val_dir,
                                  names=self.names, save_path=self.map_path, kpt_shape=(len(self.flip_idx), 3),
                                  flip_idx=self.flip_idx)
        timer.stop_timer()
        stdout_success(msg=f'YOLO formated data saved in {self.save_dir} directory', source=self.__class__.__name__,
                       elapsed_time=timer.elapsed_time_str)

#
# DLC_DIR = r'E:\maplight_videos\dlc_annotations\converted'
# SAVE_DIR = r'E:\maplight_videos\dlc_annotations\converted\yolo'
#
# runner = MultiDLC2Yolo(dlc_dir=DLC_DIR, save_dir=SAVE_DIR, verbose=True, clahe=True, names=('resident', 'intruder'))
# runner.run()

# runner = DLC2Yolo(dlc_dir=r'D:\mouse_operant_data\Operant_C57_labelled_images\labeled-data', save_dir=r"D:\imgs\dlc_annot", verbose=True, clahe=True)
# runner.run()

# runner = DLC2Yolo(dlc_dir=r'D:\mouse_operant_data\Operant_C57_labelled_images\labeled-data', save_dir=r"D:\imgs\dlc_annot", verbose=True, clahe=True)
# runner.run()
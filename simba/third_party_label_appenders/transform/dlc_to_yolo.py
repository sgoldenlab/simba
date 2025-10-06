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
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (create_directory, read_img,
                                    recursive_file_search)
from simba.utils.yolo import keypoint_array_to_yolo_annotation_str


class DLC2Yolo:
    """
    Converts DLC annotations into YOLO keypoint format formatted for model training.


    .. important::
       Use for single animal DLC data. For multi-animal DLC data,

    .. note::
       ``dlc_dir`` can be a directory with subdirectories containing images and CSV files with the ``CollectedData`` substring filename.
       For creating the ``flip_idx``, see :func:`simba.third_party_label_appenders.converters.get_yolo_keypoint_flip_idx`.
       For creating the ``bp_id_idx``, see :func:`simba.third_party_label_appenders.converters.get_yolo_keypoint_bp_id_idx`

    :param Union[str, os.PathLike] dlc_dir: Directory path containing DLC-generated CSV files with keypoint annotations and images.
    :param Union[str, os.PathLike] save_dir: Output directory where YOLO-formatted images, labels, and map YAML file will be saved. Subdirectories `images/train`, `images/val`, `labels/train`, `labels/val` will be created.
    :param float train_size: Proportion of frames randomly assigned to the training dataset. Value must be between 0.1 and 0.99. Default: 0.7.
    :param bool verbose: If True, prints progress. Default: True.
    :param float padding: Fractional padding to add around the bounding boxes (relative to image dimensions). Helps to slightly enlarge bounding boxes by this percentage. Default 0.05. E.g., Useful when all body-parts are along animal length.
    :param Tuple[int, ...] flip_idx: Tuple of keypoint indices used for horizontal flip augmentation during training. The tuple defines the order of keypoints after flipping.
    :param Tuple[str] names: Tuple of animal (class) names. Used for creating the YAML class names mapping file.
    :return: None. Results saved in ``save_dir``.

    :example:
    >>> DLC_DIR = r'D:\rat_resident_intruder\dlc_data'
    >>> SAVE_DIR = r'D:\rat_resident_intruder\yolo_3'
    >>> runner = DLC2Yolo(dlc_dir=DLC_DIR, save_dir=SAVE_DIR, verbose=True, clahe=True, names=('resident', 'intruder'))
    >>> runner.run()
    """

    def __init__(self,
                 dlc_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 train_size: float = 0.7,
                 verbose: bool = False,
                 padding: float = 0.00,
                 flip_idx: Optional[Tuple[int, ...]] = None,
                 names: Tuple[str, ...] = ('mouse',),
                 greyscale: bool = False,
                 clahe: bool = False) -> None:

        check_if_dir_exists(in_dir=dlc_dir, source=f'{self.__class__.__name__} dlc_dir')
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale')
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe')
        check_float(name=f'{self.__class__.__name__} padding', value=padding, max_value=1.0, min_value=0.0, raise_error=True)
        check_float(name=f'{self.__class__.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
        check_valid_tuple(x=names, source=f'{self.__class__.__name__} names', minimum_length=1, valid_dtypes=(str,))
        check_if_dir_exists(in_dir=save_dir)
        self.annotation_paths = recursive_file_search(directory=dlc_dir, substrings=['CollectedData'],  extensions=['csv'], case_sensitive=False, raise_error=True)
        if flip_idx is not None: check_valid_tuple(x=flip_idx, source=f'{self.__class__.__name__} flip_idx', valid_dtypes=(int,), minimum_length=1)
        self.img_dir, self.lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
        self.img_train_dir, self.img_val_dir = os.path.join(self.img_dir, 'train'), os.path.join(self.img_dir, 'val')
        self.lbl_train_dir, self.lb_val_dir = os.path.join(self.lbl_dir, 'train'), os.path.join(self.lbl_dir, 'val')
        create_directory(paths=[self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lb_val_dir], overwrite=False)
        self.names = {k: v for k, v in enumerate(names)}
        self.map_path = os.path.join(save_dir, 'map.yaml')
        self.verbose, self.greyscale, self.train_size, self.clahe = verbose, greyscale, train_size, clahe
        self.padding, self.flip_idx, self.save_dir = padding, flip_idx, save_dir


    def run(self):
        annotations, timer, body_part_headers = [], SimbaTimer(start=True), []
        for file_cnt, annotation_path in enumerate(self.annotation_paths):
            annotation_data = pd.read_csv(annotation_path, header=[0, 1, 2])
            img_paths = annotation_data.pop(annotation_data.columns[0]).reset_index(drop=True).values
            body_parts = []
            body_part_headers = []
            for i in annotation_data.columns[1:]:
                if 'unnamed:' not in i[1].lower() and i[1] not in body_parts:
                    body_parts.append(i[1])
            for i in body_parts:
                body_part_headers.append(f'{i}_x'); body_part_headers.append(f'{i}_y')
            annotation_data.columns = body_part_headers
            check_valid_dataframe(df=annotation_data, source=self.__class__.__name__, valid_dtypes=Formats.NUMERIC_DTYPES.value)
            annotation_data = annotation_data.reset_index(drop=True)
            img_paths = [os.path.join(os.path.dirname(annotation_path), os.path.basename(x)) for x in img_paths]
            annotation_data['img_path'] = img_paths
            annotations.append(annotation_data)

        if self.flip_idx is None:
            self.flip_idx = get_yolo_keypoint_flip_idx(x=list(dict.fromkeys([x[:-2] for x in body_part_headers])))

        annotations = pd.concat(annotations, axis=0).reset_index(drop=True)
        img_paths = annotations.pop('img_path').reset_index(drop=True).values
        train_idx = random.sample(list(range(0, len(annotations))), int(len(annotations) * self.train_size))
        bp_id_idx = np.array_split(np.array(range(0, int(len(body_part_headers)/2))), len(self.names.keys()))
        bp_id_idx = [list(x) for x in bp_id_idx]

        for cnt, (idx, idx_data) in enumerate(annotations.iterrows()):
            img_lbl = ''
            if self.verbose:
                print(f'Processing image {cnt + 1}/{len(annotations)}...')
            file_name = f"{os.path.basename(os.path.dirname(img_paths[cnt]))}.{os.path.splitext(os.path.basename(img_paths[cnt]))[0]}"
            if idx in train_idx:
                img_save_path, lbl_save_path = os.path.join(self.img_train_dir, f'{file_name}.png'), os.path.join(self.lbl_train_dir, f'{file_name}.txt')
            else:
                img_save_path, lbl_save_path = os.path.join(self.img_val_dir, f'{file_name}.png'), os.path.join(self.lb_val_dir, f'{file_name}.txt')
            check_file_exist_and_readable(img_paths[cnt])
            img = read_img(img_path=img_paths[cnt], greyscale=self.greyscale, clahe=self.clahe)
            img_h, img_w = img.shape[0], img.shape[1]
            keypoints_with_id = {}
            for k, idx in enumerate(bp_id_idx):
                keypoints_with_id[k] = idx_data.values.reshape(-1, 2)[idx, :]
            for id, keypoints in keypoints_with_id.items():
                if np.all(np.isnan(keypoints)) or np.all(keypoints == 0.0) or np.all(np.isnan(keypoints) | (keypoints == 0.0)):
                    continue
                visability_col = np.full((keypoints.shape[0], 1), fill_value=2).flatten()
                keypoints = np.insert(keypoints, 2, visability_col, axis=1)
                both_zero = (keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)
                has_nan_or_inf = ~np.isfinite(keypoints[:, 0]) | ~np.isfinite(keypoints[:, 1])
                mask = both_zero | has_nan_or_inf
                keypoints[mask, 2] = 0
                keypoints[~np.isfinite(keypoints)] = 0
                instance_str = f'{id} '
                instance_str += keypoint_array_to_yolo_annotation_str(x=keypoints, img_w=img_w, img_h=img_h, padding=self.padding)
                img_lbl += instance_str
            with open(lbl_save_path, mode='wt', encoding='utf-8') as f:
                f.write(img_lbl)
            cv2.imwrite(img_save_path, img)
        create_yolo_keypoint_yaml(path=self.save_dir, train_path=self.img_train_dir, val_path=self.img_val_dir, names=self.names, save_path=self.map_path, kpt_shape=(len(self.flip_idx), 3), flip_idx=self.flip_idx)
        timer.stop_timer()
        stdout_success(msg=f'YOLO formated data saved in {self.save_dir} directory', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)


#
# DLC_DIR = r'D:\rat_resident_intruder\dlc_data'
# SAVE_DIR = r'D:\rat_resident_intruder\yolo_3'
#
# runner = DLC2Yolo(dlc_dir=DLC_DIR, save_dir=SAVE_DIR, verbose=True, clahe=True, names=('resident', 'intruder'))
# runner.run()

# runner = DLC2Yolo(dlc_dir=r'D:\mouse_operant_data\Operant_C57_labelled_images\labeled-data', save_dir=r"D:\imgs\dlc_annot", verbose=True, clahe=True)
# runner.run()

# runner = DLC2Yolo(dlc_dir=r'D:\mouse_operant_data\Operant_C57_labelled_images\labeled-data', save_dir=r"D:\imgs\dlc_annot", verbose=True, clahe=True)
# runner.run()
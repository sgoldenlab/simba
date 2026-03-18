import os
import random
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.third_party_label_appenders.converters import create_yolo_yaml
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_tuple)
from simba.utils.enums import Formats
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (create_directory, get_fn_ext, read_img,
                                    recursive_file_search)


class LitPose2YOLOBbox:
    """
    Convert LitPose keypoint annotations into a YOLO bounding-box dataset.

    :param Union[str, os.PathLike] litpose_dir: Path to LitPose directory containing annotation CSV files and the ``labeled-data`` image folder.
    :param Union[str, os.PathLike] save_dir: Output directory where YOLO-formatted ``images`` and ``labels`` subdirectories are created.
    :param float train_size: Fraction of samples assigned to the training split. Default 0.7.
    :param bool verbose: If True, print per-image progress during conversion.
    :param float padding: Extra fractional padding around each axis-aligned box inferred from keypoints.
    :param Optional[int] sample_n: Optional cap on the number of sampled frames before split. If None, all frames are used.
    :param Tuple[str, ...] names: Class names in YOLO index order.
    :param bool greyscale: If True, load and save images in grayscale.
    :param bool clahe: If True, apply CLAHE preprocessing when reading images.

    References
    ----------
    .. [1] Lightning Pose documentation: https://lightning-pose.readthedocs.io/en/latest/
    .. [2] Biderman et al., Lightning Pose: improved animal pose estimation via semi-supervised learning, Bayesian ensembling and cloud-native open-source tools, *Nature Methods* (2024), doi: https://doi.org/10.1038/s41592-024-02319-1

    :example:
    >>> runner = LitPose2YOLOBbox(litpose_dir=r'Z:\home\simon\lp_300126', save_dir=r'E:\litpose_yolo\bbox', verbose=True, clahe=False, greyscale=False, sample_n=1000, padding=0.15)
    >>> runner.run()
    """

    def __init__(self,
                 litpose_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 train_size: float = 0.7,
                 verbose: bool = False,
                 padding: float = 0.00,
                 sample_n: Optional[int] = None,
                 names: Tuple[str, ...] = ('mouse',),
                 greyscale: bool = False,
                 clahe: bool = False) -> None:

        check_if_dir_exists(in_dir=litpose_dir, source=f'{self.__class__.__name__} litpose_dir')
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale')
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe')
        check_float(name=f'{self.__class__.__name__} padding', value=padding, max_value=1.0, min_value=0.0, raise_error=True)
        check_float(name=f'{self.__class__.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
        check_valid_tuple(x=names, source=f'{self.__class__.__name__} names', minimum_length=1, valid_dtypes=(str,))
        if sample_n is not None:
            check_int(name=f'{self.__class__.__name__} sample', value=sample_n, min_value=1)
        check_if_dir_exists(in_dir=save_dir)
        self.annotation_paths = recursive_file_search(directory=litpose_dir, substrings=['CollectedData'], extensions=['csv'], case_sensitive=False, raise_error=True)
        self.labeled_imgs_dir, self.litpose_dir = os.path.join(litpose_dir, 'labeled-data'), litpose_dir
        check_if_dir_exists(in_dir=self.labeled_imgs_dir)
        self.img_dir, self.lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
        self.img_train_dir, self.img_val_dir = os.path.join(self.img_dir, 'train'), os.path.join(self.img_dir, 'val')
        self.lbl_train_dir, self.lb_val_dir = os.path.join(self.lbl_dir, 'train'), os.path.join(self.lbl_dir, 'val')
        create_directory(paths=[self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lb_val_dir], overwrite=False)
        self.names = {name: idx for idx, name in enumerate(names)}
        self.map_path = os.path.join(save_dir, 'map.yaml')
        self.verbose, self.greyscale, self.train_size, self.clahe = verbose, greyscale, train_size, clahe
        self.padding, self.save_dir, self.sample_n = padding, save_dir, sample_n

    def run(self):
        annotations, timer, body_part_headers = [], SimbaTimer(start=True), []
        for _, annotation_path in enumerate(self.annotation_paths):
            annotation_filename = get_fn_ext(filepath=annotation_path)[1]
            annotation_data = pd.read_csv(annotation_path, header=[0, 1, 2])
            img_paths = annotation_data.pop(annotation_data.columns[0]).reset_index(drop=True).values
            body_parts, body_part_headers = [], []
            for i in annotation_data.columns[1:]:
                if 'unnamed:' not in i[1].lower() and i[1] not in body_parts:
                    body_parts.append(i[1])
            for i in body_parts:
                body_part_headers.extend((f'{i}_x', f'{i}_y'))
            annotation_data.columns = body_part_headers
            check_valid_dataframe(df=annotation_data, source=self.__class__.__name__, valid_dtypes=Formats.NUMERIC_DTYPES.value)
            annotation_data = annotation_data.reset_index(drop=True)
            img_names = [get_fn_ext(os.path.basename(x))[1] for x in img_paths]
            save_names = [f'{annotation_filename}_{get_fn_ext(p)[1]}' for p in img_paths]
            annotation_data['img_name'] = img_names
            annotation_data['img_path'] = img_paths
            annotation_data['save_name'] = save_names
            annotations.append(annotation_data)

        annotations = pd.concat(annotations, axis=0).reset_index(drop=True)
        if self.sample_n is not None:
            annotations = annotations.sample(n=min(self.sample_n, len(annotations))).reset_index(drop=True)
        train_idx = random.sample(list(range(0, len(annotations))), int(len(annotations) * self.train_size))
        bp_id_idx = np.array_split(np.array(range(0, int(len(body_part_headers) / 2))), len(self.names.values()))
        bp_id_idx = [list(x) for x in bp_id_idx]
        for cnt, (idx, idx_data) in enumerate(annotations.iterrows()):
            img_lbl = []
            if self.verbose:
                stdout_information(msg=f'Processing image {cnt + 1}/{len(annotations)}...')
            save_name, _, img_path = idx_data.pop('save_name'), idx_data.pop('img_name'), os.path.join(self.litpose_dir, idx_data.pop('img_path'))
            if idx in train_idx:
                img_save_path, lbl_save_path = os.path.join(self.img_train_dir, f'{save_name}.png'), os.path.join(self.lbl_train_dir, f'{save_name}.txt')
            else:
                img_save_path, lbl_save_path = os.path.join(self.img_val_dir, f'{save_name}.png'), os.path.join(self.lb_val_dir, f'{save_name}.txt')
            check_file_exist_and_readable(img_path, True)
            img = read_img(img_path=img_path, greyscale=self.greyscale, clahe=self.clahe)
            img_h, img_w = img.shape[0], img.shape[1]
            keypoints_with_id = {}
            for k, bp_idx in enumerate(bp_id_idx):
                keypoints_with_id[k] = idx_data.values.astype(float).reshape(-1, 2)[bp_idx]
            for cls_id, keypoints in keypoints_with_id.items():
                if np.all(np.isnan(keypoints)) or np.all(keypoints == 0.0) or np.all(np.isnan(keypoints) | (keypoints == 0.0)):
                    continue
                finite = np.isfinite(keypoints[:, 0]) & np.isfinite(keypoints[:, 1])
                non_zero = ~((keypoints[:, 0] == 0.0) & (keypoints[:, 1] == 0.0))
                valid_kps = keypoints[finite & non_zero]
                if valid_kps.shape[0] == 0:
                    continue
                x_min, y_min = np.min(valid_kps[:, 0]), np.min(valid_kps[:, 1])
                x_max, y_max = np.max(valid_kps[:, 0]), np.max(valid_kps[:, 1])
                w, h = max(1.0, float(x_max - x_min)), max(1.0, float(y_max - y_min))
                x_min = np.clip(x_min - (w * self.padding) / 2.0, 0, img_w - 1)
                x_max = np.clip(x_max + (w * self.padding) / 2.0, 0, img_w - 1)
                y_min = np.clip(y_min - (h * self.padding) / 2.0, 0, img_h - 1)
                y_max = np.clip(y_max + (h * self.padding) / 2.0, 0, img_h - 1)
                box_w, box_h = float(x_max - x_min), float(y_max - y_min)
                if box_w <= 0.0 or box_h <= 0.0:
                    continue
                x_center = ((x_min + x_max) / 2.0) / img_w
                y_center = ((y_min + y_max) / 2.0) / img_h
                yolo_w, yolo_h = box_w / img_w, box_h / img_h
                img_lbl.append(f'{cls_id} {x_center} {y_center} {yolo_w} {yolo_h}')
            with open(lbl_save_path, mode='wt', encoding='utf-8') as f:
                f.write('\n'.join(img_lbl))
            cv2.imwrite(img_save_path, img)
        create_yolo_yaml(path=self.save_dir, train_path=self.img_train_dir, val_path=self.img_val_dir, names=self.names, save_path=self.map_path)
        timer.stop_timer()
        stdout_success(msg=f'YOLO formated data saved in {self.save_dir} directory', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)


# runner = LitPose2YOLOBbox(litpose_dir=r'Z:\home\simon\lp_300126', save_dir=r'E:\litpose_yolo\bbox', verbose=True, clahe=False, greyscale=False, sample_n=1000, padding=0.15)
# runner.run()


#
# # Keypoint dataset:
# # runner = LitPose2YOLO(litpose_dir=LIPOSE_DIR, save_dir=SAVE_DIR, verbose=True, clahe=True, sample_n=1000, padding=0.10)
# # runner.run()
#




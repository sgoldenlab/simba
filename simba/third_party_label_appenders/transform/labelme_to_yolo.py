import json
import os
import random
from typing import Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2

from simba.mixins.image_mixin import ImageMixin
from simba.third_party_label_appenders.transform.utils import b64_to_arr
from simba.utils.checks import (check_float, check_if_dir_exists,
                                check_if_keys_exist_in_dict,
                                check_valid_boolean)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (create_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, img_array_to_clahe)
from simba.utils.warnings import ROIWarning


class LabelmeBoundingBoxes2YoloBoundingBoxes:
    """
    Convert LabelMe annotations in json to YOLO format and save the corresponding images and labels in txt format.

    .. note::
       For more information on the LabelMe annotation tool, see the `LabelMe GitHub repository <https://github.com/wkentaro/labelme>`_.
       The Labelme Json files **has too** contain a `imageData` key holding the image as a b64 string.
       For an expected Labelme json format, see `THIS FILE <https://github.com/sgoldenlab/simba/blob/master/misc/labelme_ex.json>`_.

    .. seealso::
       To split YOLO data into train, test, and validation sets (expected by e.g., UltraLytics), see :func:`simba.third_party_label_appenders.converters.split_yolo_train_test_val`.
       To convert Labelme points annotations to YOLO keypoint training data, see :func:`simba.third_party_label_appenders.transform.labelme_to_yolo_keypoints.LabelmeKeypoints2YoloKeypoints`.

    .. important::
       For YOLO bounding boxes (not YOLO keypoint data!) from labelme keypoints.

    :param Union[str, os.PathLike labelme_dir: Path to the directory containing LabelMe annotation `.json` files.
    :param Union[str, os.PathLike save_dir: Directory where the YOLO-format images and labels will be saved. Will create 'images/', 'labels/', and 'map.json' inside this directory.
    :param bool obb: If True, saves annotations as oriented bounding boxes (8 coordinates). If False, uses standard YOLO format (x_center, y_center, width, height)
    :param bool verbose: If True, prints progress messages during conversion.


    :example:
    >>> LABELME_DIR = r'D:\platea\ts_annotations'
    >>> SAVE_DIR = r"D:\platea\yolo"
    >>> runner = LabelmeBoundingBoxes2YoloBoundingBoxes(labelme_dir=LABELME_DIR, save_dir=SAVE_DIR)
    >>> runner.run()
    """


    def __init__(self,
                 labelme_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 obb: bool = False,
                 verbose: bool = True,
                 clahe: bool = False,
                 train_size: float = 0.7,
                 greyscale: bool = False) -> None:


        check_if_dir_exists(in_dir=os.path.dirname(save_dir), source=f'{self.__class__.__name__} save_dir', raise_error=True)
        check_if_dir_exists(in_dir=labelme_dir, source=f'{self.__class__.__name__} labelme_dir', raise_error=True)
        self.labelme_file_paths = find_files_of_filetypes_in_directory(directory=labelme_dir, extensions=['.json'], raise_error=True)
        self.map_path = os.path.join(save_dir, 'map.json')
        self.img_dir, self.lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
        self.img_train_dir, self.img_val_dir = os.path.join(self.img_dir, 'train'), os.path.join(self.img_dir, 'val')
        self.lbl_train_dir, self.lbl_val_dir = os.path.join(self.lbl_dir, 'train'), os.path.join(self.lbl_dir, 'val')
        create_directory(paths=[self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lbl_val_dir], overwrite=False)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=[obb], source=f'{self.__class__.__name__} obb', raise_error=True)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} clahe', raise_error=True)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} greyscale', raise_error=True)
        check_float(name=f'{self.__class__.__name__} train_size', value=train_size, min_value=0.0, max_value=1.0)
        self.obb, self.verbose, self.save_dir = obb, verbose, save_dir
        self.clahe, self.greyscale, self.train_size = clahe, greyscale, train_size

    def run(self):
        train_idx = random.sample(range(0, len(self.labelme_file_paths)), int(len(self.labelme_file_paths)*self.train_size))
        timer = SimbaTimer(start=True)
        labels = {}
        for file_cnt, file_path in enumerate(self.labelme_file_paths):
            if self.verbose: print(f'Labelme to YOLO file {file_cnt + 1}/{len(self.labelme_file_paths)}...')
            with open(file_path) as f:
                annot_data = json.load(f)
            check_if_keys_exist_in_dict(data=annot_data, key=['shapes', 'imageData', 'imagePath'], name=file_path)
            img_name = get_fn_ext(filepath=annot_data['imagePath'])[1]
            img = b64_to_arr(annot_data['imageData'])
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.greyscale:
                img = ImageMixin.img_to_greyscale(img=img)
            if self.clahe:
                img = img_array_to_clahe(img=img)
            img_h, img_w = img.shape[:2]
            if file_cnt in train_idx:
                label_save_path = os.path.join(self.lbl_train_dir, f'{img_name}.txt')
                img_save_path = os.path.join(self.img_train_dir, f'{img_name}.png')
            else:
                label_save_path = os.path.join(self.lbl_val_dir, f'{img_name}.txt')
                img_save_path = os.path.join(self.img_val_dir, f'{img_name}.png')
            roi_str = ''
            for bp_data in annot_data['shapes']:
                check_if_keys_exist_in_dict(data=bp_data, key=['label', 'points', 'shape_type'], name=file_path)
                if bp_data['shape_type'] == 'rectangle':
                    if bp_data['label'] not in labels.keys():
                        label_id = len(labels.keys())
                        labels[bp_data['label']] = len(labels.keys())
                    else:
                        label_id = labels[bp_data['label']]
                    x1, y1 = bp_data['points'][0]
                    x2, y2 = bp_data['points'][1]
                    x_min, x_max = sorted([x1, x2])
                    y_min, y_max = sorted([y1, y2])
                    if not self.obb:
                        w = (x_max - x_min) / img_w
                        h = (y_max - y_min) / img_h
                        x_center = (x_min + (x_max - x_min) / 2) / img_w
                        y_center = (y_min + (y_max - y_min) / 2) / img_h
                        roi_str += ' '.join([f"{label_id}", str(x_center), str(y_center), str(w), str(h) + '\n'])
                    else:
                        top_left = (x_min / img_w, y_min / img_h)
                        top_right = (x_max / img_w, y_min / img_h)
                        bottom_right = (x_max / img_w, y_max / img_h)
                        bottom_left = (x_min / img_w, y_max / img_h)
                        roi_str += ' '.join([f"{label_id}", str(top_left[0]), str(top_left[1]), str(top_right[0]), str(top_right[1]), str(bottom_right[0]), str(bottom_right[1]), str(bottom_left[0]), str(bottom_left[1]) + '\n'])
                else:
                    ROIWarning(msg=f'Only Labelme shape type rectangle recognized for YOLO bounding box transformation. Got {bp_data["shape_type"]}. Skipping annotation...', source=self.__class__.__name__)
            with open(label_save_path, mode='wt', encoding='utf-8') as f:
                f.write(roi_str)
            cv2.imwrite(img_save_path, img)
        with open(self.map_path, 'w') as f:
            json.dump(labels, f, indent=4)
        timer.stop_timer()
        if self.verbose: stdout_success(msg=f'Labelme to YOLO conversion complete. Data saved in directory {self.save_dir}.', elapsed_time=timer.elapsed_time_str)


# LABELME_DIR = r'D:\platea\ts_annotations'
# SAVE_DIR = r"D:\platea\yolo"
# runner = LabelmeBoundingBoxes2YoloBoundingBoxes(labelme_dir=LABELME_DIR, save_dir=SAVE_DIR)
# runner.run()
#


import os
import random
import shutil
from typing import Dict, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2

from simba.mixins.image_mixin import ImageMixin
from simba.third_party_label_appenders.converters import create_yolo_yaml
from simba.third_party_label_appenders.transform.utils import (
    labelme_img_path, labelme_img_to_arr)
from simba.utils.checks import (check_float, check_if_dir_exists,
                                check_if_keys_exist_in_dict,
                                check_valid_boolean)
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (create_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, img_array_to_clahe, read_json)
from simba.utils.warnings import NoDataFoundWarning, ROIWarning

PRINT_INTERVAL = 50   # NOTE: files between progress printouts - a line per file is noise on a large annotation set.


def _labelme_to_yolo_bbox_helper(file_path: Union[str, os.PathLike],
                                 labels: Dict[str, int],
                                 train_paths: Tuple[str, str],
                                 val_paths: Tuple[str, str],
                                 train_file_paths: Tuple[str, ...],
                                 obb: bool,
                                 greyscale: bool,
                                 clahe: bool) -> Union[str, None]:
    """
    Helper converting a single Labelme json annotation to a YOLO image and label file.

    :param Union[str, os.PathLike] file_path: Path to the Labelme json annotation.
    :param Dict[str, int] labels: Map of Labelme shape label to YOLO class id, created before the images are converted so that the ids are identical in every process.
    :param Tuple[str, str] train_paths: The (image, label) directories holding the training set.
    :param Tuple[str, str] val_paths: The (image, label) directories holding the validation set.
    :param Tuple[str, ...] train_file_paths: The Labelme json files assigned to the training set.
    :param bool obb: If True, saves annotations as oriented bounding boxes (8 coordinates), else as (x_center, y_center, width, height).
    :param bool greyscale: If True, saves the image in greyscale.
    :param bool clahe: If True, saves the image with CLAHE applied.
    :return: The path of the saved label file, or None if the annotation held no readable image and was skipped.
    :rtype: Union[str, None]
    """

    annot_data = read_json(x=file_path)
    check_if_keys_exist_in_dict(data=annot_data, key=['shapes', 'imageData', 'imagePath'], name=file_path)
    img_name = get_fn_ext(filepath=annot_data['imagePath'])[1]
    img_dir, lbl_dir = train_paths if file_path in train_file_paths else val_paths
    label_save_path, img_save_path = os.path.join(lbl_dir, f'{img_name}.txt'), os.path.join(img_dir, f'{img_name}.png')
    src_img_path = None if annot_data.get('imageData') is not None else labelme_img_path(annot_data=annot_data, annot_path=file_path, raise_error=False)
    copy_img = (not greyscale) and (not clahe) and (src_img_path is not None) and (get_fn_ext(filepath=src_img_path)[2].lower() == '.png') and (annot_data.get('imageHeight') is not None) and (annot_data.get('imageWidth') is not None)
    if (annot_data.get('imageData') is None) and (src_img_path is None):   # NOTE: one annotation without an image should not sink the entire batch - it is reported and skipped.
        NoDataFoundWarning(msg=f'The labelme file {file_path} holds no imageData, and the image {annot_data["imagePath"]} could not be found in {os.path.dirname(file_path)}. Skipping this annotation.', source=_labelme_to_yolo_bbox_helper.__name__)
        return None
    if copy_img:   # NOTE: decoding a png and re-compressing it is by far the most expensive part of the conversion - when the image needs no transform it is copied, and the size is read from the annotation.
        img_h, img_w = int(annot_data['imageHeight']), int(annot_data['imageWidth'])
        shutil.copyfile(src_img_path, img_save_path)
    else:
        img = labelme_img_to_arr(annot_data=annot_data, annot_path=file_path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if greyscale:
            img = ImageMixin.img_to_greyscale(img=img)
        if clahe:
            img = img_array_to_clahe(img=img)
        img_h, img_w = img.shape[:2]
        cv2.imwrite(img_save_path, img)
    roi_str = ''
    for bp_data in annot_data['shapes']:
        check_if_keys_exist_in_dict(data=bp_data, key=['label', 'points', 'shape_type'], name=file_path)
        if bp_data['shape_type'] == 'rectangle':
            label_id = labels[bp_data['label']]
            x1, y1 = bp_data['points'][0]
            x2, y2 = bp_data['points'][1]
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            if not obb:
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
            ROIWarning(msg=f'Only Labelme shape type rectangle recognized for YOLO bounding box transformation. Got {bp_data["shape_type"]}. Skipping annotation...', source=_labelme_to_yolo_bbox_helper.__name__)
    with open(label_save_path, mode='wt', encoding='utf-8') as f:
        f.write(roi_str)

    return label_save_path


class LabelmeBoundingBoxes2YoloBoundingBoxes:
    """
    Convert LabelMe annotations in json to YOLO format and save the corresponding images and labels in txt format.

    .. image:: _static/img/simba.third_party_label_appenders.transform.labelme_to_yolo.LabelmeBoundingBoxes2YoloBoundingBoxes.webp
       :alt: LabelMe rectangle annotations (two corner points plus a base64 image) are decoded and converted into a YOLO bounding-box dataset (class, x_center, y_center, width, height; or 8 oriented-corner coordinates when obb=True), split into train/val with a map.yaml
       :width: 800
       :align: center

    .. note::
       For more information on the LabelMe annotation tool, see the `LabelMe GitHub repository <https://github.com/wkentaro/labelme>`_.
       The Labelme Json files hold the image either in a `imageData` key as a b64 string, or as a path to the image in the `imagePath` key.
       For an expected Labelme json format, see `THIS FILE <https://github.com/sgoldenlab/simba/blob/master/misc/labelme_ex.json>`_.

    .. seealso::
       To split YOLO data into train, test, and validation sets (expected by e.g., UltraLytics), see :func:`simba.third_party_label_appenders.converters.split_yolo_train_test_val`.
       To convert Labelme points annotations to YOLO keypoint training data, see :func:`simba.third_party_label_appenders.transform.labelme_to_yolo_keypoints.LabelmeKeypoints2YoloKeypoints`.
       To generate the labelme bounding-box project in the first place - from videos, using SAM3 - see :class:`~simba.third_party_label_appenders.transform.sam3_to_labelme_bbox.SAM3ToLabelmeBBox`.

    .. important::
       For YOLO bounding boxes (not YOLO keypoint data!) from labelme keypoints.

    :param Union[str, os.PathLike labelme_dir: Path to the directory containing LabelMe annotation `.json` files.
    :param Union[str, os.PathLike save_dir: Directory where the YOLO-format images and labels will be saved. Will create 'images/', 'labels/', and 'map.yaml' inside this directory.
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
        self.map_path = os.path.join(save_dir, 'map.yaml')
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
        timer = SimbaTimer(start=True)
        train_idx = random.sample(range(0, len(self.labelme_file_paths)), int(len(self.labelme_file_paths)*self.train_size))
        train_file_paths = tuple([self.labelme_file_paths[x] for x in train_idx])
        labels, self.skipped_cnt = {}, 0
        for file_path in self.labelme_file_paths:   # NOTE: the class ids are assigned before the images are converted - the parallel workers cannot share a growing map, and this keeps the ids in Labelme file order as when run on a single core.
            annot_data = read_json(x=file_path)
            check_if_keys_exist_in_dict(data=annot_data, key=['shapes'], name=file_path)
            for bp_data in annot_data['shapes']:
                check_if_keys_exist_in_dict(data=bp_data, key=['label', 'shape_type'], name=file_path)
                if (bp_data['shape_type'] == 'rectangle') and (bp_data['label'] not in labels.keys()):
                    labels[bp_data['label']] = len(labels.keys())
        if self.verbose:
            stdout_information(msg=f'Converting {len(self.labelme_file_paths)} Labelme file(s) with {len(labels)} label(s)...', source=self.__class__.__name__)
        for file_cnt, file_path in enumerate(self.labelme_file_paths):
            result = _labelme_to_yolo_bbox_helper(file_path=file_path,
                                                 labels=labels,
                                                 train_paths=(self.img_train_dir, self.lbl_train_dir),
                                                 val_paths=(self.img_val_dir, self.lbl_val_dir),
                                                 train_file_paths=train_file_paths,
                                                 obb=self.obb,
                                                 greyscale=self.greyscale,
                                                 clahe=self.clahe)
            if result is None:
                self.skipped_cnt += 1
            if self.verbose and ((file_cnt + 1) % PRINT_INTERVAL == 0 or (file_cnt + 1) == len(self.labelme_file_paths)):
                timer.stop_timer()
                stdout_information(msg=f'Labelme to YOLO file {file_cnt+1}/{len(self.labelme_file_paths)} (elapsed: {timer.elapsed_time_str}s)', source=self.__class__.__name__)

        create_yolo_yaml(path=self.save_dir, train_path=self.img_train_dir, val_path=self.img_val_dir, names=labels, save_path=self.map_path)
        timer.stop_timer()
        skipped_str = '' if self.skipped_cnt == 0 else f' {self.skipped_cnt} file(s) skipped, see the warnings above.'
        stdout_success(msg=f'Labelme to YOLO conversion of {len(self.labelme_file_paths) - self.skipped_cnt}/{len(self.labelme_file_paths)} file(s) complete. Data saved in directory {self.save_dir}.{skipped_str}', elapsed_time=timer.elapsed_time_str)

if __name__ == '__main__':
    LABELME_DIR = r"G:\netholabs\pellet_labelme_0825_0827"
    SAVE_DIR = r"G:\netholabs\pellet_yolo_0828"
    runner = LabelmeBoundingBoxes2YoloBoundingBoxes(labelme_dir=LABELME_DIR, save_dir=SAVE_DIR)
    runner.run()

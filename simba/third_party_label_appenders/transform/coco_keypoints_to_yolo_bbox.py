import argparse
import os
import random
import sys
from copy import copy
from typing import Optional, Tuple, Union

import yaml
from shapely.geometry import Polygon

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2
import numpy as np

from simba.mixins.geometry_mixin import GeometryMixin
from simba.third_party_label_appenders.converters import create_yolo_yaml
from simba.third_party_label_appenders.transform.utils import \
    create_yolo_keypoint_yaml
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict,
                                check_valid_boolean, check_valid_tuple)
from simba.utils.enums import Options
from simba.utils.errors import (FaultyTrainingSetError, InvalidInputError,
                                NoFilesFoundError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (create_directory, get_fn_ext, read_img,
                                    read_json, recursive_file_search)


class COCOKeypoints2YoloBbox:

    """
    Convert COCO Keypoints version 1.0 data format into a YOLO bounding box training set.

    .. note::
       COCO keypoint files can be created using `https://www.cvat.ai/ <https://www.cvat.ai/>`__.

       This function expects the path to a single  COCO Keypoints version 1.0 file. To merge several before passing the file to thsi function, use
       :func:`simba.third_party_label_appenders.transform.utils.merge_coco_keypoints_files`.

    .. important::
       All image file names have to be unique.

    .. seealso::
      To convert OCO Keypoints version 1.0 data format into a YOLO keypoint training set, use :func:`simba.third_party_label_appenders.transform.coco_keypoints_to_yolo.COCOKeypoints2Yolo`

    :param Union[str, os.PathLike] coco_path: Path to coco keypoints 1.0 file in json format.
    :param Union[str, os.PathLike] img_dir: Directory holding img files representing the annotated entries in the ``coco_path``. Will search recursively, so its OK to have images in subdirectories.
    :param Union[str, os.PathLike] save_dir: Directory where to save the yolo formatted data.
    :param Tuple[float, float, float] split: The size of the training set. Value between 0-1.0 representing the percent of training data.
    :param bool verbose: If true, prints progress. Default: True.
    :param Tuple[int, ...] flip_idx: Tuple of ints, representing the flip of body-part coordinates when the animal image flips 180 degrees.
    :return: None

    :example:
    >>> runner = COCOKeypoints2Yolo(coco_path=r"D:\cvat_annotations\frames\coco_keypoints_1\s1\annotations\s1.json", img_dir=r"D:\cvat_annotations\frames\simon", save_dir=r"D:\cvat_annotations\frames\yolo_keypoints", clahe=True)
    >>> runner.run()

    :example II:
    >>> runner = COCOKeypoints2Yolo(coco_path=r"D:\cvat_annotations\frames\coco_keypoints_1\merged.json", img_dir=r"D:\cvat_annotations\frames", save_dir=r"D:\cvat_annotations\frames\yolo", clahe=False)
    >>> runner.run()

     :references:
        .. [1] Helpful YouTube tutorial by Farhan to get YOLO tracking data in animals - `https://www.youtube.com/watch?v=CcGbgFPwQTc <https://www.youtube.com/watch?v=CcGbgFPwQTc>`_.
        .. [2] Great YouTube tutorial by Felipe on annotating data and making data YOLO ready - `https://www.youtube.com/watch?v=m9fH9OWn8YM <https://www.youtube.com/watch?v=m9fH9OWn8YM>`_.
    """

    def __init__(self,
                 coco_path: Union[str, os.PathLike],
                 img_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 train_size: float = 0.7,
                 verbose: bool = True,
                 greyscale: bool = False,
                 clahe: bool = False,
                 bbox_pad: Optional[float] = None,
                 obb: Optional[bool] = False):

        check_file_exist_and_readable(file_path=coco_path)
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        check_if_dir_exists(in_dir=img_dir, source=f'{self.__class__.__name__} img_dir')
        check_float(name=f'{self.__class__.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
        if bbox_pad is not None: check_float(name=f'{self.__class__.__name__} bbox_pad', value=bbox_pad, max_value=1.0, min_value=10e-6)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale', raise_error=True)
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe', raise_error=True)
        check_valid_boolean(value=obb, source=f'{self.__class__.__name__} obb', raise_error=True)
        self.train_img_dir, self.val_img_dir = os.path.join(save_dir, 'images', 'train'), os.path.join(save_dir, 'images', 'val')
        self.train_lbl_dir, self.val_lbl_dir = os.path.join(save_dir, 'labels', 'train'), os.path.join(save_dir, 'labels', 'val')
        create_directory(paths=[self.train_img_dir, self.val_img_dir, self.train_lbl_dir, self.val_lbl_dir], overwrite=True)
        self.map_path = os.path.join(save_dir, 'map.yaml')
        self.coco_data = read_json(x=coco_path)
        check_if_keys_exist_in_dict(data=self.coco_data, key=['categories', 'images', 'annotations'], name=coco_path)
        self.map_dict = {i['id']: i['name'] for i in self.coco_data['categories']}
        map_ids = list(self.map_dict.keys())
        if sorted(map_ids) != list(range(len(map_ids))):
            self.map_id_lk = {}  # old: new
            new_map_dict = {}
            for cnt, v in enumerate(sorted(map_ids)):
                self.map_id_lk[v] = cnt
            for k, v in self.map_id_lk.items():
                new_map_dict[v] = self.map_dict[k]
            self.map_dict = copy(new_map_dict)
        else:
            self.map_id_lk = {k: k for k in self.map_dict.keys()}
        self.map_dict = {v: k for k, v in self.map_dict.items()}
        self.img_file_paths = recursive_file_search(directory=img_dir, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, case_sensitive=False, substrings=None, as_dict=True, raise_error=True)
        self.img_cnt = len(self.coco_data['images'])
        img_idx = list(range(len(self.coco_data['images']) + 1))
        self.train_idx = random.sample(img_idx, int(self.img_cnt * train_size))
        self.verbose, self.coco_path, self.img_dir, self.coco_path = verbose, coco_path, img_dir, coco_path
        self.save_dir, self.greyscale, self.clahe, self.bbox_pad, self.obb = save_dir, greyscale, clahe, bbox_pad, obb

    def run(self):
        shapes, timer = [], SimbaTimer(start=True)
        for cnt in range(len(self.coco_data['images'])):
        #for cnt in range(200):
            img_data = self.coco_data['images'][cnt]
            check_if_keys_exist_in_dict(data=img_data, key=['width', 'height', 'file_name', 'id'], name=self.coco_path)
            _, img_name, ext = get_fn_ext(filepath=img_data['file_name'])
            if self.verbose:
                print(f'Processing annotation {cnt + 1}/{self.img_cnt} from COCO to YOLO ({img_name})...')
            if not img_name in self.img_file_paths.keys():
                raise NoFilesFoundError(msg=f'The file {img_name} could not be found in the {self.img_dir} directory', source=self.__class__.__name__)
            img = read_img(img_path=self.img_file_paths[img_name], greyscale=self.greyscale, clahe=self.clahe)
            if (img.shape[0] != img_data['height']) or (img.shape[1] != img_data['width']):
                raise FaultyTrainingSetError(msg=f'Image {img_name} is of shape {img.shape[0]}x{img.shape[1]}, but the COCO data has been annotated on an image of {img_data["height"]}x{img_data["width"]}.')
            img_annotations = [x for x in self.coco_data['annotations'] if x['image_id'] == img_data['id']]
            roi_str = ''
            if cnt in self.train_idx:
                label_save_path, img_save_path = os.path.join(self.train_lbl_dir, f'{img_name}.txt'), os.path.join(self.train_img_dir, f'{img_name}.png')
            else:
                label_save_path, img_save_path = os.path.join(self.val_lbl_dir, f'{img_name}.txt'), os.path.join(self.val_img_dir, f'{img_name}.png')
            for img_annotation in img_annotations:
                check_if_keys_exist_in_dict(
                    data=img_annotation,
                    key=['bbox', 'keypoints', 'id', 'image_id', 'category_id'],
                    name=str(self.coco_path)
                )

                kps = np.array(img_annotation['keypoints']).reshape(-1, 3).astype(np.int32)
                bbox_kps = kps[kps[:, 2] != 0][:, 0:2]

                if bbox_kps.shape[0] < 2 or kps.shape[0] == 0:
                    continue

                bbox_kps = bbox_kps.reshape(-1, bbox_kps.shape[0], 2).astype(np.int32)

                if not self.obb:
                    bbox_arr = GeometryMixin().keypoints_to_axis_aligned_bounding_box(keypoints=bbox_kps)[0]
                else:
                    if bbox_kps.shape[1] < 3:
                        continue
                    else:
                        poly = Polygon(bbox_kps[0])
                        if self.bbox_pad is not None:
                            shape_stats = GeometryMixin().get_shape_statistics(shapes=poly)
                            buffer = int(min(shape_stats['widths'][0], shape_stats['lengths'][0]) * self.bbox_pad)
                            bbox_arr = np.array(
                                GeometryMixin().minimum_rotated_rectangle(shape=poly, buffer=buffer).exterior.coords
                            ).astype(np.int32)
                        else:
                            bbox_arr = np.array(
                                GeometryMixin().minimum_rotated_rectangle(shape=poly).exterior.coords
                            ).astype(np.int32)

                # Clip to image boundaries
                bbox_arr[:, 0] = np.clip(bbox_arr[:, 0], 0, img.shape[1])  # x
                bbox_arr[:, 1] = np.clip(bbox_arr[:, 1], 0, img.shape[0])  # y

                if not self.obb:
                    h = int(np.max(bbox_arr[:, 1]) - np.min(bbox_arr[:, 1]))
                    w = int(np.max(bbox_arr[:, 0]) - np.min(bbox_arr[:, 0]))
                    if self.bbox_pad is not None:
                        w = int(w + (w * self.bbox_pad))
                        h = int(h + (h * self.bbox_pad))
                    center_w = int(np.mean(bbox_arr[:, 0]))
                    center_h = int(np.mean(bbox_arr[:, 1]))

                    w_ratio = np.clip(w / img.shape[1], 0, 1)
                    h_ratio = np.clip(h / img.shape[0], 0, 1)
                    center_w_ratio = np.clip(center_w / img.shape[1], 0, 1)
                    center_h_ratio = np.clip(center_h / img.shape[0], 0, 1)

                    roi_str += ' '.join([
                        f"{self.map_id_lk[img_annotation['category_id']]}",
                        str(center_w_ratio),
                        str(center_h_ratio),
                        str(w_ratio),
                        str(h_ratio),
                        ' '
                    ])
                else:
                    if bbox_arr.shape[0] >= 4:
                        # First 4 points of the minimum rotated rectangle
                        obb_pts = bbox_arr[:4]
                        norm_obb_pts = [
                            (x / img.shape[1], y / img.shape[0]) for x, y in obb_pts
                        ]
                        roi_str += ' '.join([
                            f"{self.map_id_lk[img_annotation['category_id']]}",
                            *[f"{x:.6f} {y:.6f}" for x, y in norm_obb_pts]
                        ]) + '\n'
            if roi_str:
                with open(label_save_path, mode='wt', encoding='utf-8') as f:
                    f.write(roi_str)
                cv2.imwrite(img_save_path, img)

        create_yolo_yaml(path=self.save_dir, train_path=self.train_img_dir, val_path=self.val_img_dir, names=self.map_dict, save_path=self.map_path)
        timer.stop_timer()
        if self.verbose: stdout_success(msg=f'Labelme to YOLO conversion complete. Data saved in directory {self.save_dir}.', elapsed_time=timer.elapsed_time_str)

if __name__ == "__main__" and not hasattr(sys, 'ps1'):
    parser = argparse.ArgumentParser(description="Convert COCO keypoints json to YOLO training data.")
    parser.add_argument('--coco_path', type=str, required=True, help='Path to JSON file holdeing keypoint annotations in COCO keypoints format.')
    parser.add_argument('--img_dir', type=str, required=True, help='Path to directory holding the images of the annotation in the COCO keypoints file.')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to the directory where to save the YOLO training data.')
    parser.add_argument('--verbose', type=lambda x: str(x).lower() == 'true', default=True, help='Print verbose messages. Use "True" or "False". Default is True')
    parser.add_argument('--obb', type=lambda x: str(x).lower() == 'true', default=False, help='If True, just object oriented bounding boxes. Axis aligned bounding boxes if False')
    parser.add_argument('--greyscale', type=lambda x: str(x).lower() == 'false', default=False, help='Convert images to greyscale. Use "True" or "False". Default is False')
    parser.add_argument('--clahe', type=lambda x: str(x).lower() == 'false', default=False, help='CLAHE enhance images. Use "True" or "False". Default is False')
    parser.add_argument('--train_size', type=float, default=0.7, help='The size of the training set in percent. Default is 0.7 (70%).')
    parser.add_argument('--flip_idx', type=tuple, default=(0, 2, 1, 3, 5, 4, 6, 7, 8), help='The re-ordering of the body-part indexes following a horizontal flip of the image.')
    parser.add_argument('--bbox_pad', type=float, default=None, help='Extra padding for the bounding box in percent (e.g., 0.2) to encompass all body-parts.')


    args = parser.parse_args()
    runner = COCOKeypoints2YoloBbox(coco_path=args.coco_path,
                                    img_dir=args.img_dir,
                                    save_dir=args.save_dir,
                                    clahe=args.clahe,
                                    greyscale=args.greyscale,
                                    verbose=args.verbose,
                                    train_size=args.train_size)
    runner.run()

#
# runner = COCOKeypoints2YoloBbox(coco_path=r"D:\cvat_annotations\frames\coco_keypoints_1\merged\merged_07032025.json",
#                                 img_dir=r"D:\cvat_annotations\frames",
#                                 save_dir=r"D:\cvat_annotations\yolo_07032025\bbox_annot",
#                                 clahe=False,
#                                 bbox_pad=0.2,
#                                 obb=False)
# runner.run()

#
# runner = COCOKeypoints2Yolo(coco_path=r"D:\cvat_annotations\frames\coco_keypoints_1\s1\annotations\s1.json", img_dir=r"D:\cvat_annotations\frames\simon", save_dir=r"D:\cvat_annotations\frames\yolo_keypoints", clahe=True)
# runner.run()
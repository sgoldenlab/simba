import os
import random
from copy import copy
from typing import Optional, Union

import cv2
import numpy as np

from simba.mixins.geometry_mixin import GeometryMixin
from simba.third_party_label_appenders.converters import create_yolo_yaml
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict, check_int,
                                check_valid_boolean)
from simba.utils.enums import Options
from simba.utils.errors import FaultyTrainingSetError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (create_directory, get_fn_ext, read_img,
                                    read_json, recursive_file_search)


class COCOKeypoints2YoloSeg:

    def __init__(self,
                 coco_path: Union[str, os.PathLike],
                 img_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 train_size: float = 0.7,
                 verbose: bool = True,
                 greyscale: bool = False,
                 clahe: bool = False,
                 bbox_pad: Optional[int] = None):

        """
        Convert COCO keypoint annotations to YOLO segmentation format.

        .. image:: _static/img/cocokp_to_yolo_seg.webp
           :width: 600
           :align: center

        :param Union[str, os.PathLike] coco_path: Path to the input COCO JSON annotation file containing 'images', 'annotations', and 'categories'.
        :param Union[str, os.PathLike] img_dir: Directory where the input images referenced in the COCO file are stored.
        :param Union[str, os.PathLike] save_dir: Output directory where YOLO-formatted 'images' and 'labels' folders will be created.
        :param float train_size: Proportion of the dataset to use for training (between 0.1 and 0.99). Remaining goes to validation.
        :param bool verbose: Whether to print progress information during conversion.
        :param bool greyscale: If True, images will be loaded in greyscale.
        :param bool clahe: If True, Contrast Limited Adaptive Histogram Equalization (CLAHE) will be applied to images.
        :param Optional[int] bbox_pad: Optional padding to apply around the bounding box used to generate polygon segmentation.


        :example:
        >>>runner = COCOKeypoints2YoloSeg(coco_path=r"D:\cvat_annotations\frames\coco_keypoints_1\merged\merged_07032025.json",
        >>>                                img_dir=r"D:\cvat_annotations\frames",
        >>>                                save_dir=r"D:\cvat_annotations\yolo_07032025\bbox_seg",
        >>>                                clahe=False,
        >>>                                bbox_pad=None)
        >>>runner.run()
        """

        check_file_exist_and_readable(file_path=coco_path)
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        check_if_dir_exists(in_dir=img_dir, source=f'{self.__class__.__name__} img_dir')
        check_float(name=f'{self.__class__.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
        if bbox_pad is not None: check_int(name=f'{self.__class__.__name__} bbox_pad', value=bbox_pad, min_value=1)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale', raise_error=True)
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe', raise_error=True)
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
        self.save_dir, self.greyscale, self.clahe, self.bbox_pad = save_dir, greyscale, clahe, 1 if bbox_pad is None else bbox_pad

    def run(self):
        timer = SimbaTimer(start=True)
        for cnt in range(len(self.coco_data['images'])):
        #for cnt in range(500):
            img_data = self.coco_data['images'][cnt]
            check_if_keys_exist_in_dict(data=img_data, key=['width', 'height', 'file_name', 'id'], name=self.coco_path)
            _, img_name, ext = get_fn_ext(filepath=img_data['file_name'])
            if self.verbose:
                print(f'Processing annotation {cnt + 1}/{self.img_cnt} from COCO to YOLO ({img_name})...')
            if not img_name in self.img_file_paths.keys():
                raise NoFilesFoundError(msg=f'The file {img_name} could not be found in the {self.img_dir} directory', source=self.__class__.__name__)
            img = read_img(img_path=self.img_file_paths[img_name], greyscale=self.greyscale, clahe=self.clahe)
            img_h, img_w = img.shape[0], img.shape[1]
            if (img.shape[0] != img_data['height']) or (img.shape[1] != img_data['width']):
                raise FaultyTrainingSetError(msg=f'Image {img_name} is of shape {img.shape[0]}x{img.shape[1]}, but the coco data has been annotated on an image of {img_data["height"]}x{img_data["width"]}.')
            img_annotations = [x for x in self.coco_data['annotations'] if x['image_id'] == img_data['id']]
            roi_str = ' '
            if cnt in self.train_idx:
                label_save_path, img_save_path = os.path.join(self.train_lbl_dir, f'{img_name}.txt'), os.path.join(self.train_img_dir, f'{img_name}.png')
            else:
                label_save_path, img_save_path = os.path.join(self.val_lbl_dir, f'{img_name}.txt'), os.path.join(self.val_img_dir, f'{img_name}.png')
            for img_annotation in img_annotations:
                roi_str += '0 '
                check_if_keys_exist_in_dict(data=img_annotation, key=['bbox', 'keypoints', 'id', 'image_id', 'category_id'], name=str(self.coco_path))
                kps = np.array(img_annotation['keypoints']).reshape(-1, 3).astype(np.int32)
                bbox_kps = kps[kps[:, 2] != 0][:, 0:2]
                if bbox_kps.shape[0] < 2 or kps.shape[0] == 0:
                    continue
                bbox_kps = bbox_kps.reshape(-1, bbox_kps.shape[0], 2).astype(np.int32)
                seg_arr = GeometryMixin().bodyparts_to_polygon(data=bbox_kps, parallel_offset=self.bbox_pad)[0]
                seg_arr = np.array(seg_arr.exterior.coords).astype(np.int32)
                pts = np.unique(seg_arr, axis=0)
                center = np.mean(pts, axis=0)
                angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
                seg_arr = pts[np.argsort(angles)]
                seg_arr_x, seg_arr_y = np.clip(seg_arr[:, 0].flatten() / img_w, 0, 1), np.clip(seg_arr[:, 1].flatten() / img_h, 0, 1)
                kps = list(np.column_stack((seg_arr_x, seg_arr_y)).flatten())
                roi_str += ' '.join(str(x) for x in kps) + '\n'
            if roi_str:
                with open(label_save_path, mode='wt', encoding='utf-8') as f:
                    f.write(roi_str)
            cv2.imwrite(img_save_path, img)
        create_yolo_yaml(path=self.save_dir, train_path=self.train_img_dir, val_path=self.val_img_dir, names=self.map_dict, save_path=self.map_path)
        timer.stop_timer()
        if self.verbose: stdout_success(msg=f'Labelme to YOLO conversion complete. Data saved in directory {self.save_dir}.', elapsed_time=timer.elapsed_time_str)

# runner = COCOKeypoints2YoloSeg(coco_path=r"D:\cvat_annotations\frames\coco_keypoints_1\merged\merged_07032025.json",
#                                 img_dir=r"D:\cvat_annotations\frames",
#                                 save_dir=r"D:\cvat_annotations\yolo_07032025\bbox_seg",
#                                 clahe=False,
#                                 bbox_pad=None)
# runner.run()

import os
import random
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.image_mixin import ImageMixin
from simba.third_party_label_appenders.converters import create_yolo_yaml
from simba.third_party_label_appenders.transform.utils import (
    b64_to_arr, create_yolo_keypoint_yaml, get_yolo_keypoint_flip_idx)
from simba.utils.checks import (check_float, check_if_dir_exists,
                                check_if_keys_exist_in_dict, check_int,
                                check_valid_boolean, check_valid_tuple)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (create_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, img_array_to_clahe, read_json)


class LabelmeKeypoints2YoloSeg:


    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 greyscale: Optional[bool] = True,
                 train_size: float = 0.7,
                 padding: Optional[int] = 0,
                 names: Tuple[str, ...] = ('mouse',),
                 clahe: Optional[bool] = True,
                 verbose: Optional[bool] = True):

        """
        Converts LabelMe points into YOLO keypoint format formatted for model training.

        .. seealso:
           To convert Labelme points annotations to YOLO bounding box training format data, see :func:`simba.third_party_label_appenders.transform.labelme_to_yolo.LabelmeBoundingBoxes2YoloBoundingBoxes`.

        .. note::
           For more information on the Labelme annotation tool, see the `Labelme GitHub repository <https://github.com/wkentaro/labelme>`_.
           The Labelme Json files **has too** contain a `imageData` key holding the image as a b64 string.
           For an expected Labelme json format, see `THIS FILE <https://github.com/sgoldenlab/simba/blob/master/misc/labelme_ex.json>`_.

           Only works with one animal (as of 07/25).

        .. image:: _static/img/cocokp_to_yolo_seg.webp
           :width: 600
           :align: center

        :param Union[str, os.PathLike] dlc_dir: Directory path containing labelme JSON files with keypoint annotations.
        :param Union[str, os.PathLike] save_dir: Output directory where YOLO-formatted images, labels, and map YAML file will be saved. Subdirectories `images/train`, `images/val`, `labels/train`, `labels/val` will be created.
        :param float train_size: Proportion of frames randomly assigned to the training dataset. Value must be between 0.1 and 0.99. Default: 0.7.
        :param bool verbose: If True, prints progress. Default: True.
        :param float padding: Fractional padding to add around the bounding boxes (relative to image dimensions). Helps to slightly enlarge bounding boxes by this percentage. Default 0.05. E.g., Useful when all body-parts are along animal length.
        :param Tuple[int, ...] flip_idx: Tuple of keypoint indices used for horizontal flip augmentation during training. The tuple defines the order of keypoints after flipping.
        :param Tuple[str] names: Tuple of animal (class) names. Used for creating the YAML class names mapping file.
        :return: None. Results saved in ``save_dir``.

        :example:
        >>> x = LabelmeKeypoints2YoloSeg(data_path=r"D:\platea\ts_annotations", save_dir=r'D:\platea\yolo_071525')
        >>> x.run()
        """

        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale', raise_error=True)
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe', raise_error=True)
        check_float(name=f'{self.__class__.__name__} train_size', value=train_size, min_value=0.0, max_value=1.0)
        check_valid_tuple(x=names, source=f'{self.__class__.__name__} names', minimum_length=1, valid_dtypes=(str,))
        if padding is not None: check_int(name=f'{self.__class__.__name__} padding', value=padding, min_value=0)
        self.lbls = find_files_of_filetypes_in_directory(directory=data_path, extensions=['.json'], raise_error=True, as_dict=True)
        self.map_path = os.path.join(save_dir, 'map.yaml')
        self.img_dir, self.lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
        self.img_train_dir, self.img_val_dir = os.path.join(self.img_dir, 'train'), os.path.join(self.img_dir, 'val')
        self.lbl_train_dir, self.lbl_val_dir = os.path.join(self.lbl_dir, 'train'), os.path.join(self.lbl_dir, 'val')
        create_directory(paths=[self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lbl_val_dir], overwrite=False)
        self.clahe, self.greyscale, self.train_size, self.padding, self.verbose = clahe, greyscale, train_size, padding, verbose
        self.names = {k: v for k, v in enumerate(names)}
        self.save_dir = save_dir

    def run(self):
        train_idx = random.sample(range(0, len(self.lbls.keys())), int(len(self.lbls.keys()) * self.train_size))
        timer = SimbaTimer(start=True)
        kp_names = []
        for lbl_file_cnt, (lbl_name, lbl_path) in enumerate(self.lbls.items()):
            lbl = read_json(x=lbl_path)
            check_if_keys_exist_in_dict(data=lbl, key=['shapes', 'imageData', 'imagePath'], name=lbl_path)
            file_kp_names = [shape['label'] for shape in lbl['shapes'] if shape['shape_type'] == 'point']
            if len(file_kp_names) > 0:
                missing = [x for x in file_kp_names if x not in kp_names]
                kp_names.extend(missing)
        for lbl_file_cnt, (lbl_name, lbl_path) in enumerate(self.lbls.items()):
            lbl = read_json(x=lbl_path)
            if self.verbose:
                print(f'Processing image {lbl_file_cnt + 1}/{len(self.lbls.keys())}...')
            img_name = get_fn_ext(filepath=lbl['imagePath'])[1]
            img = b64_to_arr(lbl['imageData'])
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.greyscale:
                img = ImageMixin.img_to_greyscale(img=img)
            if self.clahe:
                img = img_array_to_clahe(img=img)
            img_h, img_w = img.shape[:2]
            if lbl_file_cnt in train_idx:
                label_save_path = os.path.join(self.lbl_train_dir, f'{img_name}.txt')
                img_save_path = os.path.join(self.img_train_dir, f'{img_name}.png')
            else:
                label_save_path = os.path.join(self.lbl_val_dir, f'{img_name}.txt')
                img_save_path = os.path.join(self.img_val_dir, f'{img_name}.png')
            bp_arr = np.full(shape=(len(kp_names), 2), fill_value=0, dtype=np.int32)
            for bp_cnt, kp_name in enumerate(kp_names):
                point = [shape['points'] for shape in lbl['shapes'] if shape['label'] == kp_name]
                if len(point) > 0:
                    bp_arr[bp_cnt] = np.array(np.array([point[0][0][0], point[0][0][1]]))
                else:
                    continue
            instance_str = f'0 '
            seg_arr = GeometryMixin().bodyparts_to_polygon(data=bp_arr.reshape(-1, len(bp_arr), 2), parallel_offset=self.padding)[0]
            seg_arr = np.array(seg_arr.exterior.coords).astype(np.int32)
            pts = np.unique(seg_arr, axis=0)
            center = np.mean(pts, axis=0)
            angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
            seg_arr = pts[np.argsort(angles)]
            seg_arr_x, seg_arr_y = np.clip(seg_arr[:, 0].flatten() / img_w, 0, 1), np.clip(seg_arr[:, 1].flatten() / img_h, 0, 1)
            kps = list(np.column_stack((seg_arr_x, seg_arr_y)).flatten())
            instance_str += ' '.join(str(x) for x in kps) + '\n'
            with open(label_save_path, mode='wt', encoding='utf-8') as f:
                f.write(instance_str)
                cv2.imwrite(img_save_path, img)
        create_yolo_yaml(path=self.save_dir, train_path=self.img_train_dir, val_path=self.img_val_dir, names={'Mouse': 0}, save_path=self.map_path)
        timer.stop_timer()
        if self.verbose: stdout_success(msg=f'Labelme to YOLO conversion complete. Data saved in directory {self.save_dir}.', elapsed_time=timer.elapsed_time_str)

        #
        #
        #     # instance_str += keypoint_array_to_yolo_annotation_str(x=bp_arr, img_w=img_w, img_h=img_h, padding=self.padding)
        #     # img_yolo_lbl += instance_str
        #     # with open(label_save_path, mode='wt', encoding='utf-8') as f:
        #     #     f.write(img_yolo_lbl)
        #     # cv2.imwrite(img_save_path, img)
        #
        # create_yolo_keypoint_yaml(path=self.save_dir, train_path=self.img_train_dir, val_path=self.img_val_dir, names=self.names, save_path=self.map_path, kpt_shape=(len(self.flip_idx), 3), flip_idx=self.flip_idx)
        # timer.stop_timer()
        # stdout_success(msg=f'YOLO formated keypoint data saved in {self.save_dir} directory', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)
        #
# x = LabelmeKeypoints2YoloSeg(data_path=r"D:\platea\ts_annotations",
#                              save_dir=r'D:\platea\yolo_071525',
#                              padding=50)
# x.run()
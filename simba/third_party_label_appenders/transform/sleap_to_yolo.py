import json
import os
import random
from typing import Optional, Union

import cv2
import h5py
import numpy as np

from simba.third_party_label_appenders.transform.utils import \
    create_yolo_keypoint_yaml
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict, check_str,
                                check_valid_boolean)
from simba.utils.errors import InvalidInputError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (create_directory,
                                    find_files_of_filetypes_in_directory,
                                    find_video_of_file, get_fn_ext,
                                    read_frm_of_video, read_img)
from simba.utils.yolo import keypoint_array_to_yolo_annotation_str


class SleapAnnotations2Yolo():

    """
    Convert SLEAP annotations to YOLO formatted training data.

    :param Union[str, os.PathLike] data_dir: Directory containing SLEAP annotations `.slp` files
    :param Union[str, os.PathLike] save_dir: Directory to save YOLO-formatted images, labels, and metadata.
    :param bool verbose: If True, print progress during processing.
    :param float train_size: Proportion of frames to assign to the training set (rest go to validation).
    :param bool greyscale: If True, convert images to grayscale.
    :param bool clahe: If True, apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    :param float padding: Relative padding to apply around the bounding box of keypoints (range 0.0 to 1.0).
    :param Optional[str] single_id: Optional custom ID to assign all annotations the same class (used in single-animal datasets).

    :example:
    >>> runner = SleapAnnotations2Yolo(sleap_dir=r'D:\cvat_annotations\frames\slp_to_yolo', save_dir=r'D:\cvat_annotations\frames\slp_to_yolo\yolo')
    >>> runner.run()
    """


    def __init__(self,
                 sleap_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 video_dir: Optional[Union[str, os.PathLike]] = None,
                 padding: Optional[float] = None,
                 train_size: float = 0.8,
                 verbose: bool = True,
                 greyscale: bool = False,
                 clahe: bool = False,
                 single_id: Optional[str] = None):

        check_if_dir_exists(in_dir=os.path.dirname(save_dir), source=f'{self.__class__.__name__} save_path')
        check_if_dir_exists(in_dir=os.path.dirname(sleap_dir), source=f'{self.__class__.__name__} sleap_dir')
        check_float(name=f'{self.__class__.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale', raise_error=True)
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe', raise_error=True)
        if single_id is not None: check_str(name=f'{self.__class__.__name__} single_id', value=single_id, raise_error=True)
        if video_dir is not None: check_if_dir_exists(in_dir=video_dir)
        self.slp_paths = find_files_of_filetypes_in_directory(directory=sleap_dir, extensions=['.slp'], as_dict=True, raise_error=True)
        self.padding, self.train_size, self.verbose, self.save_dir = padding, train_size, verbose, save_dir
        self.clahe, self.greyscale = clahe, greyscale
        self.img_dir, self.lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
        self.img_train_dir, self.img_val_dir = os.path.join(self.img_dir, 'train'), os.path.join(self.img_dir, 'val')
        self.lbl_train_dir, self.lb_val_dir = os.path.join(self.lbl_dir, 'train'), os.path.join(self.lbl_dir, 'val')
        create_directory(paths=[self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lb_val_dir], overwrite=False)
        self.map_path = os.path.join(save_dir, 'map.yaml')
        self.single_id, self.video_dir = single_id, video_dir

    def __h5_to_dict(self, name_or_group):
        if isinstance(name_or_group, h5py.Dataset):
            return name_or_group[()]
        elif isinstance(name_or_group, h5py.Group):
            return {k: self.__h5_to_dict(name_or_group[k]) for k in name_or_group}
        else:
            return None

    def run(self):
        timer = SimbaTimer(start=True)
        lbl_cnt, sleap_data = 0, {}
        for file_cnt, (file_name, file_path) in enumerate(self.slp_paths.items()):
            with h5py.File(file_path, "r") as f:
                data = self.__h5_to_dict(f)
            check_if_keys_exist_in_dict(data=data, key=['frames', 'instances', 'points', 'videos_json'])
            img_names = json.loads(data["videos_json"].item().decode("utf-8"))

            if not 'backend' in img_names.keys() and not 'filename' in img_names.keys():
                raise InvalidInputError(msg='Could not find backend or filename keys in videos_json', source=self.__class__.__name__)
            #check_if_keys_exist_in_dict(data=img_names, key=['filename'])
            sleap_data[file_name] = data

        for file_cnt, (file_name, data) in enumerate(sleap_data.items()):
            lbl_cnt += len(data["frames"])

        train_idx = random.sample(list(range(0, lbl_cnt)), int(lbl_cnt * self.train_size))
        img_cnt, kpt_shapes, names = 0, set(), {}
        for file_cnt, (file_name, file_path) in enumerate(self.slp_paths.items()):
            with h5py.File(file_path, "r") as f:
                data = self.__h5_to_dict(f)
            frames = data["frames"]
            instances = data["instances"]
            points = data["points"]
            img_names, video_path = json.loads(data["videos_json"].item().decode("utf-8")), None
            if 'backend' in img_names.keys():
                video_path = img_names['backend']['filename']
                if not os.path.isfile(video_path):
                    if self.video_dir is None:
                        raise NoFilesFoundError(msg=f'Could not locate file {video_path}, pass a video directory', source=self.__class__.__name__)
                    else:
                        video_path = find_video_of_file(video_dir=self.video_dir, filename=get_fn_ext(filepath=video_path)[1], raise_error=True)
            for i, frame_data in enumerate(frames):
                if self.verbose: print(f'Converting SLEAP annotation to YOLO {img_cnt+1}/{lbl_cnt}...')
                frame_idx = frame_data[0]
                if video_path is None:
                    frm_path = img_names['filename'][frame_idx]
                    file_name = get_fn_ext(filepath=frm_path)[1]
                    check_file_exist_and_readable(file_path=frm_path, raise_error=True)
                    img = read_img(img_path=frm_path, greyscale=self.greyscale, clahe=self.clahe)
                else:
                    img = read_frm_of_video(video_path=video_path, frame_index=frame_idx, greyscale=self.greyscale, clahe=self.clahe)
                img_w, img_h = img.shape[1], img.shape[0]
                frm_instances = [inst for inst in instances if int(inst[2]) == frame_idx]
                if img_cnt in train_idx:
                    img_save_path, lbl_save_path = os.path.join(self.img_train_dir, f'{file_name}.png'), os.path.join(self.lbl_train_dir, f'{file_name}.txt')
                else:
                    img_save_path, lbl_save_path = os.path.join(self.img_val_dir, f'{file_name}.png'), os.path.join(self.lb_val_dir, f'{file_name}.txt')
                img_lbl = ''
                for instance in frm_instances:

                    if instance[1] not in names.keys():
                        names[int(instance[1])] = f'animal_{int(instance[1])}'
                    instance_str = f'{instance[1]} ' if self.single_id is None else '0 '
                    pt_start = int(instance[7])
                    pt_end = int(instance[8])
                    pts = points[pt_start:pt_end]
                    keypoints = np.array([[p[0], p[1]] if p[2] else [0.0, 0.0] for p in pts]).astype(np.int32)
                    if keypoints.shape[0] == 0 or np.all(keypoints == 0):
                        continue
                    keypoints = np.nan_to_num(keypoints, nan=0.0)
                    kpt_shapes.add(keypoints.shape[0])
                    visability_col = np.full((keypoints.shape[0], 1), fill_value=2).flatten()
                    keypoints = np.insert(keypoints, 2, visability_col, axis=1)
                    both_zero = (keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)
                    has_nan_or_inf = ~np.isfinite(keypoints[:, 0]) | ~np.isfinite(keypoints[:, 1])
                    mask = both_zero | has_nan_or_inf
                    keypoints[mask, 2] = 0
                    keypoints[~np.isfinite(keypoints)] = 0
                    instance_str += keypoint_array_to_yolo_annotation_str(x=keypoints, img_w=img_w, img_h=img_h, padding=self.padding)
                    img_lbl += instance_str
                with open(lbl_save_path, mode='wt', encoding='utf-8') as f:
                    f.write(img_lbl)
                cv2.imwrite(img_save_path, img)
                img_cnt+=1

        if len(list(kpt_shapes)) > 1:
            raise InvalidInputError(msg=f'Found more than 1 keypoint shapes: {kpt_shapes}', source=self.__class__.__name__)

        names = names if self.single_id is None else {0: self.single_id}
        create_yolo_keypoint_yaml(path=self.save_dir, train_path=self.img_train_dir, val_path=self.img_val_dir, names=names, save_path=self.map_path, kpt_shape=(list(kpt_shapes)[0], 3), flip_idx=tuple(list(range(0, list(kpt_shapes)[0]))))
        timer.stop_timer()
        if self.verbose:
            stdout_success(msg=f'YOLO annotations saved at {self.save_dir}...', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)

# runner = SleapAnnotations2Yolo(sleap_dir=r'D:\slp_predictions', save_dir=r'D:\slp_predictions\yolo', single_id='mouse')
# runner.run()




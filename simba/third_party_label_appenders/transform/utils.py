from typing import Dict, Iterable, List, Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import base64
import io
import math
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict,
                                check_if_valid_img, check_int,
                                check_valid_array, check_valid_dict,
                                check_valid_lst, check_valid_tuple)
from simba.utils.enums import Formats, Options
from simba.utils.errors import (FaultyTrainingSetError, InvalidInputError,
                                NoFilesFoundError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (copy_files_in_directory, create_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_img, read_json,
                                    recursive_file_search, save_json)
from simba.utils.warnings import DuplicateNamesWarning


def arr_to_b64(x: np.ndarray) -> str:
    """
    Helper to convert image in array format to an image in byte string format
    """

    check_if_valid_img(data=x, source=f'{arr_to_b64.__name__} x')
    _, buffer = cv2.imencode('.jpg', x)
    return base64.b64encode(buffer).decode("utf-8")


def b64_to_arr(img_b64) -> np.ndarray:
    """
    Helper to convert byte string (e.g., created by `labelme <https://github.com/wkentaro/labelme>`__.) to image in numpy array format

    """
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(Image.open(f))
    return img_arr


def scale_pose_img_sizes(pose_data: np.ndarray,
                         imgs: Iterable[Union[np.ndarray, str]],
                         size: Union[Literal['min', 'max'], Tuple[int, int]],
                         interpolation: Optional[int] = cv2.INTER_CUBIC ) -> Tuple[np.ndarray, Iterable[Union[np.ndarray, str]]]:

    """
    Resizes images and scales corresponding pose-estimation data to match the new image sizes.

    .. image:: _static/img/scale_pose_img_sizes.webp
       :width: 400
       :align: center

    :param pose_data: 3d MxNxR array of pose-estimation data where N is number of images, N the number of body-parts in each frame and R represents x,y coordinates of the body-parts.
    :param imgs: Iteralble of images of same size as pose_data M dimension. Can be byte string representation of images, or images as arrays.
    :param size: The target size for the resizing operation. It can be: - `'min'`: Resize all images to the smallest height and width found among the input images. - `'max'`: Resize all images to the largest height and width found among the imgs.
    :param interpolation: Interpolation method to use for resizing. This can be one of OpenCV's interpolation methods.
    :return: The converted pose_data and converted images to align with the new size.
    :rtype: Tuple[np.ndarray, Iterable[Union[np.ndarray, str]]]

    :example:
    >>> df = labelme_to_df(labelme_dir=r'C:\troubleshooting\coco_data\labels\test_read', greyscale=False, pad=False, normalize=False)
    >>> imgs = list(df['image'])
    >>> pose_data = df.drop(['image', 'image_name'], axis=1)
    >>> pose_data_arr = pose_data.values.reshape(len(pose_data), int(len(pose_data.columns) / 2), 2).astype(np.float32)
    >>> new_pose, new_imgs = scale_pose_img_sizes(pose_data=pose_data_arr, imgs=imgs, size=(700, 3000))

    """
    check_valid_array(data=pose_data, source=scale_pose_img_sizes.__name__, accepted_ndims=(3,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_axis_0=1)
    if pose_data.shape[0] != len(imgs):
        raise InvalidInputError(f'The number of images {len(imgs)} and the number of pose-estimated data points {pose_data.shape[0]} do not align.', source=scale_pose_img_sizes.__name__)
    if size == 'min':
        target_h, target_w = np.inf, np.inf
        for v in imgs:
            if isinstance(v, str):
                v = b64_to_arr(v)
            target_h, target_w = min(v.shape[0], target_h), min(v.shape[1], target_w)
    elif size == 'max':
        target_h, target_w = -np.inf, -np.inf
        for v in imgs:
            if isinstance(v, str):
                v = b64_to_arr(v)
            target_h, target_w = max(v.shape[0], target_h), max(v.shape[1], target_w)
    elif isinstance(size, tuple):
        check_valid_tuple(x=size, accepted_lengths=(2,), valid_dtypes=(int,))
        check_int(name=scale_pose_img_sizes.__name__, value=size[0], min_value=1)
        check_int(name=scale_pose_img_sizes.__name__, value=size[1], min_value=1)
        target_h, target_w = size[0], size[1]
    else:
        raise InvalidInputError(msg=f'{size} is not a valid size argument.', source=scale_pose_img_sizes.__name__)
    img_results = []
    pose_results = np.zeros_like(pose_data)
    for img_idx in range(pose_data.shape[0]):
        if isinstance(imgs[img_idx], str):
            img = b64_to_arr(imgs[img_idx])
        else:
            img = imgs[img_idx]
        original_h, original_w = img.shape[0:2]
        scaling_factor_w, scaling_factor_h = target_w / original_w, target_h / original_h
        img = cv2.resize(img, dsize=(target_w, target_h), fx=0, fy=0, interpolation=interpolation)
        if isinstance(imgs[img_idx], str):
            img = arr_to_b64(img)
        img_results.append(img)
        for bp_cnt in range(pose_data[img_idx].shape[0]):
            new_bp_x, new_bp_y = pose_data[img_idx][bp_cnt][0] * scaling_factor_w, pose_data[img_idx][bp_cnt][1] * scaling_factor_h
            pose_results[img_idx][bp_cnt] = np.array([new_bp_x, new_bp_y])
    # out_img = img_results[0]
    # original_image = _b64_to_arr(imgs[0])
    # for i in range(pose_results[0].shape[0]):
    #     new_bp_loc = pose_results[0][i].astype(np.int32)
    #     old_bp_loc = pose_data[0][i].astype(np.int32)
    #     out_img = cv2.circle(out_img, (new_bp_loc[0], new_bp_loc[1]), 10, (0, 0, 255), -1)
    #     original_image = cv2.circle(original_image, (old_bp_loc[0], old_bp_loc[1]), 5, (0, 0, 255), -1)
    # cv2.imshow('asdasdasd', out_img)
    # cv2.imshow('fdghfgth', original_image)
    # cv2.waitKey(120000)

    return (pose_results, img_results)


def normalize_img_dict(img_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Normalize a dictionary of grayscale or RGB images by standardizing pixel intensities.

    :param Dict[str, np.ndarray] img_dict: Dictionary of image arrays with string keys. Each image must be a 2D or 3D NumPy array.
    :return: Dictionary of normalized image arrays, with the same keys as the input.
    :rtype: Dict[str, np.ndarray]
    """

    img_ndims = set()
    for img in img_dict.values():
        check_if_valid_img(data=img, source=normalize_img_dict.__name__, raise_error=True)
        img_ndims.add(img.ndim)
    if len(img_ndims) > 1:
        raise InvalidInputError(msg=f'Images in dictionary have to all be either color OR greyscale. Got {img_ndims} dimensions.', source=normalize_img_dict.__name__)

    results = {}
    if list(img_ndims)[0] == 2:
        all_pixels = np.concatenate([img.ravel() for img in img_dict.values()])
        mean = np.mean(all_pixels)
        std = np.std(all_pixels)
        for img_name, img in img_dict.items():
            v = (img - mean) / std
            v_rescaled = np.clip((v * 64) + 128, 0, 255)
            results[img_name] = v_rescaled.astype(np.uint8)
    else:
        r, g, b = [], [], []
        for img in img_dict.values():
            r.append(np.mean(img[:, :, 0]))
            g.append(np.mean(img[:, :, 1]))
            b.append(np.mean(img[:, :, 2]))
        r_mean, r_std = np.mean(r), np.std(r)
        g_mean, g_std = np.mean(g), np.std(g)
        b_mean, b_std = np.mean(b), np.std(b)
        for img_name, img in img_dict.items():
            r = (img[:, :, 0] - r_mean) / r_std
            g = (img[:, :, 1] - g_mean) / g_std
            b = (img[:, :, 2] - b_mean) / b_std
            r = np.clip((r * 64) + 128, 0, 255)  # Scale and shift
            g = np.clip((g * 64) + 128, 0, 255)  # Scale and shift
            b = np.clip((b * 64) + 128, 0, 255)  # Scale and shift
            results[img_name] = np.stack([r, g, b], axis=-1).astype(np.uint8)

    return results


def create_yolo_keypoint_yaml(path: Union[str, os.PathLike],
                              train_path: Union[str, os.PathLike],
                              val_path: Union[str, os.PathLike],
                              names: Dict[int, str],
                              kpt_shape: Optional[Tuple[int, int]] = None,
                              flip_idx: Optional[Tuple[int, ...]] = None,
                              save_path: Optional[Union[str, os.PathLike]] = None,
                              use_wsl_paths: bool = False) -> Union[None, dict]:
    """
    Given a set of paths to directories, create a model.yaml file for yolo pose model training though ultralytics wrappers.

    .. seealso::
       Used by :func:`simba.sandbox.coco_keypoints_to_yolo.coco_keypoints_to_yolo`

    :param Union[str, os.PathLike] path: Parent directory holding both an images and a labels directory.
    :param Union[str, os.PathLike] train_path: Directory holding training images. For example, if C:\troubleshooting\coco_data\images\train is passed, then a C:\troubleshooting\coco_data\labels\train is expected.
    :param Union[str, os.PathLike] val_path: Directory holding validation images. For example, if C:\troubleshooting\coco_data\images\test is passed, then a C:\troubleshooting\coco_data\labels\test is expected.
    :param Union[str, os.PathLike] test_path: Directory holding test images. For example, if C:\troubleshooting\coco_data\images\validation is passed, then a C:\troubleshooting\coco_data\labels\validation is expected.
    :param Dict[str, int] names: Dictionary mapping pairing object names to object integer identifiers. E.g., {'OBJECT 1': 0, 'OBJECT 2`: 2}
    :param Optional[Tuple[int, ...]] flip_idx: Optional tuple of integers representing keypoint switch indexes if image is flipped horizontally.  Only pass if pose-estimation data.
    :param  Optional[Tuple[int, int]] kpt_shape: Optional tuple of integers representing the shape of each animals keypoints, e.g., (6, 3). Only pass if pose-estimation data.
    :param Union[str, os.PathLike] save_path: Optional location where to save the yolo model yaml file. If None, then the dict is returned.
    :param bool use_wsl_paths: If True, use Windows WSL paths (e.g., `/mnt/...`) in the config file.
    :return None:
    """

    class InlineList(list):
        pass

    def represent_inline_list(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    yaml.add_representer(InlineList, represent_inline_list)
    for p in [path, train_path, val_path]:
        check_if_dir_exists(in_dir=p, source=create_yolo_keypoint_yaml.__name__)
    check_valid_dict(x=names, valid_key_dtypes=(int,), valid_values_dtypes=(str,), min_len_keys=1, source=create_yolo_keypoint_yaml.__name__)
    unique_paths = list({path, train_path, val_path})
    if len(unique_paths) < 3:
        raise InvalidInputError('The passed paths have to be unique.', source=create_yolo_keypoint_yaml.__name__)
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=f'{create_yolo_keypoint_yaml.__name__} save_path')
        if save_path in [path, train_path, val_path]:
            raise InvalidInputError('The save path cannot be the same as the other passed directories.', source=f'{create_yolo_keypoint_yaml.__name__} save_path')

    if flip_idx is not None: check_valid_tuple(x=flip_idx, source=create_yolo_keypoint_yaml.__name__, valid_dtypes=(int,))
    if kpt_shape is not None: check_valid_tuple(x=kpt_shape, source=create_yolo_keypoint_yaml.__name__, valid_dtypes=(int,), accepted_lengths=(2,))

    train_path = os.path.relpath(train_path, path)
    val_path = os.path.relpath(val_path, path)

    data = {'path': path,
            'train': train_path,  # train images (relative to 'path')
            'val': val_path,
            'kpt_shape': InlineList(list(kpt_shape)) if kpt_shape is not None else None,
            'flip_idx': InlineList(list(flip_idx)) if flip_idx is not None else None,
            'names': names}

    if kpt_shape is None: data.pop('kpt_shape', None)
    if flip_idx is None: data.pop('flip_idx', None)

    if save_path is not None:
        with open(save_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    else:
        return data

def get_yolo_keypoint_flip_idx(x: List[str]) -> Tuple[int, ...]:
    """
    Given a list of body-parts, create a ``flip_index`` YOLO yaml entry.

    .. important::
       Only works if the left and right bosy-parts have the substrings ``left`` and ``right`` (case-insensitive).


    :param List[str] x: List of the names of the body-parts. If several animals, then just a list of names for the body-parts for one animal.
    :return: The flip_idx required by the YOLO model yaml file. E.g., [1, 0, 2, 3, 4]
    :rtype: Tuple[int, ...]
    """

    LEFT, RIGHT = 'left', 'right'
    check_valid_lst(data=x, source=f'{get_yolo_keypoint_flip_idx.__name__} x', valid_dtypes=(str,), min_len=1)
    x = [i.lower().strip() for i in x]
    x_left_idx = [i for i, val in enumerate(x) if LEFT in val]
    x_right_idx = [i for i, val in enumerate(x) if RIGHT in val]
    results = []
    for idx in range(len(x)):
        if idx in x_left_idx:
            target_str = x[idx].replace(LEFT, RIGHT)
            if target_str in x:
                target_idx = x.index(target_str)
            else:
                target_idx = idx
        elif idx in x_right_idx:
            target_str = x[idx].replace(RIGHT, LEFT)
            if target_str in x:
                target_idx = x.index(target_str)
            else:
                target_idx = idx
        else:
            target_idx = idx
        results.append(target_idx)
    return tuple(results)

def get_yolo_keypoint_bp_id_idx(animal_bp_dict: Dict[str, Dict[str, List[str]]]) -> Dict[int, List[int]]:
    """
    Helper to create a dictionary holding the indexes for each animals body-parts. USed for transforming data for creating a YOLO training set.

    :param animal_bp_dict: Dictionaru of animal body-parts. Can be created by :func:`simba.mixins.config_reader.ConfigReader.create_body_part_dictionary`.
    :return: Dictionary where the key is the animal name, and the values are the indexes of the columns belonging to each animal.
    :rtype: Dict[int, List[int]
    """


    check_valid_dict(x=animal_bp_dict, valid_key_dtypes=(str,), valid_values_dtypes=(dict,))
    bp_id_idx, bp_cnt = {}, 0
    for animal_cnt, (animal_name, animal_bp_data) in enumerate(animal_bp_dict.items()):
        bp_id_idx[animal_cnt] = list(range(bp_cnt, bp_cnt + len(animal_bp_data['X_bps'])))
        bp_cnt += max(bp_id_idx[animal_cnt]) + 1
    return bp_id_idx



def merge_coco_keypoints_files(data_dir: Union[str, os.PathLike],
                               save_path: Union[str, os.PathLike],
                               max_width: Optional[int] = None,
                               max_height: Optional[int] = None):
    """
    Merges multiple annotation COCO-format keypoint JSON files into a single file.

    .. note::
       Image and annotation entries are appended after adjusting their `id` fields to be unique.

       COCO-format keypoint JSON files can be created using `https://www.cvat.ai/ <https://www.cvat.ai/>`__.

    .. seealso::
       To convert COCO-format keypoint JSON to YOLO training set, see :func:`simba.third_party_label_appenders.transform.coco_keypoints_to_yolo.COCOKeypoints2Yolo`

    :param Union[str, os.PathLike] data_dir: Directory containing multiple COCO keypoints `.json` files to merge.
    :param Union[str, os.PathLike] save_path: File path to save the merged COCO keypoints JSON.
    :param int max_width: Optional max width keypoint coordinate annotation. If above max, the annotation will be set to "not visible"
    :param int max_height: Optional max height keypoint coordinate annotation. If above max, the annotation will be set to "not visible"
    :return: None. Results are saved in ``save_path``.

    :example I:
    >>> DATA_DIR = r'D:\cvat_annotations\frames\coco_keypoints_1\TEST'
    >>> SAVE_PATH = r"D:\cvat_annotations\frames\coco_keypoints_1\TEST\merged.json"
    >>> merge_coco_keypoints_files(data_dir=DATA_DIR, save_path=SAVE_PATH)


    :example II:
    >>> merge_coco_keypoints_files(data_dir=DATA_DIR, save_path=SAVE_PATH, max_width=662, max_height=217)
    """

    timer = SimbaTimer(start=True)
    data_files = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.json'], raise_error=True, raise_warning=False, as_dict=True)
    if os.path.isdir(save_path):
        raise InvalidInputError(msg=f'save_path has to be a filepath, not a directory.', source=f'{merge_coco_keypoints_files.__name__} save_path')
    check_if_dir_exists(in_dir=os.path.dirname(save_path))
    if max_width is not None:
        check_int(name=f'{merge_coco_keypoints_files.__name__} max_width', value=max_width, min_value=1, raise_error=True)
    else:
        max_width = math.inf
    if max_height is not None:
        check_int(name=f'{merge_coco_keypoints_files.__name__} max_height', value=max_height, min_value=1, raise_error=True)
    else:
        max_height = math.inf
    results, max_image_id, max_annotation_id = None, 0, 0
    data_file_cnt, img_names = len(data_files), []
    if data_file_cnt == 1:
        raise InvalidInputError(msg=f'Only 1 JSON file found in {data_dir} directory. Cannot merge a single file.', source=merge_coco_keypoints_files.__name__)

    for file_cnt, (file_name, file_path) in enumerate(data_files.items()):
        print(f'Processing {file_cnt + 1}/{data_file_cnt} ({file_name})...')
        coco_data = read_json(file_path)
        check_if_keys_exist_in_dict(data=coco_data, key=['licenses', 'info', 'categories', 'images', 'annotations'], name=file_name)
        if file_cnt == 0:
            results = deepcopy(coco_data)
            max_image_id = max((img['id'] for img in results['images']), default=0)
            max_annotation_id = max((ann['id'] for ann in results['annotations']), default=0)
            for img in coco_data['images']:
                img_names.append(img['file_name'])
        else:
            if coco_data.get('licenses'):
                for lic in coco_data['licenses']:
                    if lic not in results['licenses']:
                        results['licenses'].append(lic)

            if coco_data.get('categories'):
                for cat in coco_data['categories']:
                    if cat not in results['categories']:
                        results['categories'].append(cat)

            id_mapping = {}
            new_images = []
            for img in coco_data['images']:
                new_id = img['id'] + max_image_id + 1
                id_mapping[img['id']] = new_id
                img['id'] = new_id
                new_images.append(img)
                img_names.append(img['file_name'])
            results['images'].extend(new_images)
            new_annotations = []
            for ann in coco_data['annotations']:
                ann['id'] += max_annotation_id + 1
                ann['image_id'] = id_mapping.get(ann['image_id'], ann['image_id'])
                new_annotations.append(ann)
            results['annotations'].extend(new_annotations)
            for annotation_cnt, annotation in enumerate(results['annotations']):
                x_kp, y_kp, p_kp = annotation['keypoints'][::3], annotation['keypoints'][1::3], annotation['keypoints'][2::3]
                x_kp = [min(max(x, 0), max_width) for x in x_kp]
                y_kp = [min(max(x, 0), max_height) for x in y_kp]
                new_keypoints = [int(item) for trio in zip(x_kp, y_kp, p_kp) for item in trio]
                results['annotations'][annotation_cnt]['keypoints'] = new_keypoints
            max_image_id = max((img['id'] for img in results['images']), default=max_image_id)
            max_annotation_id = max((ann['id'] for ann in results['annotations']), default=max_annotation_id)

    duplicates = [item for item, count in Counter(img_names).items() if count > 1]
    if len(duplicates) > 0:
        DuplicateNamesWarning(msg=f'{len(duplicates)} annotated file names have the same name: {duplicates}', source=merge_coco_keypoints_files.__name__)

    #PRINT THE NUMBER OF TOTAL ANNOTATIONS TODO

    timer.stop_timer()
    save_json(data=results, filepath=save_path)
    stdout_success(msg=f'Merged COCO key-points file (from {data_file_cnt} input files) saved at {save_path}', source=merge_coco_keypoints_files.__name__, elapsed_time=timer.elapsed_time_str)


def check_valid_yolo_map(yolo_map: Union[str, os.PathLike]) -> None:
    """
    Helper to do surface check if yaml path leads to a valid yolo map file for pose-estimation.
    """

    REQUIRED_KEYS = ['path', 'train', 'val', 'kpt_shape', 'flip_idx', 'names']
    check_file_exist_and_readable(file_path=yolo_map, raise_error=True)
    with open(yolo_map, "r") as f: yolo_map = yaml.safe_load(f)
    check_if_keys_exist_in_dict(data=yolo_map, key=REQUIRED_KEYS)

    path = yolo_map['path']
    img_train_dir = str(os.path.join(yolo_map['path'], yolo_map['train']))
    img_val_dir = str(os.path.join(yolo_map['path'], yolo_map['val']))
    lbl_train_dir = os.path.join(yolo_map['path'], 'labels', 'train')
    lbl_val_dir = os.path.join(yolo_map['path'], 'labels', 'val')

    check_if_dir_exists(in_dir=path, source=f'{yolo_map} path', raise_error=True)
    check_if_dir_exists(in_dir=img_train_dir, source=f'{img_train_dir} yolo_map {yolo_map}', raise_error=True)
    check_if_dir_exists(in_dir=img_val_dir, source=f'{img_val_dir} yolo_map {yolo_map}',raise_error=True)
    check_if_dir_exists(in_dir=lbl_train_dir, source=f'{lbl_train_dir} yolo_map {yolo_map}', raise_error=True)
    check_if_dir_exists(in_dir=lbl_val_dir, source=f'{img_val_dir} yolo_map {yolo_map}', raise_error=True)
    _ = find_files_of_filetypes_in_directory(directory=img_train_dir, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, raise_error=True)
    _ = find_files_of_filetypes_in_directory(directory=img_val_dir, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, raise_error=True)
    _ = find_files_of_filetypes_in_directory(directory=lbl_train_dir, extensions=['.txt'], raise_error=True)
    _ = find_files_of_filetypes_in_directory(directory=lbl_val_dir, extensions=['.txt'], raise_error=True)

def downsample_coco_dataset(json_path: Union[str, os.PathLike],
                            img_dir: Union[str, os.PathLike],
                            save_dir: Union[str, os.PathLike],
                            shrink_factor: int = 4,
                            verbose: bool = True):

    """
    Downsample a COCO-format dataset (images and annotations) by a fixed integer factor.

    This function resizes all images and updates annotation coordinates accordingly.
    Bounding box coordinates and keypoints (x, y only) are scaled by `shrink_factor`,
    while visibility flags in keypoints remain unchanged. The updated dataset is saved
    in COCO format to `save_dir`.

    :param Union[str, os.PathLike] json_path: Path to the input COCO JSON annotation file.
    :param Union[str, os.PathLike] img_dir: Directory containing the original images referenced in the JSON file.
    :param Union[str, os.PathLike] save_dir: Directory where resized images and updated COCO JSON will be stored.
    :param int shrink_factor: Factor by which to downsample both images and annotation coordinates. Must be >= 2. Default is 4.
    :param bool verbose: If True, prints progress information during processing. Default is True.
    :return None: Saves new images and updated COCO JSON to `save_dir`.

    :example:
    >>> downsample_coco_dataset(
    ...     json_path=r"D:\\cvat_annotations\\frames\\coco_keypoints_1\\merged\\merged_08132025.json",
    ...     img_dir=r"D:\\cvat_annotations\\frames\\all_imgs_071325",
    ...     save_dir=r"D:\\cvat_annotations\\frames\\resampled_coco_081225"
    ... )
    """

    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=json_path)
    check_if_dir_exists(in_dir=img_dir)
    check_if_dir_exists(in_dir=save_dir)
    check_int(name=f'{downsample_coco_dataset.__name__} shrink_factor', value=shrink_factor, min_value=2, raise_error=True)
    img_paths = find_files_of_filetypes_in_directory(directory=img_dir, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, raise_error=True, as_dict=True)
    coco_data = read_json(json_path)
    check_if_keys_exist_in_dict(data=coco_data, key=['licenses', 'info', 'categories', 'images', 'annotations'], name=json_path)
    out_coco = {'licenses': coco_data['licenses'], 'info': coco_data['info'], 'categories': coco_data['categories'], 'images': [], 'annotations': []}
    _, json_name, _ = get_fn_ext(filepath=json_path)
    out_coco_path = os.path.join(save_dir, f'{json_name}.json')
    img_cnt = len(coco_data['images'])

    for cnt in range(img_cnt):
        if verbose: print(f'Processing COCO image {cnt+1}/{img_cnt} (shrink factor: {shrink_factor})...')
        img_data = coco_data['images'][cnt]
        new_img_data = deepcopy(img_data)
        check_if_keys_exist_in_dict(data=img_data, key=['width', 'height', 'file_name', 'id'], name=json_path)
        _, img_name, img_ext = get_fn_ext(filepath=img_data['file_name'])
        if not img_name in img_paths.keys():
            raise NoFilesFoundError(msg=f'The file {img_name} could not be found in the {img_dir} directory', source=downsample_coco_dataset.__name__)
        img = read_img(img_path=img_paths[img_name], greyscale=False, clahe=False)
        if (img.shape[0] != img_data['height']) or (img.shape[1] != img_data['width']):
            raise FaultyTrainingSetError(msg=f'Image {img_name} is of shape {img.shape[0]}x{img.shape[1]}, but the coco data has been annotated on an image of {img_data["height"]}x{img_data["width"]}.')
        new_img = img[::shrink_factor, ::shrink_factor, :]
        new_img_data['width'], new_img_data['height'] = new_img.shape[1], new_img.shape[0]
        out_coco['images'].append(new_img_data)
        img_annotations = [x for x in coco_data['annotations'] if x['image_id'] == img_data['id']]
        for img_annotation in img_annotations:
            check_if_keys_exist_in_dict(data=img_annotation, key=['bbox', 'keypoints', 'id', 'image_id', 'category_id'], name=json_path)
            new_img_annotation = deepcopy(img_annotation)
            new_img_annotation['bbox'] = ([int(v / shrink_factor) for v in img_annotation['bbox']])
            new_img_annotation['keypoints'] = [int(v / shrink_factor) if (i % 3 != 2) else v for i, v in enumerate(img_annotation['keypoints'])]
            out_coco['annotations'].append(new_img_annotation)
        img_save_path = os.path.join(save_dir, f'{img_name}{img_ext}')
        cv2.imwrite(filename=img_save_path, img=new_img)

    save_json(data=out_coco, filepath=out_coco_path)
    timer.stop_timer()
    if verbose: print(f'New COCO data stored in {save_dir} (elapsed time: {timer.elapsed_time_str}s)')

def concatenate_dlc_annotations(data_dir: Union[str, os.PathLike], save_dir: Union[str, os.PathLike], annotator: str = 'SN'):
    """
    Concatenate DeepLabCut annotation files from multiple directories into a single CSV file.

    This function searches for DeepLabCut 'CollectedData_*.csv' files in the specified data directory,
    processes each file to standardize frame naming conventions, copies associated PNG images,
    and combines all annotation data into a single CSV file with multi-index headers.

    :param Union[str, os.PathLike] data_dir: Path to directory containing DeepLabCut annotation subdirectories.
    :param Union[str, os.PathLike] save_dir: Path to directory where concatenated results will be saved.
    :param str annotator: Name of the annotator (default: 'SN'). Used in the output filename 'CollectedData_{annotator}.csv'.

    :return: None. Creates concatenated CSV file and copies PNG images to 'labeled-data' subdirectory in save_dir.

    :example:
    >>> concatenate_dlc_annotations(
    ...     data_dir='/path/to/dlc/annotations',
    ...     save_dir='/path/to/output',
    ...     annotator='John'
    ... )

    >>> concatenate_dlc_annotations(data_dir=r'E:\crim13_imgs\CRIM_labelled_images', save_dir=r'E:\crim13_imgs\combined')
    >>> concatenate_dlc_annotations(data_dir=r'E:\rgb_white_vs_black_imgs\GB_labelled_images.zip\labeled-data', save_dir=r'E:\rgb_white_vs_black_imgs\combined')

    """

    check_if_dir_exists(in_dir=data_dir, source=concatenate_dlc_annotations.__name__, raise_error=True)
    check_if_dir_exists(in_dir=save_dir, source=concatenate_dlc_annotations.__name__, raise_error=True)
    out_dir = os.path.join(f'{save_dir}', 'labeled-data')
    df_destination = os.path.join(out_dir, f'CollectedData_{annotator}.csv')
    create_directory(out_dir)

    df_results = []
    timer = SimbaTimer(start=True)
    annotation_paths = recursive_file_search(directory=data_dir, extensions=Formats.CSV.value, case_sensitive=True, substrings='CollectedData', raise_error=True, as_dict=False)
    for file_cnt, annotation_path in enumerate(annotation_paths):
        df = pd.read_csv(annotation_path, header=[0, 1, 2])
        video_name = Path(annotation_path).parent.name
        df.iloc[:, 0] = df.iloc[:, 0].str.rsplit("\\", n=1).str.join("_")
        copy_files_in_directory(in_dir=os.path.dirname(annotation_path), out_dir=out_dir, raise_error=True, filetype='png', prefix=f'{video_name}_', verbose=True)
        df_results.append(df)
        print(f'File {file_cnt+1}/{len(annotation_paths)} complete...')
    df_results = pd.concat(df_results, axis=0)
    df_results.to_csv(df_destination, index=False)
    timer.stop_timer()
    print(f'DLC annotation concatenated (elapsed time: {timer.elapsed_time_str}s)')



#concatenate_dlc_annotations(data_dir=r'E:\rgb_white_vs_black_imgs\GB_labelled_images.zip\labeled-data', save_dir=r'E:\rgb_white_vs_black_imgs\combined')

#concatenate_dlc_annotations(data_dir=r'E:\crim13_imgs\CRIM_labelled_images', save_dir=r'E:\crim13_imgs\combined')
#merge_coco_keypoints_files(data_dir=r'E:\netholabs_videos\3d\cvat_annotations', save_path=r'E:\netholabs_videos\3d\cvat_annotations\3d_merged.json')




    #merge_coco_keypoints_files(data_dir=r'E:\netholabs_videos\50_largest_files\imgs_to_lbl', save_path=r'E:\netholabs_videos\50_largest_files\imgs_to_lbl\merged.json', max_width=662, max_height=217)
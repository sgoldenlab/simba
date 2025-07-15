import base64
import io
import itertools
import json
import os
import random
from copy import copy, deepcopy
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from PIL import Image

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2
import numpy as np
import yaml
#from pycocotools import mask
from shapely.geometry import Polygon

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.image_mixin import ImageMixin
from simba.mixins.pose_importer_mixin import PoseImporterMixin
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict,
                                check_if_valid_img, check_instance, check_int,
                                check_str, check_valid_array,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_dict, check_valid_lst,
                                check_valid_tuple,
                                check_video_and_data_frm_count_align)
from simba.utils.enums import Formats, Options
from simba.utils.errors import (FaultyTrainingSetError, InvalidInputError,
                                NoFilesFoundError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (copy_files_to_directory, create_directory,
                                    find_files_of_filetypes_in_directory,
                                    find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_df,
                                    read_frm_of_video, read_img, read_json,
                                    read_roi_data, recursive_file_search,
                                    save_json, write_pickle)
from simba.utils.warnings import FrameRangeWarning, NoDataFoundWarning

# def geometry_to_rle(geometry: Union[np.ndarray, Polygon], img_size: Tuple[int, int]):
#     """
#     Converts a geometry (polygon or NumPy array) into a Run-Length Encoding (RLE) mask, suitable for object detection or segmentation tasks.
#
#     :param geometry: The geometry to be converted into an RLE. It can be either a shapely Polygon or a (n, 2) np.ndarray with vertices.
#     :param img_size:  A tuple `(height, width)` representing the size of the image in which the geometry is to be encoded. This defines the dimensions of the output binary mask.
#     :return:
#     """
#     check_instance(source=geometry_to_rle.__name__, instance=geometry, accepted_types=(Polygon, np.ndarray))
#     if isinstance(geometry, (Polygon,)):
#         geometry = geometry.exterior.coords
#     else:
#         check_valid_array(data=geometry, source=geometry_to_rle.__name__, accepted_ndims=[(2,)], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
#     binary_mask = np.zeros(img_size, dtype=np.uint8)
#     rr, cc = polygon(geometry[:, 0].flatten(), geometry[:, 1].flatten(), img_size)
#     binary_mask[rr, cc] = 1
#     rle = mask.encode(np.asfortranarray(binary_mask))
#     rle['counts'] = rle['counts'].decode('utf-8')
#     return rle

def geometries_to_coco(geometries: Dict[str, np.ndarray],
                       video_path: Union[str, os.PathLike],
                       save_dir: Union[str, os.PathLike],
                       version: Optional[int] = 1,
                       description: Optional[str] = None,
                       licences: Optional[str] = None):
    """
    Convert a dictionary of geometries (keypoints or polygons) into COCO format annotations and save images
    extracted from a video to a specified directory.

    This function takes a dictionary of geometries (e.g., keypoints, bounding boxes, or polygons) and converts
    them into COCO format annotations. The geometries are associated with frames of a video, and the corresponding
    images are extracted from the video, saved as PNG files, and linked to the annotations.

    :example:
    >>> data_path = r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\FRR_gq_Saline_0624.csv"
    >>> animal_data = read_df(file_path=data_path, file_type='csv', usecols=['Nose_x', 'Nose_y', 'Tail_base_x', 'Tail_base_y', 'Left_side_x', 'Left_side_y', 'Right_side_x', 'Right_side_y']).values.reshape(-1, 4, 2)[0:20].astype(np.int32)
    >>> animal_polygons = GeometryMixin().bodyparts_to_polygon(data=animal_data)
    >>> animal_polygons = GeometryMixin().multiframe_minimum_rotated_rectangle(shapes=animal_polygons)
    >>> animal_polygons = GeometryMixin().geometries_to_exterior_keypoints(geometries=animal_polygons)
    >>> animal_polygons = GeometryMixin.keypoints_to_axis_aligned_bounding_box(keypoints=animal_polygons)
    >>> animal_polygons = {0: animal_polygons}
    >>> geometries_to_coco(geometries=animal_polygons, video_path=r'C:\troubleshooting\mitra\project_folder\videos\FRR_gq_Saline_0624.mp4', save_dir=r"C:\troubleshooting\coco_data")
    """

    categories = []
    for cnt, i in enumerate(geometries.keys()): categories.append({'id': i, 'name': i, 'supercategory': i})
    results = {'info': {'year': datetime.now().year, 'version': version, 'description': description}, 'licences': licences, 'categories': categories}
    video_data = get_video_meta_data(video_path)
    w, h = video_data['width'], video_data['height']
    images = []
    annotations = []
    img_names = []
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    save_img_dir = os.path.join(save_dir, 'img')
    if not os.path.isdir(save_img_dir): os.makedirs(save_img_dir)
    for category_cnt, (category_id, category_data) in enumerate(geometries.items()):
        for img_cnt in range(category_data.shape[0]):
            img_geometry = category_data[img_cnt]
            img_name = f'{video_data["video_name"]}_{img_cnt}.png'
            if img_name not in img_names:
                images.append({'id': img_cnt, 'width': w, 'height': h, 'file_name': img_name})
                img = read_frm_of_video(video_path=video_path, frame_index=img_cnt)
                img_save_path = os.path.join(save_img_dir, img_name)
                cv2.imwrite(img_save_path, img)
                img_names.append(img_name)
            annotation_id = category_cnt * img_cnt + 1
            d = GeometryMixin().get_shape_lengths_widths(shapes=Polygon(img_geometry))
            a_h, a_w, a_a = d['max_length'], d['max_width'], d['max_area']
            bbox = [int(category_data[img_cnt][0][0]), int(category_data[img_cnt][0][1]), int(a_w), int(a_h)]
            rle = geometry_to_rle(geometry=img_geometry, img_size=(h, w))
            annotation = {'id': annotation_id, 'image_id': img_cnt, 'category_id': category_id, 'bbox': bbox, 'area': a_a, 'iscrowd': 0, 'segmentation': rle}
            annotations.append(annotation)
    results['images'] = images
    results['annotations'] = annotations
    with open(os.path.join(save_dir, f"annotations.json"), "w") as final:
        json.dump(results, final)


def geometries_to_yolo(geometries: Dict[Union[str, int], np.ndarray],
                       video_path: Union[str, os.PathLike],
                       save_dir: Union[str, os.PathLike],
                       verbose: Optional[bool] = True,
                       sample: Optional[int] = None,
                       obb: Optional[bool] = False,
                       map: Optional[Dict[int, str]] = None) -> None:
    """
    Converts geometrical shapes (like polygons) into YOLO format annotations and saves them along with corresponding video frames as images.

    :param Dict[Union[str, int], np.ndarray geometries: A dictionary where the keys represent category IDs (either string or int), and the values are NumPy arrays of shape `(n_frames, n_points, 2)`. Each entry in the array represents the geometry of an object in a particular frame (e.g., keypoints or polygons).
    :param Union[str, os.PathLike] video_path: Path to the video file from which frames are extracted. The video is used to extract images corresponding to the geometrical annotations.
    :param Union[str, os.PathLike] save_dir: The directory where the output images and YOLO annotation files will be saved. Images will be stored in a subfolder `images/` and annotations in `labels/`.
    :param Optional[bool] verbose: If `True`, prints progress while processing each frame. This can be useful for monitoring long-running tasks. Default is `True`.
    :param Optional[int] sample: If provided, only a random sample of the geometries will be used for annotation. This value represents the number of frames to sample.  If `None`, all frames will be processed. Default is `None`.
    :param Optional[bool] obb: If `True`, uses oriented bounding boxes (OBB) by extracting the four corner points of the geometries. Otherwise, axis-aligned bounding boxes (AABB) are used. Default is `False`.
    :param Optional[Dict[int, str]] map: If `True`, uses oriented bounding boxes (OBB) by extracting the four corner points of the geometries. Otherwise, axis-aligned bounding boxes (AABB) are used. Default is `False`.
    :return None:

    :example:
    >>> data_path = r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\501_MA142_Gi_CNO_0514.csv"
    >>> animal_data = read_df(file_path=data_path, file_type='csv', usecols=['Nose_x', 'Nose_y', 'Tail_base_x', 'Tail_base_y', 'Left_side_x', 'Left_side_y', 'Right_side_x', 'Right_side_y']).values.reshape(-1, 4, 2).astype(np.int32)
    >>> animal_polygons = GeometryMixin().bodyparts_to_polygon(data=animal_data)
    >>> polygons = GeometryMixin().multiframe_minimum_rotated_rectangle(shapes=animal_polygons)
    >>> animal_polygons = GeometryMixin().geometries_to_exterior_keypoints(geometries=polygons)
    >>> animal_polygons = {0: animal_polygons}
    >>> geometries_to_yolo(geometries=animal_polygons, video_path=r'C:\troubleshooting\mitra\project_folder\videos\501_MA142_Gi_CNO_0514.mp4', save_dir=r"C:\troubleshooting\coco_data", sample=500, obb=True)
    """


    timer = SimbaTimer(start=True)
    video_data = get_video_meta_data(video_path)
    categories = list(geometries.keys())
    w, h = video_data['width'], video_data['height']
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    save_img_dir = os.path.join(save_dir, 'images')
    save_labels_dir = os.path.join(save_dir, 'labels')
    if not os.path.isdir(save_img_dir): os.makedirs(save_img_dir)
    if not os.path.isdir(save_labels_dir): os.makedirs(save_labels_dir)
    results, samples = {}, None
    if sample is not None:
        check_int(name='sample', value=sample, min_value=1, max_value=geometries[categories[0]].shape[0])
        samples = np.random.choice(np.arange(0, geometries[categories[0]].shape[0]-1), sample)
    if map is not None:
        check_valid_dict(x=map, valid_key_dtypes=(str,), valid_values_dtypes=(int,), min_len_keys=1)
    category_ids, lbl_cnt = set(), 0
    for category_cnt, (category_id, category_data) in enumerate(geometries.items()):
        category_ids.add(category_id)
        for img_cnt in range(category_data.shape[0]):
            if sample is not None and img_cnt not in samples:
                continue
            else:
                if verbose:
                    print(f'Writing category {category_cnt}, Image: {img_cnt} ({video_data["video_name"]})')
                img_geometry = category_data[img_cnt]
                img_name = f'{video_data["video_name"]}_{img_cnt}.png'
                if not obb:
                    shape_stats = GeometryMixin.get_shape_statistics(shapes=Polygon(img_geometry))
                    x_center = shape_stats['centers'][0][0] / w
                    y_center = shape_stats['centers'][0][1] / h
                    width = shape_stats['widths'][0] / w
                    height = shape_stats['lengths'][0] / h
                    img_results = ' '.join([str(category_id), str(x_center), str(y_center), str(width), str(height)])
                else:
                    img_geometry = img_geometry[1:]
                    x1, y1 = img_geometry[0][0] / w, img_geometry[0][1] / h
                    x2, y2 = img_geometry[1][0] / w, img_geometry[1][1] / h
                    x3, y3 = img_geometry[2][0] / w, img_geometry[2][1] / h
                    x4, y4 = img_geometry[3][0] / w, img_geometry[3][1] / h
                    img_results = ' '.join([str(category_id), str(x1), str(y1), str(x2), str(y2), str(x3), str(y3), str(x4), str(y4)])
                if img_name not in results.keys():
                    img = read_frm_of_video(video_path=video_path, frame_index=img_cnt)
                    img_save_path = os.path.join(save_img_dir, img_name)
                    cv2.imwrite(img_save_path, img)
                    results[img_name] = [img_results]
                else:
                    results[img_name].append(img_results)
                lbl_cnt =+ 1

    for k, v in results.items():
        name = k.split(sep='.', maxsplit=2)[0]
        file_name = os.path.join(save_labels_dir, f'{name}.txt')
        with open(file_name, mode='wt', encoding='utf-8') as f:
            f.write('\n'.join(v))

    if map is None:
        map = {}
        for cnt, i in enumerate(list(category_ids)):
            map[f'Animal_{cnt+1}'] = i
    write_pickle(data=map, save_path=os.path.join(save_dir, 'map.pickle'))
    timer.stop_timer()
    if verbose:
        stdout_success(msg=f'{lbl_cnt} yolo labels saved in {save_dir}', elapsed_time=timer.elapsed_time_str)


def b64_to_arr(img_b64) -> np.ndarray:
    """
    Helper to convert byte string (e.g., created by `labelme <https://github.com/wkentaro/labelme>`__.) to image in numpy array format

    """
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(Image.open(f))
    return img_arr


def arr_to_b64(x: np.ndarray) -> str:
    """
    Helper to convert image in array format to an image in byte string format
    """
    _, buffer = cv2.imencode('.jpg', x)
    return base64.b64encode(buffer).decode("utf-8")

def create_yolo_yaml(path: Union[str, os.PathLike],
                     train_path: Union[str, os.PathLike],
                     val_path: Union[str, os.PathLike],
                     names: Dict[str, int],
                     save_path: Optional[Union[str, os.PathLike]] = None,
                     test_path: Optional[Union[str, os.PathLike]] = None,
                     reverse_ids: Optional[bool] = True) -> Union[None, dict]:
    """
    Given a set of paths to directories, create a model.yaml file for model training though ultralytics wrappers.

    :param Union[str, os.PathLike] path: Parent directory holding both an images and a labels directory.
    :param Union[str, os.PathLike] train_path: Directory holding training images. For example, if C:\troubleshooting\coco_data\images\train is passed, then a C:\troubleshooting\coco_data\labels\train is expected.
    :param Union[str, os.PathLike] val_path: Directory holding validation images. For example, if C:\troubleshooting\coco_data\images\test is passed, then a C:\troubleshooting\coco_data\labels\test is expected.
    :param Union[str, os.PathLike] test_path: Directory holding test images. For example, if C:\troubleshooting\coco_data\images\validation is passed, then a C:\troubleshooting\coco_data\labels\validation is expected.
    :param Dict[str, int] names: Dictionary mapping pairing object names to object integer identifiers. E.g., {'OBJECT 1': 0, 'OBJECT 2`: 2}
    :param Union[str, os.PathLike] save_path: Optional location where to save the yolo model yaml file. If None, then the dict is returned.
    :return None:
    """

    path_to_check = [path, train_path, val_path]
    if test_path is not None: path_to_check.append(test_path)

    for p in path_to_check:
        check_if_dir_exists(in_dir=p, source=create_yolo_yaml.__name__)
    check_valid_dict(x=names, valid_key_dtypes=(str,), valid_values_dtypes=(int,), min_len_keys=1)
    reversed_names = {v: k for k, v in names.items()}

    unique_paths = list(set(path_to_check))
    if len(unique_paths) < len(path_to_check):
        raise InvalidInputError('The passed paths have to be unique.', source=create_yolo_yaml.__name__)
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=f'{create_yolo_yaml.__name__} save_path')
        if save_path in [path, train_path, val_path, test_path]:
            raise InvalidInputError('The save path cannot be the same as the other passed directories.', source=f'{create_yolo_yaml.__name__} save_path')

    train_path = os.path.relpath(train_path, path)
    val_path = os.path.relpath(val_path, path)

    data = {'path': path,
            'train': train_path,  # train images (relative to 'path')
            'val': val_path,  # val images (relative to 'path')
            'names': reversed_names}

    if test_path is not None:
        test_path = os.path.relpath(test_path, path)
        data['test'] = test_path

    if save_path is not None:
        with open(save_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
    else:
        return data


def labelme_to_dlc(labelme_dir: Union[str, os.PathLike],
                   scorer: Optional[str] = 'SN',
                   save_dir: Optional[Union[str, os.PathLike]] = None) -> None:
    """
    Convert labels from labelme format to DLC format.


    .. note::
        See `labelme GitHub repo <https://github.com/wkentaro/labelme>`__.

    :param Union[str, os.PathLike] labelme_dir: Directory with labelme json files.
    :param Optional[str] scorer: Name of the scorer (anticipated by DLC as header)
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where to save the DLC annotations. If None, then same directory as labelme_dir with `_dlc_annotations` suffix.
    :return: None

    :example:
    >>> labelme_dir = 'D:\ts_annotations'
    >>> labelme_to_dlc(labelme_dir=labelme_dir)
    """

    check_if_dir_exists(in_dir=labelme_dir)
    annotation_paths = find_files_of_filetypes_in_directory(directory=labelme_dir, extensions=['.json'], raise_error=True)
    results_dict = {}
    images = {}
    for annot_path in annotation_paths:
        with open(annot_path) as f:
            annot_data = json.load(f)
        check_if_keys_exist_in_dict(data=annot_data, key=['shapes', 'imageData', 'imagePath'], name=annot_path)
        img_name = os.path.basename(annot_data['imagePath'])
        images[img_name] = b64_to_arr(annot_data['imageData'])
        for bp_data in annot_data['shapes']:
            check_if_keys_exist_in_dict(data=bp_data, key=['label', 'points'], name=annot_path)
            point_x, point_y = bp_data['points'][0][0], bp_data['points'][0][1]
            lbl = bp_data['label']
            id = os.path.join('labeled-data', os.path.basename(labelme_dir), img_name)
            if id not in results_dict.keys():
                results_dict[id] = {f'{lbl}': {'x': point_x, 'y': point_y}}
            else:
                results_dict[id].update({f'{lbl}': {'x': point_x, 'y': point_y}})

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(labelme_dir), os.path.basename(labelme_dir) + '_dlc_annotations')
        if not os.path.isdir(save_dir): os.makedirs(save_dir)

    bp_names = set()
    for img, bp in results_dict.items(): bp_names.update(set(bp.keys()))
    col_names = list(itertools.product(*[[scorer], bp_names, ['x', 'y']]))
    columns = pd.MultiIndex.from_tuples(col_names)
    results = pd.DataFrame(columns=columns)
    results.columns.names = ['scorer', 'bodyparts', 'coords']
    for img, bp_data in results_dict.items():
        for bp_name, bp_cords in bp_data.items():
            results.at[img, (scorer, bp_name, 'x')] = bp_cords['x']
            results.at[img, (scorer, bp_name, 'y')] = bp_cords['y']

    for img_name, img in images.items():
        img_save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(img_save_path, img)
    save_path = os.path.join(save_dir, f'CollectedData_{scorer}.csv')
    results.to_csv(save_path)




def dlc_to_labelme(dlc_dir: Union[str, os.PathLike],
                   save_dir: Union[str, os.PathLike],
                   labelme_version: Optional[str] = '5.3.1',
                   flags: Optional[Dict[Any, Any]] = None,
                   verbose: Optional[bool] = True) -> None:

    """
    Convert a folder of DLC annotations into labelme json format.

    :param Union[str, os.PathLike] dlc_dir: Folder with DLC annotations. I.e., directory inside
    :param Union[str, os.PathLike] save_dir: Directory to where to save the labelme json files.
    :param Optional[str] labelme_version: Version number encoded in the json files.
    :param Optional[Dict[Any, Any] flags: Flags included in the json files.
    :param Optional[bool] verbose: If True, prints progress.
    :return: None

    :example:
    >>> dlc_to_labelme(dlc_dir="D:\TS_DLC\labeled-data\ts_annotations", save_dir="C:\troubleshooting\coco_data\labels\test")
    >>> dlc_to_labelme(dlc_dir=r'D:\rat_resident_intruder\dlc_data\WIN_20190816081353', save_dir=r'D:\rat_resident_intruder\labelme')
    """

    timer = SimbaTimer(start=True)
    check_if_dir_exists(dlc_dir, source=f'{dlc_to_labelme.__name__} dlc_dir')
    check_if_dir_exists(save_dir, source=f'{dlc_to_labelme.__name__} save_dir')
    check_valid_boolean(value=verbose, source=f'{dlc_to_labelme.__name__} verbose')
    collected_data_path = recursive_file_search(directory=dlc_dir, substrings='CollectedData', extensions='csv', raise_error=True)
    version = labelme_version
    if flags is not None:
        check_instance(source=f'{dlc_to_labelme.__name__} flags', instance=flags, accepted_types=(dict,))
    flags = {} if flags is None else {}
    body_parts_per_file, filecnt = {}, 0
    for file_cnt, file_path in enumerate(collected_data_path):
        file_dir = os.path.dirname(file_path)
        video_name = os.path.basename(os.path.dirname(file_path))
        body_part_headers = ['image']
        annotation_data = pd.read_csv(file_path, header=[0, 1, 2])
        body_parts = []
        for i in annotation_data.columns[1:]:
            if 'unnamed:' not in i[1].lower() and i[1] not in body_parts:
                body_parts.append(i[1])
        for i in body_parts:
            body_part_headers.append(f'{i}_x'); body_part_headers.append(f'{i}_y')
        body_parts_per_file[file_path] = body_part_headers
        annotation_data.columns = body_part_headers
        for cnt, (idx, idx_data) in enumerate(annotation_data.iterrows()):
            if verbose:
                print(f'Processing image {cnt + 1}/{len(annotation_data)}... (video {file_cnt+1}/{len(collected_data_path)} ({video_name}))')
            _, img_name, ext = get_fn_ext(filepath=idx_data['image'])
            video_img_name = f'{video_name}.{img_name}'
            img_path = os.path.join(file_dir, os.path.join(f'{img_name}{ext}'))
            check_file_exist_and_readable(file_path=img_path)
            img = read_img(img_path=img_path)
            idx_data = idx_data.to_dict()
            shapes = []
            for bp_name in body_parts:
                img_shapes = {'label': bp_name,
                              'points': [idx_data[f'{bp_name}_x'], idx_data[f'{bp_name}_y']],
                              'group_id': None,
                              'description': "",
                              'shape_type': 'point',
                              'flags': {}}
                shapes.append(img_shapes)
            out = {"version": version,
                   'flags': flags,
                   'shapes': shapes,
                   'imagePath': img_path,
                   'imageData': arr_to_b64(img),
                   'imageHeight': img.shape[0],
                   'imageWidth': img.shape[1]}
            save_path = os.path.join(save_dir, f'{video_img_name}.json')
            with open(save_path, "w") as f:
                json.dump(out, f)
            filecnt += 1
    timer.stop_timer()
    if verbose:
        stdout_success(f'Labelme data for {filecnt} image(s) saved in {save_dir} directory', elapsed_time=timer.elapsed_time_str)


def b64_dict_to_imgs(x: Dict[str, np.ndarray]):
    """
    Helper to convert a dictionary of images in byte64 format to a dictionary of images in array format.

    :example:
    >>> df = labelme_to_df(labelme_dir=r'C:\troubleshooting\coco_data\labels\test_2')
    >>> x = df.set_index('image_name')['image'].to_dict()
    >>> b64_dict_to_imgs(x)
    """
    results = {}
    for k, v in x.items():
        results[k] = b64_to_arr(v)
    return results


def normalize_img_dict(img_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    img_ndims = set()
    for img in img_dict.values():
        check_if_valid_img(data=img, source=normalize_img_dict.__name__, raise_error=True)
        img_ndims.add(img.ndim)
    if len(img_ndims) > 1:
        raise InvalidInputError(msg=f'Images in dictonary have to all be either color OR greyscale. Got {img_ndims} dimensions.', source=normalize_img_dict.__name__)

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

def labelme_to_df(labelme_dir: Union[str, os.PathLike],
                  greyscale: Optional[bool] = False,
                  pad: Optional[bool] = False,
                  size: Union[Literal['min', 'max'], Tuple[int, int]] = None,
                  normalize: Optional[bool] = False,
                  save_path: Optional[Union[str, os.PathLike]] = None,
                  verbose: bool = True) -> Union[None, pd.DataFrame]:

    """
    Convert a directory of labelme .json files into a pandas dataframe.

    .. note::
       The images are stores as a 64-bit bytestring under the ``image`` header of the output dataframe.

    :param Union[str, os.PathLike] labelme_dir: Directory with labelme json files.
    :param Optional[bool] greyscale: If True, converts the labelme images to greyscale if in rgb format. Default: False.
    :param Optional[bool] pad: If True, checks if all images are the same size and if not; pads the images with black border so all images are the same size.
    :param Union[Literal['min', 'max'], Tuple[int, int]] size: The size of the output images. Can be the smallesgt (min) the largest (max) or a tuple with the width and height of the images. Automatically corrects the labels to account for the image size.
    :param Optional[bool] normalize: If true, normalizes the images. Default: False.
    :param Optional[Union[str, os.PathLike]] save_path: The location where to store the dataframe. If None, then returns the dataframe. Default: None.

    :rtype: Union[None, pd.DataFrame]

    :example:
    >>> labelme_to_df(labelme_dir=r'C:\troubleshooting\coco_data\labels\test_2')
    >>> df = labelme_to_df(labelme_dir=r'C:\troubleshooting\coco_data\labels\test_read', greyscale=False, pad=False, normalize=False, size='min')
    """
    timer = SimbaTimer(start=True)
    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, f'labelme_data_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
    elif save_path is not None:
        save_path = save_path
    if size is not None:
        if isinstance(size, tuple):
            check_valid_tuple(source=f'{labelme_to_df.__name__} size', accepted_lengths=(2,), valid_dtypes=(int,), x=size)
        elif isinstance(size, str):
            check_str(name=f'{labelme_to_df.__name__} size', value=size, options=('min', 'max',), raise_error=True)
    check_if_dir_exists(in_dir=labelme_dir)
    check_valid_boolean(value=greyscale, source=f'{labelme_to_df.__name__} greyscale', raise_error=True)
    check_valid_boolean(value=pad, source=f'{labelme_to_df.__name__} pad', raise_error=True)
    check_valid_boolean(value=normalize, source=f'{labelme_to_df.__name__} normalize', raise_error=True)
    check_valid_boolean(value=verbose, source=f'{labelme_to_df.__name__} verbose', raise_error=True)
    annotation_paths = find_files_of_filetypes_in_directory(directory=labelme_dir, extensions=['.json'], raise_error=True)
    images = {}
    annotations = []
    for cnt, annot_path in enumerate(annotation_paths):
        if verbose:
            print(f'Reading annotation file {cnt+1}/{len(annotation_paths)}...')
        with open(annot_path) as f: annot_data = json.load(f)
        check_if_keys_exist_in_dict(data=annot_data, key=['shapes', 'imageData'], name=annot_path)
        img_name = annot_data['imagePath']
        images[img_name] = b64_to_arr(annot_data['imageData'])
        if greyscale:
            if len(images[img_name].shape) != 2:
                images[img_name] = (0.07 * images[img_name][:, :, 2] + 0.72 * images[img_name][:, :, 1] + 0.21 * images[img_name][:, :, 0]).astype(np.uint8)
        img_data = {}
        for bp_data in annot_data['shapes']:
            check_if_keys_exist_in_dict(data=bp_data, key=['label', 'points'], name=annot_path)
            point_x, point_y = bp_data['points'][0], bp_data['points'][1]
            lbl = bp_data['label']
            img_data[f'{lbl}_x'], img_data[f'{lbl}_y'] = point_x, point_y
        img_data['image_name'] = img_name
        annotations.append(pd.DataFrame.from_dict(img_data, orient='index').T)
    if pad:
        if verbose: print('Padding images...')
        images = ImageMixin.pad_img_stack(image_dict=images)
    if normalize:
        if verbose: print('Normalizing images...')
        images = normalize_img_dict(img_dict=images)
    img_lst = []
    for k, v in images.items():
        img_lst.append(arr_to_b64(v))
    out = pd.concat(annotations).reset_index(drop=True)
    out['image'] = img_lst
    if size is not None:
        if verbose: print('Resizing images...')
        pose_data = out.drop(['image', 'image_name'], axis=1)
        pose_data_arr = pose_data.values.reshape(-1, int(pose_data.shape[1] / 2), 2).astype(np.float32)
        new_pose, out['image'] = scale_pose_img_sizes(pose_data=pose_data_arr, imgs=list(out['image']), size=size)
        new_pose = new_pose.reshape(pose_data.shape[0], pose_data.shape[1])
        out.iloc[:, : new_pose.shape[1]] = new_pose
    if save_path is None:
        return out
    else:
        if verbose: print('Saving CSV file...')
        out.to_csv(save_path)
        timer.stop_timer()
        if verbose:
            stdout_success(msg=f'Labelme CSV file saved at {save_path}', elapsed_time=timer.elapsed_time_str, source=labelme_to_df.__name__)

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


def split_yolo_train_test_val(data_dir: Union[str, os.PathLike],
                              save_dir: Union[str, os.PathLike],
                              split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                              verbose: bool = False) -> None:
    """
    Split a directory of yolo labels and associated images into training, testing, and validation batches and create a mapping file for downstream model training.

    :param Union[str, os.PathLike] data_dir: Directory holding yolo labels and associated images. This directory is created by :func:`simba.third_party_label_appenders.converters.simba_rois_to_yolo`
    :param Union[str, os.PathLike]: Empty directory where to save the splitted data.
    :param Tuple[float, float, float] split: The percent size of the training, testing, and validation sets as ratios of the input data.
    :param bool verbose: If True, prints progress. Default False.
    :return: None
    """

    timer = SimbaTimer(start=True)
    check_if_dir_exists(in_dir=data_dir)
    check_if_dir_exists(in_dir=save_dir)
    train_img_dir, test_img_dir, val_img_dir = os.path.join(save_dir, 'images', 'train'), os.path.join(save_dir, 'images', 'test'), os.path.join(save_dir, 'images', 'val')
    train_lbl_dir, test_lbl_dir, val_lbl_dir = os.path.join(save_dir, 'labels', 'train'), os.path.join(save_dir, 'labels', 'test'), os.path.join(save_dir, 'labels', 'val')
    for i in [train_img_dir, test_img_dir, val_img_dir, train_lbl_dir, test_lbl_dir, val_lbl_dir]:
        if not os.path.isdir(i): os.makedirs(i)
    check_valid_tuple(x=split, source=split_yolo_train_test_val.__name__, accepted_lengths=(3,), valid_dtypes=(float,))
    if np.round(np.sum(split), 2) != 1.0:
        raise InvalidInputError(msg=f'Split has to add up to 1. Got: {np.round(np.sum(split), 2)}',  source=split_yolo_train_test_val.__name__)
    check_valid_boolean(value=[verbose], source=split_yolo_train_test_val.__name__)
    labels_dir, img_dir = os.path.join(data_dir, 'labels'), os.path.join(data_dir, 'images')
    check_if_dir_exists(in_dir=labels_dir)
    check_if_dir_exists(in_dir=img_dir)
    map_path = os.path.join(data_dir, 'map.json')
    check_file_exist_and_readable(file_path=map_path)
    map_dict = read_json(x=map_path)
    yolo_yaml_path = os.path.join(save_dir, 'map.yaml')
    img_paths = np.array(sorted(find_files_of_filetypes_in_directory(directory=img_dir, extensions=['.png'], raise_error=True)))
    lbls_paths = np.array(sorted(find_files_of_filetypes_in_directory(directory=labels_dir, extensions=['.txt'], raise_error=True)))
    img_names = np.array([get_fn_ext(filepath=x)[1] for x in img_paths])
    lbl_names = np.array([get_fn_ext(filepath=x)[1] for x in lbls_paths])
    missing_imgs = [x for x in img_names if x not in lbl_names]
    missing_lbls = [x for x in lbl_names if x not in img_names]
    if len(missing_imgs) > 0: raise InvalidInputError(
        msg=f'{len(missing_imgs)} label(s) are missing an image: {missing_imgs}',
        source=split_yolo_train_test_val.__name__)
    if len(missing_lbls) > 0: raise InvalidInputError(
        msg=f'{len(missing_lbls)} images(s) are missing a label: {missing_lbls}',
        source=split_yolo_train_test_val.__name__)
    train_cnt, test_cnt, val_cnt = int(np.ceil(len(img_names) * split[0])), int(
        np.ceil(len(img_names) * split[1])), int(np.ceil(len(img_names) * split[2]))
    lbl_idx = np.arange(0, len(img_names))
    np.random.shuffle(lbl_idx)
    train_idx, test_idx, val_idx = lbl_idx[:train_cnt], lbl_idx[train_cnt:train_cnt + test_cnt], lbl_idx[train_cnt + test_cnt:train_cnt + test_cnt + val_cnt]
    train_img_paths, test_img_paths, val_img_paths = img_paths[train_idx], img_paths[test_idx], img_paths[val_idx]
    train_lbl_paths, test_lbl_paths, val_lbl_paths = lbls_paths[train_idx], lbls_paths[test_idx], lbls_paths[val_idx]
    create_yolo_yaml(path=save_dir, train_path=train_img_dir, val_path=val_img_dir, test_path=test_img_dir, names=map_dict, save_path=yolo_yaml_path)
    copy_files_to_directory(file_paths=list(train_img_paths), dir=train_img_dir, verbose=verbose)
    copy_files_to_directory(file_paths=list(test_img_paths), dir=test_img_dir, verbose=verbose)
    copy_files_to_directory(file_paths=list(val_img_paths), dir=val_img_dir, verbose=verbose)
    copy_files_to_directory(file_paths=list(train_lbl_paths), dir=train_lbl_dir, verbose=verbose)
    copy_files_to_directory(file_paths=list(test_lbl_paths), dir=test_lbl_dir, verbose=verbose)
    copy_files_to_directory(file_paths=list(val_lbl_paths), dir=val_lbl_dir, verbose=verbose)
    timer.stop_timer()
    if verbose:
        stdout_success(msg=f'YOLO training data saved in {save_dir}', elapsed_time=timer.elapsed_time_str)


def simba_rois_to_yolo(config_path: Optional[Union[str, os.PathLike]] = None,
                       roi_path: Optional[Union[str, os.PathLike]] = None,
                       video_dir: Optional[Union[str, os.PathLike]] = None,
                       save_dir: Optional[Union[str, os.PathLike]] = None,
                       roi_frm_cnt: Optional[int] = 10,
                       train_size: Optional[float] = 0.7,
                       obb: Optional[bool] = False,
                       greyscale: Optional[bool] = True,
                       verbose: Optional[bool] = False) -> None:
    """
    Converts SimBA roi definitions into annotations and images for training yolo network.

    :param Optional[Union[str, os.PathLike]] config_path: Optional path to the project config file in SimBA project.
    :param Optional[Union[str, os.PathLike]] roi_path: Path to the SimBA roi definitions .h5 file. If None, then the ``roi_coordinates_path`` of the project.
    :param Optional[Union[str, os.PathLike]] video_dir: Directory where to find the videos. If None, then the videos folder of the project.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where to save the labels and images. If None, then the logs folder of the project.
    :param Optional[int] roi_frm_cnt: Number of frames for each video to create bounding boxes for.
    :param float train_size: Proportion of frames randomly assigned to the training dataset. Value must be between 0.1 and 0.99. Default: 0.7.
    :param Optional[bool] obb: If True, created object-oriented yolo bounding boxes. Else, axis aligned yolo bounding boxes. Default False.
    :param Optional[bool] greyscale: If True, converts the images to greyscale if rgb. Default: True.
    :param Optional[bool] verbose: If True, prints progress. Default: False.
    :return: None

    :example I:
    >>> simba_rois_to_yolo(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")

    :example II:
    >>> simba_rois_to_yolo(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini", save_dir=r"C:\troubleshooting\RAT_NOR\project_folder\logs\yolo", video_dir=r"C:\troubleshooting\RAT_NOR\project_folder\videos", roi_path=r"C:\troubleshooting\RAT_NOR\project_folder\logs\measures\ROI_definitions.h5")

    :example III:
    >>> simba_rois_to_yolo(video_dir=r"C:\troubleshooting\RAT_NOR\project_folder\videos", roi_path=r"C:\troubleshooting\RAT_NOR\project_folder\logs\measures\ROI_definitions.h5", save_dir=r'C:\troubleshooting\RAT_NOR\project_folder\yolo', verbose=True, roi_frm_cnt=20, obb=True)
    """

    timer = SimbaTimer(start=True)
    if roi_path is None or video_dir is None or save_dir is None:
        config = ConfigReader(config_path=config_path)
        roi_path = config.roi_coordinates_path
        video_dir = config.video_dir
        save_dir = config.logs_path
    check_int(name=f'{simba_rois_to_yolo.__name__} roi_frm_cnt', value=roi_frm_cnt, min_value=1)
    check_valid_boolean(value=verbose, source=f'{simba_rois_to_yolo.__name__} verbose')
    check_valid_boolean(value=greyscale, source=f'{simba_rois_to_yolo.__name__} greyscale')
    check_valid_boolean(value=obb, source=f'{simba_rois_to_yolo.__name__} obb')
    check_float(name=f'{simba_rois_to_yolo.__name__} train_size', min_value=0.001, max_value=0.9999, value=train_size, raise_error=True)
    check_if_dir_exists(in_dir=video_dir)
    roi_data = read_roi_data(roi_path=roi_path)
    roi_geometries = GeometryMixin.simba_roi_to_geometries(rectangles_df=roi_data[0], circles_df=roi_data[1], polygons_df=roi_data[2])[0]
    video_files = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True, raise_warning=False, as_dict=True)
    sliced_roi_geometries = {k: v for k, v in roi_geometries.items() if k in video_files.keys()}
    if len(sliced_roi_geometries.keys()) == 0:
        raise NoFilesFoundError(
            msg=f'No video files for in {video_dir} directory for the videos represented in the {roi_path} file: {roi_geometries.keys()}',
            source=simba_rois_to_yolo.__name__)
    roi_geometries_rectangles = {}
    roi_ids, roi_cnt = {}, 0
    map_path = os.path.join(save_dir, 'map.yaml')
    img_dir, lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
    img_train_dir, img_val_dir = os.path.join(img_dir, 'train'), os.path.join(img_dir, 'val')
    lbl_train_dir, lb_val_dir = os.path.join(lbl_dir, 'train'), os.path.join(lbl_dir, 'val')
    create_directory(paths=[img_train_dir, img_val_dir, lbl_train_dir, lb_val_dir], overwrite=False)
    if verbose: print('Reading geometries...')
    for video_cnt, (video_name, roi_data) in enumerate(sliced_roi_geometries.items()):
        if verbose: print(
            f'Reading ROI geometries for video {video_name}... ({video_cnt + 1}/{len(list(roi_geometries.keys()))})')
        roi_geometries_rectangles[video_name] = {}
        for roi_name, roi in roi_data.items():
            if obb:
                roi_geometries_rectangles[video_name][roi_name] = GeometryMixin.minimum_rotated_rectangle(shape=roi)
            else:
                keypoints = np.array(roi.exterior.coords).astype(np.int32).reshape(1, -1, 2)
                roi_geometries_rectangles[video_name][roi_name] = Polygon(
                    GeometryMixin.keypoints_to_axis_aligned_bounding_box(keypoints=keypoints)[0])
                print(roi_geometries_rectangles[video_name][roi_name])
            if roi_name not in roi_ids.keys():
                roi_ids[roi_name] = roi_cnt
                roi_cnt += 1

    roi_results, img_results = {}, {}
    if verbose: print('Reading coordinates ...')
    for video_cnt, (video_name, roi_data) in enumerate(roi_geometries_rectangles.items()):
        if verbose: print(
            f'Reading ROI coordinates for video {video_name}... ({video_cnt + 1}/{len(list(roi_geometries_rectangles.keys()))})')
        roi_results[video_name] = {}
        img_results[video_name] = []
        video_path = find_video_of_file(video_dir=video_dir, filename=video_name)
        video_meta_data = get_video_meta_data(video_path)
        if roi_frm_cnt > video_meta_data['frame_count']:
            roi_frm_cnt = video_meta_data['frame_count']
        cap = cv2.VideoCapture(video_path)
        frm_idx = np.sort(np.random.choice(np.arange(0, video_meta_data['frame_count']), size=roi_frm_cnt))
        for idx in frm_idx:
            img_results[video_name].append(read_frm_of_video(video_path=cap, frame_index=idx, greyscale=greyscale))
        w, h = video_meta_data['width'], video_meta_data['height']
        for roi_name, roi in roi_data.items():
            roi_id = roi_ids[roi_name]
            if not obb:
                shape_stats = GeometryMixin.get_shape_statistics(shapes=roi)
                x_center = shape_stats['centers'][0][0] / w
                y_center = shape_stats['centers'][0][1] / h
                width = shape_stats['widths'][0] / w
                height = shape_stats['lengths'][0] / h
                roi_str = ' '.join([str(roi_id), str(x_center), str(y_center), str(width), str(height)])
            else:
                img_geometry = np.array(roi.exterior.coords).astype(np.int32)[1:]
                x1, y1 = img_geometry[0][0] / w, img_geometry[0][1] / h
                x2, y2 = img_geometry[1][0] / w, img_geometry[1][1] / h
                x3, y3 = img_geometry[2][0] / w, img_geometry[2][1] / h
                x4, y4 = img_geometry[3][0] / w, img_geometry[3][1] / h
                roi_str = ' '.join([str(roi_id), str(x1), str(y1), str(x2), str(y2), str(x3), str(y3), str(x4), str(y4), '\n'])
            roi_results[video_name][roi_name] = roi_str

    total_img_cnt = sum(len(v) for v in img_results.values())
    train_idx = random.sample(list(range(0, total_img_cnt)), int(total_img_cnt * train_size))

    if verbose: print('Reading images ...')
    cnt = 0
    for video_cnt, (video_name, imgs) in enumerate(img_results.items()):
        if verbose: print(
            f'Reading ROI images for video {video_name}... ({video_cnt + 1}/{len(list(img_results.keys()))})')
        for img_cnt, img in enumerate(imgs):
            if cnt in train_idx:
                img_save_path = os.path.join(img_train_dir, f'{video_name}_{img_cnt}.png')
                lbl_save_path = os.path.join(lbl_train_dir, f'{video_name}_{img_cnt}.txt')
            else:
                img_save_path = os.path.join(img_val_dir, f'{video_name}_{img_cnt}.png')
                lbl_save_path = os.path.join(lb_val_dir, f'{video_name}_{img_cnt}.txt')
            cv2.imwrite(img_save_path, img)
            # circle = roi_geometries_rectangles[video_name]['circle']
            # pts = np.array(circle.exterior.coords, dtype=np.int32)
            # pts = pts.reshape((-1, 1, 2))
            # cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            # cv2.imshow('sadasdasd', img)
            x = list(roi_results[video_name].values())
            with open(lbl_save_path, mode='wt', encoding='utf-8') as f:
                f.write('\n'.join(x))
            cnt += 1

    roi_ids = {v: k for k, v in roi_ids.items()}

    create_yolo_keypoint_yaml(path=save_dir, train_path=img_train_dir, val_path=img_val_dir, names=roi_ids, save_path=map_path)
    timer.stop_timer()
    if verbose:
        stdout_success(msg=f'YOLO ROI data saved in {save_dir}', elapsed_time=timer.elapsed_time_str)



def yolo_obb_data_to_bounding_box(center_x: float, center_y: float, width: float, height: float, angle: float) -> np.ndarray:
    """
    Converts the YOLO-oriented bounding box data to a set of bounding box corner points.

    Given the center coordinates, width, height, and rotation angle of an oriented bounding box,
    this function computes the coordinates of the four corner points of the bounding box,
    with rotation applied about the center.

    :param float center_x: The x-coordinate of the bounding box center.
    :param float center_y: The y-coordinate of the bounding box center.
    :param float width: The width of the bounding box.
    :param float height: The height of the bounding box.
    :param float angle: The rotation angle of the bounding box in degrees, measured counterclockwise.

    :return: An array of shape (4, 2) containing the (x, y) coordinates of the four corners of the bounding box in the following order: top-left, top-right, bottom-right, and bottom-left.
    :rtype: np.ndarray
    """

    for value in [center_x, center_y, width, height, angle]:
        check_float(name=yolo_obb_data_to_bounding_box.__name__, value=value)
    angle_rad = np.deg2rad(angle)
    half_width, half_height = width / 2, height / 2
    corners = np.array([[-half_width, -half_height],  # tl
                        [half_width, -half_height],  # tr
                        [half_width, half_height],  # br
                        [-half_width, half_height]])  # bl
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    box = np.dot(corners, rotation_matrix) + [center_x, center_y]
    return box.astype(np.int32)


def labelme_to_img_dir(labelme_dir: Union[str, os.PathLike],
                       img_dir: Union[str, os.PathLike],
                       img_format: str = 'png',
                       verbose: bool = True,
                       greyscale: bool = False) -> None:

    """
    Given a directory of labelme JSON annotations, extract the images from the JSONs in b64 format and store them as images in a directory

    :param labelme_dir: Directory containing labelme json annotations.
    :param img_dir: Directory where to store the images.
    :param img_format: Format in which to save the images.
    :return: None

    :example:
    >>> labelme_to_img_dir(img_dir=r"C:\troubleshooting\coco_data\labels\train_images", labelme_dir=r'C:\troubleshooting\coco_data\labels\train_')
    >>> labelme_to_img_dir(img_dir=r"D:\rat_resident_intruder\labelme", labelme_dir=r'D:\rat_resident_intruder\imgs')
    """

    timer = SimbaTimer(start=True)
    check_if_dir_exists(in_dir=labelme_dir)
    check_if_dir_exists(in_dir=img_dir)
    check_valid_boolean(value=verbose, source=f'{labelme_to_img_dir.__name__} verbose')
    check_valid_boolean(value=greyscale, source=f'{labelme_to_img_dir.__name__} greyscale')
    check_str(name=f'{labelme_to_img_dir.__name__} img_format', value=f'.{img_format}', options=Options.ALL_IMAGE_FORMAT_OPTIONS.value)
    img_format = f'.{img_format}'
    annotation_paths = find_files_of_filetypes_in_directory(directory=labelme_dir, extensions=['.json'], raise_error=True)
    for file_cnt, annot_path in enumerate(annotation_paths):
        with open(annot_path) as f: annot_data = json.load(f)
        check_if_keys_exist_in_dict(data=annot_data, key=['shapes', 'imageData', 'imagePath'], name=annot_path)
        img_name = os.path.basename(annot_data['imagePath'])
        if verbose:
            print(f'Reading image {file_cnt+1}/{len(annotation_paths)} ({img_name})...')
        img = b64_to_arr(annot_data['imageData'])
        if greyscale:
            img = ImageMixin.img_to_greyscale(img=img)
        save_path = os.path.join(img_dir, f'{img_name}{img_format}')
        cv2.imwrite(filename=save_path, img=img)
    timer.stop_timer()
    stdout_success(msg=f'{len(annotation_paths)} images saved in {img_dir}.', elapsed_time=timer.elapsed_time_str)


def labelme_to_yolo(labelme_dir: Union[str, os.PathLike],
                    save_dir: Union[str, os.PathLike],
                    obb: bool = False,
                    verbose: bool = True) -> None:
    """
    Convert LabelMe annotations in json to YOLO format and save the corresponding images and labels in txt format.

    .. note::
       For more information on the LabelMe annotation tool, see the `LabelMe GitHub repository <https://github.com/wkentaro/labelme>`_.
       The LabemLe Json files **has too** contain a `imageData` key holding the image as a b64 string.

    .. seealso::
       To split YOLO data into train, test, and validation sets (expected by e.g., UltraLytics), see :func:`simba.third_party_label_appenders.converters.split_yolo_train_test_val`.

    :param Union[str, os.PathLike labelme_dir: Path to the directory containing LabelMe annotation `.json` files.
    :param Union[str, os.PathLike save_dir: Directory where the YOLO-format images and labels will be saved. Will create 'images/', 'labels/', and 'map.json' inside this directory.
    :param bool obb: If True, saves annotations as oriented bounding boxes (8 coordinates). If False, uses standard YOLO format (x_center, y_center, width, height)
    :param bool verbose: If True, prints progress messages during conversion.


    :example:
    >>> LABELME_DIR = r'D:/annotations'
    >>> SAVE_DIR = r"D:/yolo_data"
    >>> labelme_to_yolo(labelme_dir=LABELME_DIR, save_dir=SAVE_DIR)
    """

    timer = SimbaTimer(start=True)
    check_if_dir_exists(in_dir=os.path.dirname(save_dir), source=f'{labelme_to_yolo.__name__} save_dir', raise_error=True)
    check_if_dir_exists(in_dir=labelme_dir, source=f'{labelme_to_yolo.__name__} labelme_dir', raise_error=True)
    labelme_file_paths = find_files_of_filetypes_in_directory(directory=labelme_dir, extensions=['.json'], raise_error=True)
    save_img_dir = os.path.join(save_dir, 'images')
    save_labels_dir = os.path.join(save_dir, 'labels')
    map_path = os.path.join(save_dir, 'map.json')
    create_directory(paths=save_img_dir, overwrite=True)
    create_directory(paths=save_labels_dir, overwrite=True)
    check_valid_boolean(value=[verbose], source=f'{labelme_to_yolo.__name__} verbose', raise_error=True)
    check_valid_boolean(value=[obb], source=f'{labelme_to_yolo.__name__} obb', raise_error=True)
    labels = {}
    for file_cnt, file_path in enumerate(labelme_file_paths):
        if verbose: print(f'Labelme to YOLO file {file_cnt+1}/{len(labelme_file_paths)}...')
        with open(file_path) as f: annot_data = json.load(f)
        check_if_keys_exist_in_dict(data=annot_data, key=['shapes', 'imageData', 'imagePath'], name=file_path)
        img_name = get_fn_ext(filepath=annot_data['imagePath'])[1]
        img = b64_to_arr(annot_data['imageData'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        label_save_path = os.path.join(save_labels_dir, f'{img_name}.txt')
        img_save_path = os.path.join(save_img_dir, f'{img_name}.png')
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
        with open(label_save_path, mode='wt', encoding='utf-8') as f:
            f.write(roi_str)
        cv2.imwrite(img_save_path, img)
    with open(map_path, 'w') as f:
        json.dump(labels, f, indent=4)
    timer.stop_timer()
    if verbose: stdout_success(msg=f'Labelme to YOLO conversion complete. Data saved in directory {save_dir}.', elapsed_time=timer.elapsed_time_str)


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


def coco_keypoints_to_yolo(coco_path: Union[str, os.PathLike],
                           img_dir: Union[str, os.PathLike],
                           save_dir: Union[str, os.PathLike],
                           train_size: float = 0.7,
                           flip_idx: Tuple[int, ...] = (0, 2, 1, 3, 5, 4, 6),
                           verbose: bool = True):
    """
    Convert COCO Keypoints version 1.0 data format into a YOLO keypoints training set.

    .. note::
       COCO keypoint files can be created using `https://www.cvat.ai/ <https://www.cvat.ai/>`__.

    :param Union[str, os.PathLike] coco_path: Path to coco keypoints 1.0 file in json format.
    :param Union[str, os.PathLike] img_dir: Directory holding img files representing the annotated entries in the ``coco_path``.
    :param Union[str, os.PathLike] save_dir: Directory where to save the yolo formatted data.
    :param Tuple[float, float, float] split: The size of the training set. Value between 0-1.0 representing the percent of training data.
    :param bool verbose: If true, prints progress. Default: True.
    :param Tuple[int, ...] flip_idx: Tuple of ints, representing the flip of body-part coordinates when the animal image flips 180 degrees.
    :return: None

    :example:
    >>> coco_path = r"D:\netholabs\imgs_vcat\batch_1\batch_1\coco_annotations\person_keypoints_default.json"
    >>> coco_keypoints_to_yolo(coco_path=coco_path, img_dir=r'D:\netholabs\imgs_vcat\batch_1', save_dir=r"D:\netholabs\imgs_vcat\batch_1\batch_1\yolo_annotations")
    """

    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=coco_path)
    check_if_dir_exists(in_dir=save_dir)
    check_float(name=f'{coco_keypoints_to_yolo.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
    train_img_dir, val_img_dir = os.path.join(save_dir, 'images', 'train'), os.path.join(save_dir, 'images', 'val')
    train_lbl_dir, val_lbl_dir = os.path.join(save_dir, 'labels', 'train'), os.path.join(save_dir, 'labels', 'val')
    for i in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        create_directory(paths=i, overwrite=True)
    map_path = os.path.join(save_dir, 'map.yaml')
    coco_data = read_json(x=coco_path)
    check_if_keys_exist_in_dict(data=coco_data, key=['categories', 'images', 'annotations'], name=coco_path)
    map_dict = {i['id']: i['name'] for i in coco_data['categories']}
    map_ids = list(map_dict.keys())
    if sorted(map_ids) != list(range(len(map_ids))):
        map_id_lk = {}  # old: new
        new_map_dict = {}
        for cnt, v in enumerate(sorted(map_ids)):
            map_id_lk[v] = cnt
        for k, v in map_id_lk.items():
            new_map_dict[v] = map_dict[k]
        map_dict = copy(new_map_dict)
    else:
        map_id_lk = {k: k for k in map_dict.keys()}

    img_file_paths = find_files_of_filetypes_in_directory(directory=img_dir,
                                                          extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value,
                                                          raise_error=True, as_dict=True)
    img_cnt = len(coco_data['images'])
    img_idx = list(range(len(coco_data['images']) + 1))
    train_idx = random.sample(img_idx, int(img_cnt * train_size))
    check_valid_tuple(x=flip_idx, source=coco_keypoints_to_yolo.__name__, valid_dtypes=(int,), minimum_length=1)
    shapes = []
    for cnt in range(len(coco_data['images'])):
        # for cnt in range(10):
        img_data = coco_data['images'][cnt]
        check_if_keys_exist_in_dict(data=img_data, key=['width', 'height', 'file_name', 'id'], name=coco_path)
        _, img_name, ext = get_fn_ext(filepath=img_data['file_name'])
        if verbose:
            print(f'Processing annotation {cnt + 1}/{img_cnt} ({img_name})...')
        if not img_name in img_file_paths.keys():
            raise NoFilesFoundError(msg=f'The file {img_name} could not be found in the {img_dir} directory',
                                    source=coco_keypoints_to_yolo.__name__)
        img = read_img(img_path=img_file_paths[img_name])
        if (img.shape[0] != img_data['height']) or (img.shape[1] != img_data['width']):
            raise FaultyTrainingSetError(
                msg=f'Image {img_name} is of shape {img.shape[0]}x{img.shape[1]}, but the coco data has been annotated on an image of {img_data["height"]}x{img_data["width"]}.')
        img_annotations = [x for x in coco_data['annotations'] if x['image_id'] == img_data['id']]
        roi_str = ''
        if cnt in train_idx:
            label_save_path = os.path.join(train_lbl_dir, f'{img_name}.txt')
            img_save_path = os.path.join(train_img_dir, f'{img_name}.png')
        else:
            label_save_path = os.path.join(val_lbl_dir, f'{img_name}.txt')
            img_save_path = os.path.join(val_img_dir, f'{img_name}.png')
        for img_annotation in img_annotations:
            check_if_keys_exist_in_dict(data=img_annotation, key=['bbox', 'keypoints', 'id', 'image_id', 'category_id'],
                                        name=coco_path)
            x1, y1 = img_annotation['bbox'][0], img_annotation['bbox'][1]
            w, h = img_annotation['bbox'][2], img_annotation['bbox'][3]
            x_center = (x1 + (w / 2)) / img_data['width']
            y_center = (y1 + (h / 2)) / img_data['height']
            w = img_annotation['bbox'][2] / img_data['width']
            h = img_annotation['bbox'][3] / img_data['height']
            roi_str += ' '.join([f"{map_id_lk[img_annotation['category_id']]}", str(x_center), str(y_center), str(w), str(h), ' '])
            kps = np.array(img_annotation['keypoints']).reshape(-1, 3).astype(np.int32)
            x, y, v = kps[:, 0], kps[:, 1], kps[:, 2]
            x, y = x / img_data['width'], y / img_data['height']
            shapes.append(x.shape[0])
            kps = list(np.column_stack((x, y, v)).flatten())
            roi_str += ' '.join(str(x) for x in kps) + '\n'

        with open(label_save_path, mode='wt', encoding='utf-8') as f:
            f.write(roi_str)
        cv2.imwrite(img_save_path, img)
    if len(list(set(shapes))) > 1:
        raise InvalidInputError(
            msg=f'The annotation data {coco_path} contains more than one keypoint shapes: {set(shapes)}',
            source=coco_keypoints_to_yolo.__name__)
    if len(flip_idx) != shapes[0]:
        raise InvalidInputError(
            msg=f'flip_idx contains {len(flip_idx)} values but {shapes[0]} keypoints detected per image in coco data.',
            source=coco_keypoints_to_yolo.__name__)
    missing = [x for x in flip_idx if x not in list(range(shapes[0]))]
    if len(missing) > 0:
        raise InvalidInputError(msg=f'flip_idx contains index values not in keypoints ({missing}).',  source=coco_keypoints_to_yolo.__name__)
    missing = [x for x in list(range(shapes[0])) if x not in flip_idx]
    if len(missing) > 0:
        raise InvalidInputError(msg=f'keypoints contains index values not in flip_idx ({missing}).',
                                source=coco_keypoints_to_yolo.__name__)

    create_yolo_keypoint_yaml(path=save_dir, train_path=train_img_dir, val_path=val_img_dir, names=map_dict,
                              save_path=map_path, kpt_shape=(int(shapes[0]), 3), flip_idx=flip_idx)
    timer.stop_timer()
    if verbose: stdout_success(msg=f'COCO keypoints to YOLO conversion complete. Data saved in directory {save_dir}.',
                               elapsed_time=timer.elapsed_time_str)


def merge_coco_keypoints_files(data_dir: Union[str, os.PathLike],
                               save_path: Union[str, os.PathLike]):
    """
    Merges multiple annotation COCO-format keypoint JSON files into a single file.

    .. note::
       Image and annotation entries are appended after adjusting their `id` fields to be unique.

       These files can be created using `https://www.cvat.ai/ <https://www.cvat.ai/>`__.

    :param Union[str, os.PathLike] data_dir: Directory containing multiple COCO keypoints `.json` files to merge.
    :param Union[str, os.PathLike] save_path: File path to save the merged COCO keypoints JSON.
    :return: None. Results are saved in ``save_path``.
    """

    data_files = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.json'], raise_error=True,
                                                      raise_warning=False, as_dict=True)
    check_if_dir_exists(in_dir=os.path.dirname(save_path))
    results, max_image_id, max_annotation_id = None, 0, 0
    data_file_cnt = len(data_files)

    for file_cnt, (file_name, file_path) in enumerate(data_files.items()):
        print(f'Processing {file_cnt + 1}/{data_file_cnt} ({file_name})...')
        coco_data = read_json(file_path)
        check_if_keys_exist_in_dict(data=coco_data, key=['licenses', 'info', 'categories', 'images', 'annotations'],
                                    name=file_name)

        if file_cnt == 0:
            results = deepcopy(coco_data)
            max_image_id = max((img['id'] for img in results['images']), default=0)
            max_annotation_id = max((ann['id'] for ann in results['annotations']), default=0)
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
            results['images'].extend(new_images)
            new_annotations = []
            for ann in coco_data['annotations']:
                ann['id'] += max_annotation_id + 1
                ann['image_id'] = id_mapping.get(ann['image_id'], ann['image_id'])
                new_annotations.append(ann)
            results['annotations'].extend(new_annotations)
            max_image_id = max((img['id'] for img in results['images']), default=max_image_id)
            max_annotation_id = max((ann['id'] for ann in results['annotations']), default=max_annotation_id)

    save_json(data=results, filepath=save_path)
    stdout_success(msg=f'COCO keypoints file saved at {save_path}', source=merge_coco_keypoints_files.__name__)

def sleap_to_yolo_keypoints(data_dir: Union[str, os.PathLike],
                            video_dir: Union[str, os.PathLike],
                            save_dir: Union[str, os.PathLike],
                            frms_cnt: Optional[int] = None,
                            verbose: bool = True,
                            instance_threshold: float = 0,
                            train_size: float = 0.7,
                            flip_idx: Tuple[int, ...] = None,
                            names: Tuple[str, ...] = None,
                            greyscale: bool = False,
                            padding: float = 0.00):

    """
    Convert SLEAP pose estimation CSV data and corresponding videos into YOLO keypoint dataset format.

    .. note::
       This converts SLEAP **inference** data to YOLO keypoints (not SLEAP annotations).

    :param Union[str, os.PathLike] data_dir: Directory path containing SLEAP-generated CSV files with inferred keypoints.
    :param Union[str, os.PathLike] video_dir: Directory path containing corresponding videos from which frames are to be extracted.
    :param Union[str, os.PathLike] save_dir: Output directory where YOLO-formatted images, labels, and map YAML file will be saved. Subdirectories `images/train`, `images/val`, `labels/train`, `labels/val` will be created.
    :param Optional[int] frms_cnt: Number of frames to randomly sample from each video for conversion. If None, all frames are used.
    :param float instance_threshold: Minimum confidence score threshold to filter out low-confidence pose instances. Only instances with `instance.score` >= this threshold are used.
    :param float train_size: Proportion of frames randomly assigned to the training dataset. Value must be between 0.1 and 0.99. Default: 0.7.
    :param bool verbose: If True, prints progress. Default: True.
    :param Tuple[int, ...] flip_idx: Tuple of keypoint indices used for horizontal flip augmentation during training. The tuple defines the order of keypoints after flipping.
    :param Dict[str, int] map_dict: Dictionary mapping class indices to class names. Used for creating the YAML class names mapping file.
    :param float padding: Fractional padding to add around the bounding boxes (relative to image dimensions). Helps to slightly enlarge bounding boxes by this percentage. Default 0.05. E.g., Useful when all body-parts are along animal length.
    :return: None. Results saved in ``save_dir``.

    :example:
    >>> sleap_to_yolo_keypoints(data_dir=r'D:\ares\data\ant\sleap_csv', video_dir=r'D:\ares\data\ant\sleap_video', frms_cnt=550, train_size=0.8, instance_threshold=0.9, save_dir=r"D:\ares\data\ant\yolo")

    """

    data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'], as_dict=True, raise_error=True)
    video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, as_dict=True, raise_error=True)
    missing_video_paths = [x for x in video_paths.keys() if x not in data_paths.keys()]
    missing_data_paths = [x for x in data_paths.keys() if x not in video_paths.keys()]
    check_if_dir_exists(in_dir=save_dir)
    img_dir, lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
    img_train_dir, img_val_dir = os.path.join(save_dir, 'images', 'train'), os.path.join(save_dir, 'images', 'val')
    lbl_train_dir, lb_val_dir = os.path.join(save_dir, 'labels', 'train'), os.path.join(save_dir, 'labels', 'val')
    if flip_idx is not None: check_valid_tuple(x=flip_idx, source=f'{sleap_to_yolo_keypoints.__name__} flip_idx', valid_dtypes=(int,), minimum_length=1)
    if names is not None: check_valid_tuple(x=names, source=f'{sleap_to_yolo_keypoints.__name__} names', valid_dtypes=(str,), minimum_length=1)
    create_directory(paths=img_train_dir); create_directory(paths=img_val_dir)
    create_directory(paths=lbl_train_dir); create_directory(paths=lb_val_dir)
    check_float(name=f'{sleap_to_yolo_keypoints.__name__} instance_threshold', min_value=0.0, max_value=1.0, raise_error=True, value=instance_threshold)
    check_valid_boolean(value=verbose, source=f'{sleap_to_yolo_keypoints.__name__} verbose', raise_error=True)
    check_valid_boolean(value=greyscale, source=f'{sleap_to_yolo_keypoints.__name__} greyscale', raise_error=True)
    check_float(name=f'{sleap_to_yolo_keypoints.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
    check_float(name=f'{sleap_to_yolo_keypoints.__name__} padding', value=padding, max_value=1.0, min_value=0.0, raise_error=True)

    map_path = os.path.join(save_dir, 'map.yaml')
    timer = SimbaTimer(start=True)
    if frms_cnt is not None:
        check_int(name=f'{sleap_to_yolo_keypoints.__name__} frms_cnt', value=frms_cnt, min_value=1, raise_error=True)
    if len(missing_video_paths) > 0:
        raise NoFilesFoundError(msg=f'Video(s) {missing_video_paths} could not be found in {video_dir} directory', source=sleap_to_yolo_keypoints.__name__)
    if len(missing_data_paths) > 0:
        raise NoFilesFoundError(msg=f'CSV data for {missing_data_paths} could not be found in {data_dir} directory', source=sleap_to_yolo_keypoints.__name__)

    dfs = []
    for file_cnt, (file_name, file_path) in enumerate(data_paths.items()):
        df = pd.read_csv(filepath_or_buffer=file_path)
        check_valid_dataframe(df=df, source=sleap_to_yolo_keypoints.__name__, required_fields=['track', 'frame_idx', 'instance.score'])
        df = df if instance_threshold is None else df[df['instance.score'] >= instance_threshold]
        cord_cols, frame_idx = df.drop(['track', 'frame_idx', 'instance.score'], axis=1), df['frame_idx']
        selected_frms = random.sample(list(frame_idx.unique()), frms_cnt) if frms_cnt is not None else list(frame_idx.unique())
        df = df[df['frame_idx'].isin(selected_frms)]
        df['video'] = video_paths[file_name]
        dfs.append(df)

    dfs = pd.concat(dfs, axis=0)
    unique_tracks_lk = {v: k for k, v in enumerate(dfs['track'].unique())}
    if names is not None:
        check_valid_tuple(x=names, source=f'{sleap_to_yolo_keypoints.__name__} names', valid_dtypes=(str,), accepted_lengths=(len(list(unique_tracks_lk.keys())),))
    else:
        names = [f'animal_{k+1}' for k in range(len(list(unique_tracks_lk.keys())),)]
    map_dict = {k: v for k, v in enumerate(names)}
    dfs['id'] = dfs['frame_idx'].astype(str) + dfs['video'].astype(str)
    train_idx = random.sample(list(dfs['id'].unique()), int(len(dfs['frame_idx'].unique()) * train_size))
    for frm_cnt, frm_id in enumerate(dfs['id'].unique()):
        frm_data = dfs[dfs['id'] == frm_id]
        video_path = list(frm_data['video'])[0]
        frm_idx = list(frm_data['frame_idx'])[0]
        video_meta = get_video_meta_data(video_path=video_path)
        if verbose:
            print(f'Processing frame: {frm_cnt+1}/{len(dfs)} ...')
        img = read_frm_of_video(video_path=video_path, frame_index=frm_idx, greyscale=greyscale)
        img_h, img_w = img.shape[0], img.shape[1]
        if list(frm_data['id'])[0] in train_idx:
            img_save_path = os.path.join(img_dir, 'train', f'{video_meta["video_name"]}_{frm_idx}.png')
            lbl_save_path = os.path.join(lbl_dir, 'train', f'{video_meta["video_name"]}_{frm_idx}.txt')
        else:
            img_save_path  = os.path.join(img_dir, 'val', f'{video_meta["video_name"]}_{frm_idx}.png')
            lbl_save_path = os.path.join(lbl_dir, 'val', f'{video_meta["video_name"]}_{frm_idx}.txt')
        img_lbl = ''
        for track_cnt, (_, track_data) in enumerate(frm_data.iterrows()):
            track_id, keypoints = unique_tracks_lk[track_data['track']], track_data.drop(['track', 'frame_idx', 'instance.score', 'video', 'id']),
            keypoints = keypoints.values.reshape(-1, 3)
            if frms_cnt == 0 and track_cnt == 0:
                if flip_idx is not None and keypoints.shape[0] != len(flip_idx):
                    raise InvalidInputError(msg=f'The SLEAP data contains data for {keypoints.shape[0]} body-parts, but passed flip_idx suggests {len(flip_idx)} body-parts', source=sleap_to_yolo_keypoints.__name__)
                else:
                    flip_idx = list(range(0, keypoints.shape[0]))
            keypoints[keypoints[:, 2] != 0.0, 2] = 2
            instance_str = f'{track_id} '
            x_coords, y_coords = keypoints[:, 0], keypoints[:, 1]
            min_x, max_x = np.nanmin(x_coords), np.nanmax(x_coords)
            min_y, max_y = np.nanmin(y_coords), np.nanmax(y_coords)
            pad_w, pad_h = padding * img_w, padding * img_h
            min_x, max_x = max(min_x - pad_w / 2, 0), min(max_x + pad_w / 2, img_w)
            min_y, max_y = max(min_y - pad_h / 2, 0), min(max_y + pad_h / 2, img_h)
            bbox_w, bbox_h = max_x - min_x, max_y - min_y
            x_center, y_center = min_x + bbox_w / 2, min_y + bbox_h / 2
            x_center /= img_w
            y_center /= img_h
            bbox_w /= img_w
            bbox_h /= img_h
            x_center = np.clip(x_center, 0.0, 1.0)
            y_center = np.clip(y_center, 0.0, 1.0)
            bbox_w = np.clip(bbox_w, 0.0, 1.0)
            bbox_h = np.clip(bbox_h, 0.0, 1.0)
            keypoints[:, 0] /= img_w
            keypoints[:, 1] /= img_h
            keypoints[:, 0:2] = np.clip( keypoints[:, 0:2], 0.0, 1.0)
            instance_str += f"{x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f} "
            for kp in keypoints:
                instance_str += f"{kp[0]:.6f} {kp[1]:.6f} {int(kp[2])} "
            img_lbl += instance_str.strip() + '\n'
        with open(lbl_save_path, mode='wt', encoding='utf-8') as f:
            f.write(img_lbl)
        cv2.imwrite(img_save_path, img)


    create_yolo_keypoint_yaml(path=save_dir, train_path=img_train_dir, val_path=img_val_dir, names=map_dict, save_path=map_path, kpt_shape=(len(flip_idx), 3), flip_idx=flip_idx)
    timer.stop_timer()
    stdout_success(msg=f'YOLO formated data saved in {save_dir} directory', source=sleap_to_yolo_keypoints.__name__, elapsed_time=timer.elapsed_time_str)

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

    bp_id_idx, bp_cnt = {}, 0
    for animal_cnt, (animal_name, animal_bp_data) in enumerate(animal_bp_dict.items()):
        bp_id_idx[animal_cnt] = list(range(bp_cnt, bp_cnt + len(animal_bp_data['X_bps'])))
        bp_cnt += max(bp_id_idx[animal_cnt]) + 1
    return bp_id_idx


def dlc_to_yolo_keypoints(dlc_dir: Union[str, os.PathLike],
                          save_dir: Union[str, os.PathLike],
                          train_size: float = 0.7,
                          verbose: bool = False,
                          padding: float = 0.00,
                          flip_idx: Tuple[int, ...] = (1, 0, 2, 3, 5, 4, 6, 7),
                          map_dict: Dict[int, str] = {0: 'mouse'},
                          greyscale: bool = False,
                          bp_id_idx: Optional[Dict[int, Union[Tuple[int], List[int]]]] = None) -> None:

    """
    Converts DLC annotations into YOLO keypoint format formatted for model training.

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
    :param Dict[int, str] map_dict: Dictionary mapping class indices to class names. Used for creating the YAML class names mapping file.
    :return: None. Results saved in ``save_dir``.

    :example:
    >>> dlc_to_yolo_keypoints(dlc_dir=r'D:\mouse_operant_data\Operant_C57_labelled_images\labeled-data', save_dir=r"D:\mouse_operant_data\yolo", verbose=True)
    >>> dlc_to_yolo_keypoints(dlc_dir=r'D:\rat_resident_intruder\dlc_data', save_dir=r"D:\rat_resident_intruder\yolo", verbose=True, bp_id_idx={0: list(range(0, 8)), 1: list(range(8, 16))},  map_dict={0: 'resident', 1: 'intruder'})
    """

    timer = SimbaTimer(start=True)
    check_if_dir_exists(in_dir=dlc_dir, source=f'{dlc_to_yolo_keypoints.__name__} dlc_dir')
    check_valid_boolean(value=verbose, source=f'{dlc_to_yolo_keypoints.__name__} verbose')
    check_valid_boolean(value=greyscale, source=f'{dlc_to_yolo_keypoints.__name__} greyscale')
    check_float(name=f'{dlc_to_yolo_keypoints.__name__} padding', value=padding, max_value=1.0, min_value=0.0, raise_error=True)
    check_float(name=f'{dlc_to_yolo_keypoints.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
    check_valid_dict(x=map_dict, valid_key_dtypes=(int,), valid_values_dtypes=(str,), min_len_keys=1)
    check_if_dir_exists(in_dir=save_dir)
    annotation_paths = recursive_file_search(directory=dlc_dir, substrings=['CollectedData'], extensions=['csv'], case_sensitive=False, raise_error=True)
    check_valid_tuple(x=flip_idx, source=dlc_to_yolo_keypoints.__name__, valid_dtypes=(int,), minimum_length=1)
    img_dir, lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
    img_train_dir, img_val_dir = os.path.join(img_dir, 'train'), os.path.join(img_dir, 'val')
    lbl_train_dir, lb_val_dir = os.path.join(lbl_dir, 'train'), os.path.join(lbl_dir, 'val')
    create_directory(paths=[img_train_dir, img_val_dir, lbl_train_dir, lb_val_dir], overwrite=False)
    annotations = []
    if bp_id_idx is not None:
        check_valid_dict(x=bp_id_idx, valid_key_dtypes=(int,), valid_values_dtypes=(tuple, list,))
        missing_map_dict_keys = [x for x in bp_id_idx.keys() if x not in map_dict.keys()]
        missing_bp_id_keys = [x for x in map_dict.keys() if x not in bp_id_idx.keys()]
        if len(missing_map_dict_keys) > 0:
            raise InvalidInputError(msg=f'Keys {missing_map_dict_keys} exist in bp_id_idx but is not passed in map_dict', source=dlc_to_yolo_keypoints.__name__)
        if len(missing_bp_id_keys) > 0:
            raise InvalidInputError(msg=f'Keys {missing_bp_id_keys} exist in map_dict but is not passed in bp_id_idx', source=dlc_to_yolo_keypoints.__name__)

    map_path = os.path.join(save_dir, 'map.yaml')
    for file_cnt, annotation_path in enumerate(annotation_paths):
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
        check_valid_dataframe(df=annotation_data, source=dlc_to_yolo_keypoints.__name__, valid_dtypes=Formats.NUMERIC_DTYPES.value)
        annotation_data = annotation_data.reset_index(drop=True)
        img_paths = [os.path.join(os.path.dirname(annotation_path), os.path.basename(x)) for x in img_paths]
        annotation_data['img_path'] = img_paths
        annotations.append(annotation_data)
    annotations = pd.concat(annotations, axis=0).reset_index(drop=True)
    img_paths = annotations.pop('img_path').reset_index(drop=True).values
    train_idx = random.sample(list(range(0, len(annotations))), int(len(annotations) * train_size))
    for cnt, (idx, idx_data) in enumerate(annotations.iterrows()):
        img_lbl = ''
        if verbose:
            print(f'Processing image {cnt+1}/{len(annotations)}...')
        file_name = f"{os.path.basename(os.path.dirname(img_paths[cnt]))}.{os.path.splitext(os.path.basename(img_paths[cnt]))[0]}"
        if idx in train_idx:
            img_save_path, lbl_save_path = os.path.join(img_dir, 'train', f'{file_name}.png'), os.path.join(lbl_dir, 'train', f'{file_name}.txt')
        else:
            img_save_path, lbl_save_path = os.path.join(img_dir, 'val', f'{file_name}.png'), os.path.join(lbl_dir, 'val', f'{file_name}.txt')
        check_file_exist_and_readable(img_paths[cnt])
        img = read_img(img_path=img_paths[cnt], greyscale=greyscale)
        img_h, img_w = img.shape[0], img.shape[1]
        keypoints_with_id = {}
        if bp_id_idx is not None:
            for k, idx in bp_id_idx.items():
                keypoints_with_id[k] = idx_data.values.reshape(-1, 2)[idx, :]
        else:
            keypoints_with_id[0] = idx_data.values.reshape(-1, 2)

        for id, keypoints in keypoints_with_id.items():
            if np.all(np.isnan(keypoints)) or np.all(keypoints == 0.0) or np.all(np.isnan(keypoints) | (keypoints == 0.0)):
               continue
            instance_str = f'{id} '
            x_coords, y_coords = keypoints[:, 0], keypoints[:, 1]
            min_x, max_x = np.nanmin(x_coords), np.nanmax(x_coords)
            min_y, max_y = np.nanmin(y_coords), np.nanmax(y_coords)
            pad_w, pad_h = padding * img_w, padding * img_h
            min_x, max_x = max(min_x - pad_w / 2, 0), min(max_x + pad_w / 2, img_w)
            min_y, max_y = max(min_y - pad_h / 2, 0), min(max_y + pad_h / 2, img_h)
            if max_x - min_x < 1 or max_y - min_y < 1:
                continue
            bbox_w, bbox_h = max_x - min_x, max_y - min_y
            x_center, y_center = min_x + bbox_w / 2, min_y + bbox_h / 2
            x_center /= img_w
            y_center /= img_h
            bbox_w /= img_w
            bbox_h /= img_h
            x_center = np.clip(x_center, 0.0, 1.0)
            y_center = np.clip(y_center, 0.0, 1.0)
            bbox_w = np.clip(bbox_w, 0.0, 1.0)
            bbox_h = np.clip(bbox_h, 0.0, 1.0)
            # for kp in keypoints:
            #     if not np.isnan(kp).any():
            #         p = (int(kp[0]), int(kp[1]))
            #         img = cv2.circle(img, p, 5, (255, 255, 0), 2,)
            # cv2.imshow('saasd', img)
            # cv2.waitKey(1000)
            keypoints[:, 0] /= img_w
            keypoints[:, 1] /= img_h
            keypoints[:, 0:2] = np.clip(keypoints[:, 0:2], 0.0, 1.0)
            visability_col = np.where(np.isnan(keypoints).any(axis=1), 0, 2)
            keypoints = np.nan_to_num(np.hstack((keypoints, visability_col[:, np.newaxis])), nan=0.0)
            instance_str += f"{x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f} "
            for kp in keypoints:
                instance_str += f"{kp[0]:.6f} {kp[1]:.6f} {int(kp[2])} "
            img_lbl += instance_str.strip() + '\n'
        with open(lbl_save_path, mode='wt', encoding='utf-8') as f:
            f.write(img_lbl)
        cv2.imwrite(img_save_path, img)
    create_yolo_keypoint_yaml(path=save_dir, train_path=img_train_dir, val_path=img_val_dir, names=map_dict, save_path=map_path, kpt_shape=(len(flip_idx), 3), flip_idx=flip_idx)
    timer.stop_timer()
    stdout_success(msg=f'YOLO formated data saved in {save_dir} directory', source=dlc_to_yolo_keypoints.__name__, elapsed_time=timer.elapsed_time_str)



def dlc_multi_animal_h5_to_yolo_keypoints(data_dir: Union[str, os.PathLike],
                                          video_dir: Union[str, os.PathLike],
                                          save_dir: Union[str, os.PathLike],
                                          frms_cnt: Optional[int] = None,
                                          verbose: bool = True,
                                          threshold: float = 0,
                                          train_size: float = 0.7,
                                          flip_idx: Tuple[int, ...] = None,
                                          names: Tuple[str, ...] = None,
                                          greyscale: bool = False,
                                          padding: float = 0.00):

    """
    Convert SLEAP pose estimation CSV data and corresponding videos into YOLO keypoint dataset format.

    .. note::
       This converts SLEAP **inference** data to YOLO keypoints (not SLEAP annotations).

    :param Union[str, os.PathLike] data_dir: Directory path containing DLC-generated H5 files with inferred keypoints.
    :param Union[str, os.PathLike] video_dir: Directory path containing corresponding videos from which frames are to be extracted.
    :param Union[str, os.PathLike] save_dir: Output directory where YOLO-formatted images, labels, and map YAML file will be saved. Subdirectories `images/train`, `images/val`, `labels/train`, `labels/val` will be created.
    :param Optional[int] frms_cnt: Number of frames to randomly sample from each video for conversion. If None, all frames are used.
    :param float instance_threshold: Minimum confidence score threshold to filter out low-confidence pose instances. Only instances with `instance.score` >= this threshold are used.
    :param float train_size: Proportion of frames randomly assigned to the training dataset. Value must be between 0.1 and 0.99. Default: 0.7.
    :param bool verbose: If True, prints progress. Default: True.
    :param Tuple[int, ...] flip_idx: Tuple of keypoint indices used for horizontal flip augmentation during training. The tuple defines the order of keypoints after flipping.
    :param Dict[str, int] map_dict: Dictionary mapping class indices to class names. Used for creating the YAML class names mapping file.
    :param float padding: Fractional padding to add around the bounding boxes (relative to image dimensions). Helps to slightly enlarge bounding boxes by this percentage. Default 0.05. E.g., Useful when all body-parts are along animal length.
    :return: None. Results saved in ``save_dir``.

    :example:
    >>> dlc_multi_animal_h5_to_yolo_keypoints(data_dir=r'D:\troubleshooting\dlc_h5_multianimal_to_yolo\data', video_dir=r'D:\troubleshooting\dlc_h5_multianimal_to_yolo\videos', save_dir=r'D:\troubleshooting\dlc_h5_multianimal_to_yolo\yolo')

    """
    timer = SimbaTimer(start=True)
    data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.h5', '.H5'], as_dict=True, raise_error=True)
    video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, as_dict=True, raise_error=True)
    check_if_dir_exists(in_dir=save_dir)
    img_dir, lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
    img_train_dir, img_val_dir = os.path.join(save_dir, 'images', 'train'), os.path.join(save_dir, 'images', 'val')
    lbl_train_dir, lb_val_dir = os.path.join(save_dir, 'labels', 'train'), os.path.join(save_dir, 'labels', 'val')
    if flip_idx is not None: check_valid_tuple(x=flip_idx, source=f'{dlc_multi_animal_h5_to_yolo_keypoints.__name__} flip_idx', valid_dtypes=(int,), minimum_length=1)
    if names is not None: check_valid_tuple(x=names, source=f'{dlc_multi_animal_h5_to_yolo_keypoints.__name__} names', valid_dtypes=(str,), minimum_length=1)
    create_directory(paths=img_train_dir); create_directory(paths=img_val_dir)
    create_directory(paths=lbl_train_dir); create_directory(paths=lb_val_dir)
    check_float(name=f'{dlc_multi_animal_h5_to_yolo_keypoints.__name__} threshold', min_value=0.0, max_value=1.0, raise_error=True, value=threshold)
    check_valid_boolean(value=verbose, source=f'{dlc_multi_animal_h5_to_yolo_keypoints.__name__} verbose', raise_error=True)
    check_valid_boolean(value=greyscale, source=f'{dlc_multi_animal_h5_to_yolo_keypoints.__name__} greyscale', raise_error=True)
    check_float(name=f'{dlc_multi_animal_h5_to_yolo_keypoints.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
    check_float(name=f'{dlc_multi_animal_h5_to_yolo_keypoints.__name__} padding', value=padding, max_value=1.0, min_value=0.0, raise_error=True)
    data_and_videos_lk = PoseImporterMixin().link_video_paths_to_data_paths(data_paths=list(data_paths.values()), video_paths=list(video_paths.values()), str_splits=Formats.DLC_NETWORK_FILE_NAMES.value)
    map_path, animal_names = os.path.join(save_dir, 'map.yaml'), names

    if frms_cnt is not None:
        check_int(name=f'{dlc_multi_animal_h5_to_yolo_keypoints.__name__} frms_cnt', value=frms_cnt, min_value=1, raise_error=True)

    dfs = []
    for cnt, (video_name, video_data) in enumerate(data_and_videos_lk.items()):
        data = pd.read_hdf(video_data["DATA"], header=[0, 1, 2, 3]).replace([np.inf, -np.inf], np.nan).fillna(-1)
        new_cols = [(x[1], x[2], x[3]) for x in list(data.columns)]
        if cnt == 0 and animal_names is None:
            animal_names = list(dict.fromkeys([x[1] for x in list(data.columns)]))
        data.columns = new_cols
        p_data = data[[x for x in data.columns if 'likelihood' in x]]
        data = data.iloc[(p_data[(p_data > threshold).all(axis=1)].index)]
        data['frm_idx'] = data.index
        data['video'] = video_data['VIDEO']
        selected_frms = random.sample(list(data['frm_idx'].unique()), frms_cnt) if frms_cnt is not None else list(data['frm_idx'].unique())
        data = data[data['frm_idx'].isin(selected_frms)]
        dfs.append(data)

    dfs = pd.concat(dfs, axis=0)
    dfs['id'] = dfs['frm_idx'].astype(str) + dfs['video'].astype(str)
    train_idx = random.sample(list(dfs['id'].unique()), int(len(dfs['frm_idx'].unique()) * train_size))
    map_dict = {k: v for k, v in enumerate(animal_names)}
    id_dict = {v: k for k, v in enumerate(animal_names)}

    for frm_cnt, frm_id in enumerate(dfs['id'].unique()):
        frm_data = dfs[dfs['id'] == frm_id]
        video_path = list(frm_data['video'])[0]
        frm_idx = list(frm_data['frm_idx'])[0]
        video_meta = get_video_meta_data(video_path=video_path)
        if verbose:
            print(f'Processing frame: {frm_cnt + 1}/{len(dfs)} ...')
        if frm_idx > video_meta['frame_count']:
            FrameRangeWarning(msg=f'Frame {frm_idx} could not be read from video {video_path}. The video {video_meta["video_name"]} has {video_meta["frame_count"]} frames', source=dlc_multi_animal_h5_to_yolo_keypoints.__name__)
            continue
        img = read_frm_of_video(video_path=video_path, frame_index=frm_idx, greyscale=greyscale)
        img_h, img_w = img.shape[0], img.shape[1]
        if list(frm_data['id'])[0] in train_idx:
            img_save_path = os.path.join(img_dir, 'train', f'{video_meta["video_name"]}_{frm_idx}.png')
            lbl_save_path = os.path.join(lbl_dir, 'train', f'{video_meta["video_name"]}_{frm_idx}.txt')
        else:
            img_save_path  = os.path.join(img_dir, 'val', f'{video_meta["video_name"]}_{frm_idx}.png')
            lbl_save_path = os.path.join(lbl_dir, 'val', f'{video_meta["video_name"]}_{frm_idx}.txt')
        img_lbl = ''
        frm_data = frm_data.drop(['video', 'id', 'frm_idx'], axis=1).T.iloc[:, 0]
        for track_cnt, animal_name in enumerate(animal_names):
            keypoints = frm_data.T[[idx[0] == animal_name for idx in frm_data.index]].values.reshape(-1, 3)
            keypoints[keypoints[:, 2] != 0.0, 2] = 2
            if frm_cnt == 0 and track_cnt == 0:
                if flip_idx is not None and keypoints.shape[0] != len(flip_idx):
                    raise InvalidInputError(msg=f'The SLEAP data contains data for {keypoints.shape[0]} body-parts, but passed flip_idx suggests {len(flip_idx)} body-parts.', source=dlc_multi_animal_h5_to_yolo_keypoints.__name__)
                elif flip_idx is None:
                    flip_idx = tuple(list(range(0, keypoints.shape[0])))
            instance_str = f'{id_dict[animal_name]} '
            x_coords, y_coords = keypoints[:, 0], keypoints[:, 1]
            min_x, max_x = np.nanmin(x_coords), np.nanmax(x_coords)
            min_y, max_y = np.nanmin(y_coords), np.nanmax(y_coords)
            pad_w, pad_h = padding * img_w, padding * img_h
            min_x, max_x = max(min_x - pad_w / 2, 0), min(max_x + pad_w / 2, img_w)
            min_y, max_y = max(min_y - pad_h / 2, 0), min(max_y + pad_h / 2, img_h)
            bbox_w, bbox_h = max_x - min_x, max_y - min_y
            x_center, y_center = min_x + bbox_w / 2, min_y + bbox_h / 2
            x_center /= img_w
            y_center /= img_h
            bbox_w /= img_w
            bbox_h /= img_h
            x_center = np.clip(x_center, 0.0, 1.0)
            y_center = np.clip(y_center, 0.0, 1.0)
            bbox_w = np.clip(bbox_w, 0.0, 1.0)
            bbox_h = np.clip(bbox_h, 0.0, 1.0)
            keypoints[:, 0] /= img_w
            keypoints[:, 1] /= img_h
            keypoints[:, 0:2] = np.clip(keypoints[:, 0:2], 0.0, 1.0)
            instance_str += f"{x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f} "
            for kp in keypoints:
                instance_str += f"{kp[0]:.6f} {kp[1]:.6f} {int(kp[2])} "
            img_lbl += instance_str.strip() + '\n'
        with open(lbl_save_path, mode='wt', encoding='utf-8') as f:
            f.write(img_lbl)
        cv2.imwrite(img_save_path, img)
    create_yolo_keypoint_yaml(path=save_dir, train_path=img_train_dir, val_path=img_val_dir, names=map_dict, save_path=map_path, kpt_shape=(len(flip_idx), 3), flip_idx=flip_idx)
    timer.stop_timer()
    stdout_success(msg=f'YOLO formated data saved in {save_dir} directory', source=dlc_multi_animal_h5_to_yolo_keypoints.__name__, elapsed_time=timer.elapsed_time_str)

def simba_to_yolo_keypoints(config_path: Union[str, os.PathLike],
                            save_dir: Union[str, os.PathLike],
                            data_dir: Optional[Union[str, os.PathLike]] = None,
                            train_size: float = 0.7,
                            verbose: bool = False,
                            greyscale: bool = False,
                            padding: float = 0.00,
                            flip_idx: Tuple[int, ...] = (1, 0, 2, 4, 3, 5, 6, 7, 8, 9),
                            map_dict: Dict[int, str] = {0: 'mouse'},
                            sample_size: Optional[int] = None,
                            bp_id_idx: Optional[Dict[int, Union[Tuple[int], List[int]]]] = None) -> None:

    """
    Convert pose estimation data from a SimBA project into the YOLO keypoint format, including frame sampling,
    image-label pair creation, bounding box computation, and train/validation splitting.

    .. note::
       For creating the ``flip_idx``, see :func:`simba.third_party_label_appenders.converters.get_yolo_keypoint_flip_idx`.
       For creating the ``bp_id_idx``, see :func:`simba.third_party_label_appenders.converters.get_yolo_keypoint_bp_id_idx`

    :param Union[str, os.PathLike] config_path: Path to the SimBA project `.ini` configuration file.
    :param Union[str, os.PathLike] save_dir: Directory where YOLO-formatted data will be saved. Subdirectories for images/labels (train/val) are created.
    :param Optional[Union[str, os.PathLike] data_dir: Optional directory containing outlier-corrected SimBA pose estimation data. If None, uses path from config.
    :param float train_size: Proportion of samples to allocate to the training set (range 0.1–0.99). Remaining samples go to validation.
    :param bool verbose: If True, prints progress updates to the console.
    :param bool greyscale: If True, saves extracted video frames in greyscale. Otherwise, saves in color.
    :param float padding: Padding added around the bounding box (as a proportion of image dimensions, range 0.0–1.0). Useful if animal body-parts are in a "line".
    :param Tuple[int, ...] flip_idx: Tuple defining symmetric keypoint indices for horizontal flipping. Used to write the `map.yaml` file.
    :param Dict[int, str] map_dict: Dictionary mapping instance IDs to class names. Used in annotation labels and `map.yaml`.
    :param Optional[int] sample_size: If specified, limits the number of randomly sampled frames per video. If None, all frames are used.
    :param Optional[Dict[int, Union[Tuple[int], List[int]]]] bp_id_idx: Optional mapping of instance IDs to keypoint index groups, allowing support for multiple animals per frame. Must match keys in `map_dict`.
    :return: None. Saves YOLO-formatted images and annotations to disk in the `save_dir` location.

    :example:
    >>> simba_to_yolo_keypoints(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini", save_dir=r'C:\troubleshooting\mitra\yolo', sample_size=150, verbose=True)
    """

    timer = SimbaTimer(start=True)

    check_valid_boolean(value=verbose, source=f'{simba_to_yolo_keypoints.__name__} verbose')
    check_valid_boolean(value=greyscale, source=f'{simba_to_yolo_keypoints.__name__} greyscale')
    check_file_exist_and_readable(file_path=config_path)
    check_float(name=f'{simba_to_yolo_keypoints.__name__} padding', value=padding, max_value=1.0, min_value=0.0, raise_error=True)
    check_float(name=f'{simba_to_yolo_keypoints.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
    check_valid_dict(x=map_dict, valid_key_dtypes=(int,), valid_values_dtypes=(str,), min_len_keys=1)
    check_if_dir_exists(in_dir=save_dir)
    check_valid_tuple(x=flip_idx, source=simba_to_yolo_keypoints.__name__, valid_dtypes=(int,), minimum_length=1)
    img_dir, lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
    img_train_dir, img_val_dir = os.path.join(img_dir, 'train'), os.path.join(img_dir, 'val')
    lbl_train_dir, lb_val_dir = os.path.join(lbl_dir, 'train'), os.path.join(lbl_dir, 'val')
    create_directory(paths=[img_train_dir, img_val_dir, lbl_train_dir, lb_val_dir], overwrite=False)
    map_path = os.path.join(save_dir, 'map.yaml')
    if sample_size is not None:
        check_int(name=f'{simba_to_yolo_keypoints.__name__} sample', value=sample_size, min_value=1)
    config = ConfigReader(config_path=config_path)
    if data_dir is not None:
        check_if_dir_exists(in_dir=data_dir, source=f'{simba_to_yolo_keypoints.__name__} data_dir')
    else:
        data_dir = config.outlier_corrected_dir
    data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=[f'.{config.file_type}'], raise_error=True, as_dict=True)
    video_paths = find_files_of_filetypes_in_directory(directory=config.video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True, as_dict=True)
    missing_videos = [x for x in data_paths.keys() if x not in video_paths.keys()]
    if len(missing_videos) > 0:
        NoDataFoundWarning(msg=f'Data files {missing_videos} do not have corresponding videos in the {config.video_dir} directory', source=simba_to_yolo_keypoints.__name__)
    data_w_video = [x for x in data_paths.keys() if x in video_paths.keys()]
    if len(data_w_video) == 0:
        raise NoFilesFoundError(msg=f'None of the data files in {data_dir} have matching videos in the {config.video_dir} directory', source=simba_to_yolo_keypoints.__name__)
    if bp_id_idx is not None:
        check_valid_dict(x=bp_id_idx, valid_key_dtypes=(int,), valid_values_dtypes=(tuple, list,))
        missing_map_dict_keys = [x for x in bp_id_idx.keys() if x not in map_dict.keys()]
        missing_bp_id_keys = [x for x in map_dict.keys() if x not in bp_id_idx.keys()]
        if len(missing_map_dict_keys) > 0:
            raise InvalidInputError(msg=f'Keys {missing_map_dict_keys} exist in bp_id_idx but is not passed in map_dict', source=simba_to_yolo_keypoints.__name__)
        if len(missing_bp_id_keys) > 0:
            raise InvalidInputError(msg=f'Keys {missing_bp_id_keys} exist in map_dict but is not passed in bp_id_idx', source=simba_to_yolo_keypoints.__name__)

    annotations = []
    for file_cnt, video_name in enumerate(data_w_video):
        data = read_df(file_path=data_paths[video_name], file_type=config.file_type)
        check_valid_dataframe(df=data, source=simba_to_yolo_keypoints.__name__, valid_dtypes=Formats.NUMERIC_DTYPES.value)
        video_path = video_paths[video_name]
        check_video_and_data_frm_count_align(video=video_path, data=data, name=data_paths[video_name], raise_error=True)
        frm_cnt = len(data)
        data = data.loc[:, ~data.columns.str.endswith('_p')].reset_index(drop=True)
        data['video'] = video_name
        #data = data.values.reshape(len(data), -1, 2).astype(np.float32)
        #img_w, img_h = video_meta['width'], video_meta['height']
        if sample_size is None:
            video_sample_idx = list(range(0, frm_cnt))
        else:
            video_sample_idx = list(range(0, frm_cnt)) if sample_size > frm_cnt else random.sample(list(range(0, frm_cnt)), sample_size)
        annotations.append(data.iloc[video_sample_idx].reset_index(drop=False))


    annotations = pd.concat(annotations, axis=0).reset_index(drop=True)
    video_names = annotations.pop('video').reset_index(drop=True).values
    train_idx = random.sample(list(annotations['index']), int(len(annotations) * train_size))

    for cnt, (idx, idx_data) in enumerate(annotations.iterrows()):
        vid_path = video_paths[video_names[cnt]]
        video_meta = get_video_meta_data(video_path=vid_path)
        frm_idx, keypoints = idx_data[0], idx_data.values[1:].reshape(-1, 2)
        mask = (keypoints[:, 0] == 0.0) & (keypoints[:, 1] == 0.0)
        keypoints[mask] = np.nan
        if np.all(np.isnan(keypoints)) or np.all(keypoints == 0.0) or np.all(np.isnan(keypoints) | (keypoints == 0.0)):
            continue
        img_lbl = ''; instance_str = f'0 '
        if verbose:
            print(f'Processing image {cnt+1}/{len(annotations)}...')
        file_name = f'{video_meta["video_name"]}.{frm_idx}'
        if frm_idx in train_idx:
            img_save_path, lbl_save_path = os.path.join(img_dir, 'train', f'{file_name}.png'), os.path.join(lbl_dir, 'train', f'{file_name}.txt')
        else:
            img_save_path, lbl_save_path = os.path.join(img_dir, 'val', f'{file_name}.png'), os.path.join(lbl_dir, 'val', f'{file_name}.txt')
        img = read_frm_of_video(video_path=vid_path, frame_index=frm_idx, greyscale=greyscale)
        img_h, img_w = img.shape[0], img.shape[1]
        keypoints_with_id = {}
        if bp_id_idx is not None:
            for k, idx in bp_id_idx.items():
                keypoints_with_id[k] = keypoints.reshape(-1, 2)[idx, :]
        else:
            keypoints_with_id[0] = keypoints.reshape(-1, 2)

        for id, keypoints in keypoints_with_id.items():
            if np.all(np.isnan(keypoints)) or np.all(keypoints == 0.0) or np.all(np.isnan(keypoints) | (keypoints == 0.0)):
               continue
            instance_str = f'{id} '
            x_coords, y_coords = keypoints[:, 0], keypoints[:, 1]
            min_x, max_x = np.nanmin(x_coords), np.nanmax(x_coords)
            min_y, max_y = np.nanmin(y_coords), np.nanmax(y_coords)
            pad_w, pad_h = padding * img_w, padding * img_h
            min_x, max_x = max(min_x - pad_w / 2, 0), min(max_x + pad_w / 2, img_w)
            min_y, max_y = max(min_y - pad_h / 2, 0), min(max_y + pad_h / 2, img_h)
            if max_x - min_x < 1 or max_y - min_y < 1:
                continue
            bbox_w, bbox_h = max_x - min_x, max_y - min_y
            x_center, y_center = min_x + bbox_w / 2, min_y + bbox_h / 2
            x_center /= img_w
            y_center /= img_h
            bbox_w /= img_w
            bbox_h /= img_h
            x_center = np.clip(x_center, 0.0, 1.0)
            y_center = np.clip(y_center, 0.0, 1.0)
            bbox_w = np.clip(bbox_w, 0.0, 1.0)
            bbox_h = np.clip(bbox_h, 0.0, 1.0)
            keypoints[:, 0] /= img_w
            keypoints[:, 1] /= img_h
            keypoints[:, 0:2] = np.clip(keypoints[:, 0:2], 0.0, 1.0)
            visability_col = np.where(np.isnan(keypoints).any(axis=1), 0, 2)
            keypoints = np.nan_to_num(np.hstack((keypoints, visability_col[:, np.newaxis])), nan=0.0)
            mask = (keypoints[:, 0] == 0.0) & (keypoints[:, 1] == 0.0)
            keypoints[mask, 2] = 0.0
            instance_str += f"{x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f} "
            for kp in keypoints:
                instance_str += f"{kp[0]:.6f} {kp[1]:.6f} {int(kp[2])} "
            img_lbl += instance_str.strip() + '\n'
            with open(lbl_save_path, mode='wt', encoding='utf-8') as f:
                f.write(img_lbl)
            cv2.imwrite(img_save_path, img)

    create_yolo_keypoint_yaml(path=save_dir, train_path=img_train_dir, val_path=img_val_dir, names=map_dict, save_path=map_path, kpt_shape=(len(flip_idx), 3), flip_idx=flip_idx)
    timer.stop_timer()
    stdout_success(msg=f'YOLO formated data saved in {save_dir} directory', source=simba_to_yolo_keypoints.__name__, elapsed_time=timer.elapsed_time_str)



#simba_to_yolo_keypoints(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini", save_dir=r'C:\troubleshooting\mitra\yolo', sample_size=150, verbose=True)

#labelme_to_yolo(labelme_dir=r'D:\netholabs\imgs\labels_labelme', save_dir=r'D:\netholabs\imgs\labels_yolo', obb=False, verbose=True)
#split_yolo_train_test_val(data_dir=r'D:\netholabs\imgs\labels_yolo', save_dir=r"D:\netholabs\imgs\yolo_train_test_val", verbose=True)

#labelme_to_img_dir(img_dir=r"C:\troubleshooting\coco_data\labels\train_images", labelme_dir=r'C:\troubleshooting\coco_data\labels\train_')










#df = labelme_to_df(labelme_dir=r'C:\troubleshooting\coco_data\labels\test_read', greyscale=False, pad=False, normalize=False, size='min', save_path=r'C:\Users\sroni\OneDrive\Desktop\labelme_test.csv')
#imgs = ImageMixin().read_all_img_in_dir(dir=r'C:\Users\sroni\OneDrive\Desktop\predefined_sizes')




# imgs = ImageMixin.resize_img_dict(imgs=imgs, size='max')
#



#
# for k, v in imgs.items():
#     print(v.shape)

#




#dlc_to_labelme(dlc_dir=r"D:\TS_DLC\labeled-data\ts_annotations", save_dir=r"C:\troubleshooting\coco_data\labels\test")

#





# x = df.set_index('image_name')['image'].to_dict()
# _b64_dict_to_imgs(x)



# dlc_to_labelme(dlc_dir=r"D:\TS_DLC\labeled-data\ts_annotations", save_dir=r"C:\troubleshooting\coco_data\labels\test")


#
# def dlc_to_coco():

#



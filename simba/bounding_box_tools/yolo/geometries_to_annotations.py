import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from pycocotools import mask
from shapely.geometry import Polygon
from skimage.draw import polygon

from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.checks import check_instance, check_int, check_valid_array
from simba.utils.enums import Formats
from simba.utils.read_write import (get_video_meta_data, read_df,
                                    read_frm_of_video)


def geometry_to_rle(geometry: Union[np.ndarray, Polygon], img_size: Tuple[int, int]):
    """
    Converts a geometry (polygon or NumPy array) into a Run-Length Encoding (RLE) mask, suitable for object detection or segmentation tasks.

    :param geometry: The geometry to be converted into an RLE. It can be either a shapely Polygon or a (n, 2) np.ndarray with vertices.
    :param img_size:  A tuple `(height, width)` representing the size of the image in which the geometry is to be encoded. This defines the dimensions of the output binary mask.
    :return:
    """
    check_instance(source=geometry_to_rle.__name__, instance=geometry, accepted_types=(Polygon, np.ndarray))
    if isinstance(geometry, (Polygon,)):
        geometry = geometry.exterior.coords
    else:
        check_valid_array(data=geometry, source=geometry_to_rle.__name__, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    binary_mask = np.zeros(img_size, dtype=np.uint8)
    rr, cc = polygon(geometry[:, 0].flatten(), geometry[:, 1].flatten(), img_size)
    binary_mask[rr, cc] = 1
    rle = mask.encode(np.asfortranarray(binary_mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def geometries_to_coco(geometries: Dict[str, np.ndarray],
                       video_path: Union[str, os.PathLike],
                       save_dir: Union[str, os.PathLike],
                       version: Optional[int] = 1,
                       description: Optional[str] = None,
                       licences: Optional[str] = None):
    """
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
                       obb: Optional[bool] = False) -> None:
    """
    Converts geometrical shapes (like polygons) into YOLO format annotations and saves them along with corresponding video frames as images.

    :param Dict[Union[str, int], np.ndarray geometries: A dictionary where the keys represent category IDs (either string or int), and the values are NumPy arrays of shape `(n_frames, n_points, 2)`. Each entry in the array represents the geometry of an object in a particular frame (e.g., keypoints or polygons).
    :param Union[str, os.PathLike] video_path: Path to the video file from which frames are extracted. The video is used to extract images corresponding to the geometrical annotations.
    :param Union[str, os.PathLike] save_dir: The directory where the output images and YOLO annotation files will be saved. Images will be stored in a subfolder `images/` and annotations in `labels/`.
    :param verbose: If `True`, prints progress while processing each frame. This can be useful for monitoring long-running tasks. Default is `True`.
    :param sample: If provided, only a random sample of the geometries will be used for annotation. This value represents the number of frames to sample.  If `None`, all frames will be processed. Default is `None`.
    :param obb: If `True`, uses oriented bounding boxes (OBB) by extracting the four corner points of the geometries. Otherwise, axis-aligned bounding boxes (AABB) are used. Default is `False`.
    :return None:

    :example:
    >>> data_path = r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\501_MA142_Gi_CNO_0514.csv"
    >>> animal_data = read_df(file_path=data_path, file_type='csv', usecols=['Nose_x', 'Nose_y', 'Tail_base_x', 'Tail_base_y', 'Left_side_x', 'Left_side_y', 'Right_side_x', 'Right_side_y']).values.reshape(-1, 4, 2).astype(np.int32)
    >>> animal_polygons = GeometryMixin().bodyparts_to_polygon(data=animal_data)
    >>> poygons = GeometryMixin().multiframe_minimum_rotated_rectangle(shapes=animal_polygons)
    >>> animal_polygons = GeometryMixin().geometries_to_exterior_keypoints(geometries=poygons)
    >>> animal_polygons = {0: animal_polygons}
    >>> geometries_to_yolo(geometries=animal_polygons, video_path=r'C:\troubleshooting\mitra\project_folder\videos\501_MA142_Gi_CNO_0514.mp4', save_dir=r"C:\troubleshooting\coco_data", sample=500, obb=True)
    """

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
    for category_cnt, (category_id, category_data) in enumerate(geometries.items()):
        for img_cnt in range(category_data.shape[0]):
            if sample is not None and img_cnt not in samples:
                continue
            else:
                if verbose:
                    print(f'Writing category {category_cnt}, Image: {img_cnt}.')
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

    for k, v in results.items():
        name = k.split(sep='.', maxsplit=2)[0]
        file_name = os.path.join(save_labels_dir, f'{name}.txt')
        with open(file_name, mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(v))



#def geometries_to_yolo_obb(geometries: Dict[Union[str, int], np.ndarray]):



#
#
#
data_path = r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\FL_gq_CNO_0625.csv"
animal_data = read_df(file_path=data_path, file_type='csv', usecols=['Nose_x', 'Nose_y', 'Tail_base_x', 'Tail_base_y', 'Left_side_x', 'Left_side_y', 'Right_side_x', 'Right_side_y']).values.reshape(-1, 4, 2).astype(np.int32)
animal_polygons = GeometryMixin().bodyparts_to_polygon(data=animal_data)
poygons = GeometryMixin().multiframe_minimum_rotated_rectangle(shapes=animal_polygons)
animal_polygons = GeometryMixin().geometries_to_exterior_keypoints(geometries=poygons)
# animal_polygons = GeometryMixin.keypoints_to_axis_aligned_bounding_box(keypoints=animal_polygons)
animal_polygons = {0: animal_polygons}
geometries_to_yolo(geometries=animal_polygons, video_path=r'C:\troubleshooting\mitra\project_folder\videos\FL_gq_CNO_0625.mp4', save_dir=r"C:\troubleshooting\coco_data", sample=500, obb=True)

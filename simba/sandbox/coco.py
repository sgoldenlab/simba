import base64
import json
import multiprocessing
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numba import njit, prange
from pycocotools import mask
from shapely.geometry import Polygon
from skimage.draw import polygon

from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.image_mixin import ImageMixin
from simba.sandbox.keyPoi import _geometries_to_exterior_keypoints_helper
from simba.utils.checks import (check_instance, check_int, check_valid_array,
                                check_valid_lst)
from simba.utils.enums import Defaults, Formats
from simba.utils.read_write import (find_core_cnt, get_video_meta_data,
                                    read_df, read_frm_of_video)


def geometry_to_rle(geometry: Union[np.ndarray, Polygon], img_size: Tuple[int, int]):
    check_instance(source=geometry_to_rle.__name__, instance=geometry, accepted_types=(Polygon, np.ndarray))
    if isinstance(geometry, (Polygon,)):
        geometry = geometry.exterior.coords
    else:
        check_valid_array(data=geometry, source=geometry_to_rle.__name__, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    binary_mask = np.zeros(img_size, dtype=np.uint8)
    rr, cc = polygon(geometry[:, 0].flatten(), geometry[:, 1].flatten(), img_size)
    binary_mask[rr, cc] = 1
    pixels = binary_mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle_counts = ' '.join(map(str, runs))
    compressed_counts = base64.b64encode(rle_counts.encode('ascii')).decode('ascii')
    return {'counts': compressed_counts, 'size': list(binary_mask.shape)}
#
#
# data_path = r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\FRR_gq_Saline_0624.csv"
# animal_data = read_df(file_path=data_path, file_type='csv', usecols=['Nose_x', 'Nose_y', 'Tail_base_x', 'Tail_base_y', 'Left_side_x', 'Left_side_y', 'Right_side_x', 'Right_side_y']).values.reshape(-1, 4, 2)[0:20].astype(np.int32)
# animal_data = animal_data[0]
# geometry_to_rle(geometry=animal_data, img_size=(1000, 1000))


def geometries_to_coco(geometries: Dict[str, np.ndarray],
                       video_path: Union[str, os.PathLike],
                       save_dir: Union[str, os.PathLike],
                       version: Optional[int] = 1,
                       description: Optional[str] = None,
                       licences: Optional[str] = None):


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
            segmentation = {'size': [h, w], 'counts': geometry_to_rle(geometry=img_geometry, img_size=(h, w))['counts']}
            annotation = {'id': annotation_id, 'image_id': img_cnt, 'category_id': category_id, 'bbox': bbox, 'area': a_a, 'iscrowd': 0, 'segmentation': segmentation}
            annotations.append(annotation)
    results['images'] = images
    results['annotations'] = annotations
    with open(os.path.join(save_dir, f"annotations.json"), "w") as final:
        json.dump(results, final)


#
#
#
#
#







    #
    #
    #
    #
    #     for geo_cnt, geo in enumerate(img_geo):
    #         annotation_id = img_cnt+1 * geo_cnt
    #         annotation = {'id': annotation_id, 'image_id': img_cnt,  'category_id': 1}
    #         print(dir(geo))
    #         print(np.array(geo.exterior.coords))

        {"id": 20173, "image_id": 8499, "category_id": 15, "segmentation": {"size": [333, 500],
                                                                            "counts": "e\\d02[:0O1N2XO2jF5S9c00N0001000O1O10M501O2N>CN1TOXG3n8@[G0KN`PX4"},
         "area": 528.0, "bbox": [62, 189, 19, 51], "iscrowd": 0}








data_path = r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\FRR_gq_Saline_0624.csv"
animal_data = read_df(file_path=data_path, file_type='csv', usecols=['Nose_x', 'Nose_y', 'Tail_base_x', 'Tail_base_y', 'Left_side_x', 'Left_side_y', 'Right_side_x', 'Right_side_y']).values.reshape(-1, 4, 2)[0:20].astype(np.int32)
animal_polygons = GeometryMixin().bodyparts_to_polygon(data=animal_data)
#animal_polygons = GeometryMixin().multiframe_minimum_rotated_rectangle(shapes=animal_polygons)
animal_polygons = GeometryMixin().geometries_to_exterior_keypoints(geometries=animal_polygons)
animal_polygons = GeometryMixin.keypoints_to_axis_aligned_bounding_box(keypoints=animal_polygons)
animal_polygons = {0: animal_polygons}


geometries_to_coco(geometries=animal_polygons, video_path=r'C:\troubleshooting\mitra\project_folder\videos\FRR_gq_Saline_0624.mp4', save_dir=r"C:\troubleshooting\coco_data")



#geometries_to_exterior_keypoints(geometries=animal_polygons)

# animal_polygons = [[x] for x in animal_polygons]
#




# import json
#
# file_path = r"C:\Users\sroni\Downloads\sbd_coco_anns\pascal_sbd_train.json"
#
#
# with open(file_path, 'r') as file:
#     data = json.load(file)
#
#

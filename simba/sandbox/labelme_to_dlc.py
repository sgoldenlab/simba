import base64
import io
import itertools
import json
import os
from typing import Optional, Union

import cv2
import numpy as np
import pandas as pd
import PIL

from simba.utils.checks import check_if_dir_exists, check_if_keys_exist_in_dict
from simba.utils.errors import NoFilesFoundError
from simba.utils.read_write import (copy_files_to_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext)


def _b64_to_arr(img_b64):
    """
    Helper to convert byte string (e.g., from labelme, to image in numpy format
    """
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr


def labelme_to_dlc(labelme_dir: Union[str, os.PathLike],
                   scorer: Optional[str] = 'SN',
                   save_dir: Optional[Union[str, os.PathLike]] = None) -> None:

    """
    :param Union[str, os.PathLike] labelme_dir: Directory with labelme json files.
    :param Optional[str] scorer: Name of the scorer (anticipated by DLC as header)
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where to save the DLC annotations. If None, then same directory as labelme_dir with `_dlc_annotations` suffix.
    :return: None

    :example:
    >>> labelme_dir = r'D:\ts_annotations'
    >>> labelme_to_dlc(labelme_dir=labelme_dir)
    """

    check_if_dir_exists(in_dir=labelme_dir)
    annotation_paths = find_files_of_filetypes_in_directory(directory=labelme_dir, extensions=['.json'], raise_error=True)
    results_dict = {}
    images = {}
    for annot_path in annotation_paths:
        with open(annot_path) as f: annot_data = json.load(f)
        check_if_keys_exist_in_dict(data=annot_data, key=['shapes', 'imageData', 'imagePath'], name=annot_path)
        img_name = os.path.basename(annot_data['imagePath'])
        images[img_name] = _b64_to_arr(annot_data['imageData'])
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


labelme_dir = r'D:\ts_annotations'
labelme_to_dlc(labelme_dir=labelme_dir)





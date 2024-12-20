from typing import Dict, Optional, Union

import numpy as np
from numba import jit, prange, typed, types

from simba.mixins.image_mixin import ImageMixin
from simba.utils.checks import check_float
from simba.utils.errors import InvalidInputError


@jit(nopython=True)
def _reduce_img_sizes(imgs: typed.Dict, scale: float):
    results = {}
    img_keys = list(imgs.keys())
    for img_idx in prange(len(img_keys)):
        img = imgs[img_keys[img_idx]]
        target_h, target_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
        row_indices = np.linspace(0, int(img.shape[0] - 1), target_h).astype(np.int32)
        col_indices = np.linspace(0, int(img.shape[1] - 1), target_w).astype(np.int32)
        if img.ndim == 3:
            img = img[row_indices][:, col_indices, :]
        else:
            img = img[row_indices][:, col_indices]
        results[img_keys[img_idx]] = img
    return results

def reduce_img_sizes(imgs: Dict[Union[str, int], np.ndarray], scale: float):
    check_float(name=f'{reduce_img_sizes.__name__} scale', value=scale, max_value=0.99, min_value=0.01)
    if not isinstance(imgs, dict):
        raise InvalidInputError(msg=f'imgs has to be a dict, got {type(imgs)}', source=reduce_img_sizes.__name__)
    clrs = set()
    key_types = set()
    for k, v in imgs.items():
        clrs.add(v.ndim)
        key_types.add(type(k))
    if len(clrs) > 1:
        raise InvalidInputError(msg=f'imgs has to be uniform colors, got {clrs}', source=reduce_img_sizes.__name__)
    if len(key_types) > 1:
        raise InvalidInputError(msg=f'imgs keys has to be int or strings, got {key_types}', source=reduce_img_sizes.__name__)
    if list(clrs)[0] == 3:
        value_type = types.uint8[:, :, :]
    elif list(clrs)[0] == 2:
        value_type = types.uint8[:, :]
    else:
        raise InvalidInputError(msg=f'imgs has to be 2 or 3 dimensions, got {list(clrs)[0]}', source=reduce_img_sizes.__name__)
    if isinstance(list(key_types)[0], type(str)):
        key_type = types.unicode_type
    elif isinstance(list(key_types)[0], type(int)):
        key_type = types.int64
    else:
        raise InvalidInputError(msg=f'imgs keys has to be int or strings, got {list(key_types)[0]}', source=reduce_img_sizes.__name__)
    if list(clrs)[0] == 3:
        results = typed.Dict.empty(key_type=key_type, value_type=value_type)
    else:
        results = typed.Dict.empty(key_type=key_type, value_type=value_type)
    for k, v in imgs.items():
        results[k] = v
    return dict(_reduce_img_sizes(imgs=results, scale=scale))


imgs = ImageMixin.read_all_img_in_dir(dir=r"C:\troubleshooting\two_animals_16_bp_JAG\project_folder\videos\Together_1")
new_imgs = {}
for k, v in imgs.items():
    new_imgs[k] = ImageMixin.img_to_greyscale(v)

new_imgs = reduce_img_sizes(imgs=imgs, scale=0.5)

import cv2

cv2.imshow('sadasdas', new_imgs['count_values_in_ranges_cuda'])
cv2.waitKey(30000)

import pandas as pd
import numpy as np
from typing import Optional
import cv2
from simba.utils.read_write import read_df
from itertools import groupby
from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.checks import check_valid_array


def find_path_loops(data: np.ndarray):
    """
    Compute the loops detected within a 2-dimensional path.

    :param np.ndarray data: Nx2 2-dimensional array with the x and y coordinated represented on axis 1.
    :return: Dictionary with the coordinate tuple(x, y) as keys, and sequential frame numbers as values when animals visited, and re-visited the key coordinate.

    :example:
    >>> data = read_df(file_path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/csv/outlier_corrected_movement_location/SI_DAY3_308_CD1_PRESENT.csv', usecols=['Center_x', 'Center_y'], file_type='csv').values.astype(int)
    >>> find_path_loops(data=data)
    """

    check_valid_array(data=data, source=find_path_loops.__name__, accepted_ndims=(2,), accepted_dtypes=(np.int32, np.int64, np.int8))

    seen = {}
    for i in range(data.shape[0]):
        value = tuple(data[i])
        if value not in seen.keys(): seen[value] = [i]
        else: seen[value].append(i)
    seen_dedup = {}
    for k, v in seen.items():
        seen_dedup[k] = [x for cnt, x in enumerate(v) if cnt == 0 or v[cnt] > v[cnt-1] +1]
    return {k:v for k,v in seen_dedup.items() if len(v) > 1}


data = read_df(file_path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/csv/outlier_corrected_movement_location/SI_DAY3_308_CD1_PRESENT.csv', usecols=['Center_x', 'Center_y'], file_type='csv').values.astype(int)
find_path_loops(data=data)


# linestring = GeometryMixin.to_linestring(data=data)
#
# img = GeometryMixin.view_shapes(shapes=[linestring], size=500)
# cv2.imshow('asdasd', img)
# cv2.waitKey(10000)
#

#
#
# G
#
#
# data = pd.read_csv('/Users/simon/Desktop/envs/simba/troubleshooting/zebrafish/project_folder/csv/outlier_corrected_movement_location/20200318_AB_7dpf_ctl_0003.csv')
#
#
#
#
# data = np.load('/Users/simon/Desktop/envs/simba/simba/simba/sandbox/data.npy')
# data[:, 0] = data[:, 0] + 10
# data[:, 1] = data[:, 1] + 400


# linestring = to_linestring(data=data)
#
# img = GeometryMixin.view_shapes(shapes=[linestring])
# cv2.imshow('asdasd', img)
# cv2.waitKey(10000)


# plt.axis('equal')
# plt.title('Path of Increasingly Larger Circles')
# plt.show()

from typing import Dict, Tuple

import esda
import libpysal as lps
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.checks import check_valid_array, check_valid_dict
from simba.utils.data import create_color_palette, find_ranked_colors
from simba.utils.enums import Formats
from simba.utils.errors import InvalidInputError
from simba.utils.read_write import (get_video_meta_data, read_df,
                                    read_frm_of_video)

QUAD_MAP = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
MORAN_COLORS = {'HH': (215,25,28), 'LH': (171,217,233), 'LL': (44,123,182), 'HL': (253,174,97)}


def morans_local_i(x: np.ndarray,
                   grid: Dict[Tuple[int, ...], Polygon],
                   bg_img: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:

    check_valid_dict(x=grid, valid_key_dtypes=(tuple,), valid_values_dtypes=(Polygon,), min_len_keys=2)
    check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    indices = [(i, j) for i in range(x.shape[0]) for j in range(x.shape[1])]
    if (len([x for x in indices if x not in grid.keys()])) > 0 or (len([x for x in grid.keys() if x not in indices])):
        raise InvalidInputError(msg=f'The size of x ({x.shape}) and the number of keys in grid ({len(grid.keys())}) do not match', source=morans_local_i.__name__)
    df = pd.DataFrame(x.flatten(), index=indices, columns=["value"])
    df['geometry'] = grid.values()
    queen_weights = lps.weights.Queen.from_dataframe(df, silence_warnings=True)
    li = esda.moran.Moran_Local(df['value'], queen_weights)
    li = [[x, y] for x, y in zip(list(li.q), list(li.p_sim))]

    moran_df = pd.DataFrame.from_records(li, columns=['quadrant_type', 'significance'], index=indices)
    moran_df[['hotzone_index', 'outlier_index']] = None
    moran_df['hotzone_index'] = moran_df.apply(lambda x: 1 - x['significance'] if x['quadrant_type'] == 1 else x['hotzone_index'], axis=1)
    moran_df['hotzone_index'] = moran_df.apply(lambda x: - 1 + x['significance'] if x['quadrant_type'] == 3 else x['hotzone_index'], axis=1)
    moran_df['outlier_index'] = moran_df.apply(lambda x: - 1 + x['significance'] if x['quadrant_type'] == 2 else x['outlier_index'], axis=1)
    moran_df['outlier_index'] = moran_df.apply(lambda x: 1 - x['significance'] if x['quadrant_type'] == 4 else x['outlier_index'], axis=1)
    moran_df.fillna(0, inplace=True)

    moran_df['quadrant_code'] = moran_df['quadrant_type'].map(QUAD_MAP)
    moran_df['quadrant_clr'] = moran_df['quadrant_code'].map(MORAN_COLORS)
    moran_df['quadrant'] = list(grid.values())
    clrs = list(moran_df['quadrant_clr'])

    img = GeometryMixin.view_shapes(shapes=list(grid.values()), color_palette=clrs, fill_shapes=True, pixel_buffer=0, bg_img=bg_img)

    return moran_df, img


DATA_PATH = r"D:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\FR_MA152_Saline2_0711.csv"
data_arr = read_df(file_path=DATA_PATH, file_type='csv', usecols=['nose_x', 'nose_y']).values
grid = GeometryMixin.bucket_img_into_grid_square(img_size=(668, 540), bucket_grid_size=(5, 5))[0]
cumsum_time = GeometryMixin().cumsum_coord_geometries(data=data_arr, geometries=grid, fps=30, verbose=False)[-1]

data, img = morans_local_i(x=cumsum_time, grid=grid, bg_img=np.zeros(shape=(668, 540)))

import cv2

cv2.imshow('sdasdasd', img)
cv2.waitKey(10000)



#
#
#
#
# # VIDEO_PATH = '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/2022-06-20_NOB_DOT_4.mp4'
# # video_frm = read_frm_of_video(video_path=VIDEO_PATH)
#
#
#
#
# indices = [f'{i} {j}' for i in range(cumsum_time.shape[0]) for j in range(cumsum_time.shape[1])]
#
# df = pd.DataFrame(cumsum_time.flatten(), index=indices, columns=["value"])
# df['geometry'] = grid.values()
#
# queen_weights = lps.weights.Queen.from_dataframe(df, silence_warnings=True)
# li = esda.moran.Moran_Local(df['value'], queen_weights)
# morans_lst = [[x, y] for x, y in zip(list(li.q), list(li.p_sim))]
#
# moran_df = pd.DataFrame.from_records(morans_lst, columns=['quadrant_type', 'significance'], index=indices)
#
# moran_df[['hotzone_index', 'outlier_index']] = None
# moran_df['hotzone_index'] = moran_df.apply(lambda x: 1 - x['significance'] if x['quadrant_type'] == 1 else x['hotzone_index'], axis=1)
# moran_df['hotzone_index'] = moran_df.apply(lambda x: - 1 + x['significance'] if x['quadrant_type'] == 3 else x['hotzone_index'], axis=1)
# moran_df['outlier_index'] = moran_df.apply(lambda x: - 1 + x['significance'] if x['quadrant_type'] == 2 else x['outlier_index'], axis=1)
# moran_df['outlier_index'] = moran_df.apply(lambda x: 1 - x['significance'] if x['quadrant_type'] == 4 else x['outlier_index'], axis=1)
# moran_df.fillna(0, inplace=True)
#
# moran_df['quadrant_code'] = moran_df['quadrant_type'].map(QUAD_MAP)
# moran_df['quadrant_clr'] = moran_df['quadrant_code'].map(colors)
# moran_df['quadrant'] = list(grid.values())
#
#
# clrs = list(moran_df['quadrant_clr'])
# clrs = [[int(value) for value in sublist] for sublist in clrs]
# clrs = [tuple(i) for i in clrs]
#
# import cv2
# img = GeometryMixin.view_shapes(shapes=list(grid.values()), color_palette=clrs, fill_shapes=True)
# #img = cv2.resize(img, (400, 500))
#
# cv2.imshow('sdasdasd', img)
# cv2.waitKey(10000)
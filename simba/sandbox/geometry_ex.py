import argparse
import os.path
from copy import deepcopy
from typing import Dict, Optional, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simba.mixins.abstract_classes import AbstractFeatureExtraction
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.image_mixin import ImageMixin
from simba.plotting.geometry_plotter import GeometryPlotter
from simba.utils.checks import check_instance, check_int
from simba.utils.enums import Formats
from simba.utils.errors import NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (get_video_meta_data, read_df,
                                    read_frm_of_video)

dpi = matplotlib.rcParams["figure.dpi"]

CONFIG_PATH = "/Users/simon/Desktop/envs/simba/troubleshooting/RI/project_folder/project_config.ini"
VIDEO_NAME = "RI_01_165_clipped"
WHITE = "Animal_1"
BLACK = "Animal_2"

config = ConfigReader(config_path=CONFIG_PATH, create_logger=False)
data_path = os.path.join(
    config.outlier_corrected_dir, f"{VIDEO_NAME}.{config.file_type}"
)
video_path = os.path.join(config.video_dir, f"{VIDEO_NAME}.mp4")
get_video_meta_data(video_path)
df = read_df(file_path=data_path, file_type=config.file_type)
resolution, pixels_per_mm, fps = config.read_video_info(video_name=VIDEO_NAME)

white_animal_bp_names, black_animal_bp_names = (
    config.animal_bp_dict[WHITE],
    config.animal_bp_dict[BLACK],
)
white_animal_cols, black_animal_cols = [], []
for x, y in zip(white_animal_bp_names["X_bps"], white_animal_bp_names["Y_bps"]):
    white_animal_cols.extend((x, y))
for x, y in zip(black_animal_bp_names["X_bps"], black_animal_bp_names["Y_bps"]):
    black_animal_cols.extend((x, y))
white_animal_df, black_animal_df = df[white_animal_cols].astype(int), df[
    black_animal_cols
].astype(int)
white_animal_df_arr = white_animal_df.values.reshape(len(white_animal_df), -1, 2)
black_animal_df_arr = black_animal_df.values.reshape(len(black_animal_df), -1, 2)

white_animal_polygons = GeometryMixin().multiframe_bodyparts_to_polygon(
    data=white_animal_df_arr,
    pixels_per_mm=pixels_per_mm,
    parallel_offset=40,
    verbose=False,
    core_cnt=-1,
)
black_animal_polygons = GeometryMixin().multiframe_bodyparts_to_polygon(
    data=black_animal_df_arr,
    pixels_per_mm=pixels_per_mm,
    parallel_offset=40,
    verbose=False,
    core_cnt=-1,
)


black_animal_center = df[["Nose_2_x", "Nose_2_y"]].values.astype(np.int)
black_animal_circles = GeometryMixin().multiframe_bodyparts_to_circle(
    data=black_animal_center,
    pixels_per_mm=pixels_per_mm,
    parallel_offset=100,
    core_cnt=-1,
)


# white_animal_polygons = GeometryMixin().multiframe_minimum_rotated_rectangle(
#     shapes=white_animal_polygons, verbose=False, core_cnt=-1
# )
# black_animal_polygons = GeometryMixin().multiframe_minimum_rotated_rectangle(
#     shapes=black_animal_polygons, verbose=False, core_cnt=-1
# )

# IF WE WANT, THEN WE CAN VISUALIZE THE ANIMAL GEOMETRIES.
imgs = ImageMixin().slice_shapes_in_imgs(
    imgs=video_path, shapes=white_animal_polygons, verbose=False
)
ImageMixin.img_stack_to_video(
    imgs=imgs,
    save_path="/Users/simon/Desktop/envs/simba/troubleshooting/RI/project_folder/frames/output/stack/test_3.mp4",
    fps=fps,
)

im = imgs[4]
height, width, depth = im.shape
figsize = width / float(dpi), height / float(dpi)

plt.figure(figsize=figsize)
plt.axis("off")
plt.imshow(im)
plt.show()


# Entire_video
GeometryPlotter(
    config_path=CONFIG_PATH,
    geometries=[white_animal_polygons, black_animal_polygons],
    video_name=VIDEO_NAME,
    core_cnt=-1,
    save_path="/Users/simon/Desktop/envs/simba/troubleshooting/RI/project_folder/frames/output/stack/test_2.mp4",
).run()

# #OVERLAP
# overlap = GeometryMixin().multiframe_compute_pct_shape_overlap(shape_1=white_animal_polygons, shape_2=black_animal_polygons, verbose=False)

# #area
# white_animal_area = GeometryMixin().multiframe_area(shapes=white_animal_polygons, pixels_per_mm=pixels_per_mm, verbose=False)
#
# #skeleton
# white_animal_skeleton = [['Ear_left_1', 'Nose_1'],
#                          ['Ear_left_1', 'Ear_right_1'],
#                          ['Ear_right_1', 'Nose_1'],
#                          ['Nose_1', 'Center_1'],
#                          ['Center_1', 'Lat_left_1'],
#                          ['Center_1', 'Lat_right_1'],
#                          ['Center_1', 'Tail_base_1']]
#
# black_animal_skeleton = [['Ear_left_2', 'Nose_2'],
#                          ['Ear_left_2', 'Ear_right_2'],
#                          ['Ear_right_2', 'Nose_2'],
#                          ['Nose_2', 'Center_2'],
#                          ['Center_2', 'Lat_left_2'],
#                          ['Center_2', 'Lat_right_2'],
#                          ['Center_2', 'Tail_base_2']]
#
# white_animal_skeleton = GeometryMixin().multiframe_bodyparts_to_multistring_skeleton(data_df=df, skeleton=white_animal_skeleton, verbose=False)
# black_animal_skeleton = GeometryMixin().multiframe_bodyparts_to_multistring_skeleton(data_df=df, skeleton=black_animal_skeleton, verbose=False)


# union
# bodyparts_to_multistring_skeleton

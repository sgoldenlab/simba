import argparse
import os.path
from copy import deepcopy
from typing import Dict, Optional, Union

import cv2
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
from simba.utils.read_write import (find_core_cnt, get_fn_ext,
                                    get_video_meta_data, read_df, write_df)

CONFIG_PATH = "/Users/simon/Desktop/envs/simba/troubleshooting/RI/project_folder/project_config.ini"
VIDEO_NAME = "RI_01_165_clipped"
WHITE = "Animal_1"
BLACK = "Animal_2"


def pad_img_stack(
    image_dict: Dict[int, np.ndarray], pad_value: Optional[int] = 0
) -> Dict[int, np.ndarray]:
    """
    Pad images in a dictionary stack to have the same dimensions.

    :param Dict[int, np.ndarray] image_dict: A dictionary mapping integer keys to numpy arrays representing images.
    :param Optional[int] pad_value: The value used for padding. Defaults to 0 (black)
    :return Dict[int, np.ndarray]: A dictionary mapping integer keys to numpy arrays representing padded images.
    """

    check_instance(
        source=pad_img_stack.__name__, instance=image_dict, accepted_types=(dict,)
    )
    check_int(
        name=f"{pad_img_stack.__name__} pad_value",
        value=pad_value,
        max_value=255,
        min_value=0,
    )
    max_height = max(image.shape[0] for image in image_dict.values())
    max_width = max(image.shape[1] for image in image_dict.values())
    padded_images = {}
    for key, image in image_dict.items():
        pad_height = max_height - image.shape[0]
        pad_width = max_width - image.shape[1]
        print(image.shape)
        padded_image = np.pad(
            image,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )
        padded_images[key] = padded_image
    return padded_images


def img_stack_to_video(
    imgs: Dict[int, np.ndarray],
    save_path: Union[str, os.PathLike],
    fps: int,
    verbose: Optional[bool] = True,
):

    check_instance(
        source=img_stack_to_video.__name__, instance=imgs, accepted_types=(dict,)
    )
    img_sizes = set()
    for k, v in imgs.items():
        img_sizes.add(v.shape)
    if len(list(img_sizes)) > 1:
        imgs = pad_img_stack(imgs)
    imgs = np.stack(imgs.values())
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    writer = cv2.VideoWriter(
        save_path, fourcc, fps, (imgs[0].shape[1], imgs[0].shape[0])
    )
    for i in range(imgs.shape[0]):
        if verbose:
            print(f"Writing img {i+1}...")
        writer.write(imgs[i])
    writer.release()


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
white_animal_polygons = GeometryMixin().multiframe_minimum_rotated_rectangle(
    shapes=white_animal_polygons, verbose=False, core_cnt=-1
)
black_animal_polygons = GeometryMixin().multiframe_minimum_rotated_rectangle(
    shapes=black_animal_polygons, verbose=False, core_cnt=-1
)

# IF WE WANT, THEN WE CAN VISUALIZE THE ANIMAL GEOMETRIES.
imgs = ImageMixin().slice_shapes_in_imgs(
    imgs=video_path, shapes=white_animal_polygons, verbose=False
)
ImageMixin.img_stack_to_video(
    imgs=imgs,
    save_path="/Users/simon/Desktop/envs/simba/troubleshooting/RI/project_folder/frames/output/stack/test.mp4",
    fps=fps,
)

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
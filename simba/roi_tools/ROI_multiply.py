__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import os
import warnings
from copy import deepcopy
from typing import Union

import pandas as pd

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


from simba.utils.checks import (check_file_exist_and_readable,
                                check_valid_dataframe)
from simba.utils.enums import ConfigKey, Keys, Options, Paths
from simba.utils.errors import NoROIDataError, NotDirectoryError
from simba.utils.printing import stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_config_file)


def create_emty_df(shape_type):
    col_list = None
    if shape_type == Keys.ROI_RECTANGLES.value:
        col_list = [
            "Video",
            "Shape_type",
            "Name",
            "Color name",
            "Color BGR",
            "Thickness",
            "topLeftX",
            "topLeftY",
            "Bottom_right_X",
            "Bottom_right_Y",
            "width",
            "height",
            "Tags",
            "Ear_tag_size",
        ]
    if shape_type == Keys.ROI_CIRCLES.value:
        col_list = [
            "Video",
            "Shape_type",
            "Name",
            "Color name",
            "Color BGR",
            "Thickness",
            "centerX",
            "centerY",
            "radius",
            "Tags",
            "Ear_tag_size",
        ]
    if shape_type == Keys.ROI_POLYGONS.value:
        col_list = [
            "Video",
            "Shape_type",
            "Name",
            "Color name",
            "Color BGR",
            "Thickness",
            "Center_X",
            "Center_Y",
            "vertices",
            "Tags",
            "Ear_tag_size",
        ]
    return pd.DataFrame(columns=col_list)


def multiply_ROIs(config_path: Union[str, os.PathLike],
                  filename: Union[str, os.PathLike]) -> None:

    """
    Reproduce ROIs in one video to all other videos in SimBA project.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file.
    :param Union[str, os.PathLike] filename: Path to video in project for which ROIs should be duplicated in the other videos in the project
    :return: None. The results are stored in the ``/project_folder/logs/measures\ROI_definitions.h5`` of the SimBA project

    :example:
    >>> multiply_ROIs(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini", filename=r"C:\troubleshooting\mitra\project_folder\videos\501_MA142_Gi_CNO_0514.mp4")
    """

    check_file_exist_and_readable(file_path=config_path)
    check_file_exist_and_readable(file_path=filename)
    _, video_name, video_ext = get_fn_ext(filename)
    config = read_config_file(config_path=config_path)
    project_path = config.get(ConfigKey.GENERAL_SETTINGS.value, ConfigKey.PROJECT_PATH.value)
    videos_dir = os.path.join(project_path, "videos")
    roi_coordinates_path = os.path.join(project_path, "logs", Paths.ROI_DEFINITIONS.value)
    if not os.path.isdir(videos_dir):
        raise NotDirectoryError(msg=f'Could not find the videos directory in the SimBA project. SimBA expected a directory at location: {videos_dir}')
    if not os.path.isfile(roi_coordinates_path):
        raise NoROIDataError(msg=f"Cannot multiply ROI definitions: no ROI definitions exist in SimBA project. Could find find a file at expected location {roi_coordinates_path}", source=multiply_ROIs.__name__)

    with pd.HDFStore(roi_coordinates_path) as hdf: roi_data_keys = [x[1:] for x in hdf.keys()]
    missing_keys = [x for x in roi_data_keys if x not in [Keys.ROI_RECTANGLES.value, Keys.ROI_CIRCLES.value, Keys.ROI_POLYGONS.value]]
    if len(missing_keys) > 0:
        raise NoROIDataError(msg=f'The ROI data file {roi_coordinates_path} is corrupted. Missing the following keys: {missing_keys}', source=multiply_ROIs.__name__)

    rectangles_df = pd.read_hdf(path_or_buf=roi_coordinates_path, key=Keys.ROI_RECTANGLES.value)
    circles_df = pd.read_hdf(path_or_buf=roi_coordinates_path, key=Keys.ROI_CIRCLES.value)
    polygon_df = pd.read_hdf(path_or_buf=roi_coordinates_path, key=Keys.ROI_POLYGONS.value)

    check_valid_dataframe(df=rectangles_df, source=f'{multiply_ROIs.__name__} rectangles_df', required_fields=['Video', 'Name'])
    check_valid_dataframe(df=circles_df, source=f'{multiply_ROIs.__name__} circles_df', required_fields=['Video', 'Name'])
    check_valid_dataframe(df=polygon_df, source=f'{multiply_ROIs.__name__} polygon_df', required_fields=['Video', 'Name'])

    videos_w_rectangles = list(rectangles_df["Video"].unique())
    videos_w_circles = list(circles_df["Video"].unique())
    videos_w_polygons = list(polygon_df["Video"].unique())
    videos_w_shapes = list(set(videos_w_rectangles + videos_w_circles + videos_w_polygons))
    if video_name not in videos_w_shapes:
        raise NoROIDataError(msg=f"Cannot replicate ROIs to all other videos: no ROI records exist for {video_name}. Create ROIs for for video {video_name}", source=multiply_ROIs.__name__)

    other_video_file_paths = find_files_of_filetypes_in_directory(directory=videos_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value)
    other_video_file_paths = [x for x in other_video_file_paths if x != filename]
    if len(other_video_file_paths) == 0:
        raise NoROIDataError(msg=f"Cannot replicate ROIs to other videos. No other videos exist in project {videos_dir} directory.", source=multiply_ROIs.__name__)

    r_df = [create_emty_df(Keys.ROI_RECTANGLES.value) if video_name not in videos_w_rectangles else rectangles_df[rectangles_df["Video"] == video_name]][0]
    c_df = [create_emty_df(Keys.ROI_CIRCLES.value) if video_name not in videos_w_circles else circles_df[circles_df["Video"] == video_name]][0]
    p_df = [create_emty_df(Keys.ROI_POLYGONS.value) if video_name not in videos_w_polygons else polygon_df[polygon_df["Video"] == video_name]][0]

    rectangle_results, circle_results, polygon_results = deepcopy(r_df), deepcopy(c_df), deepcopy(p_df)
    for other_video_file_name in other_video_file_paths:
        _, other_vid_name, ext = get_fn_ext(other_video_file_name)
        if len(r_df) > 0:
            x = deepcopy(r_df); x['Video'] = other_vid_name
            rectangle_results = pd.concat([rectangle_results, x], axis=0)
        if len(circle_results) > 0:
            x = deepcopy(c_df); x['Video'] = other_vid_name
            circle_results = pd.concat([circle_results, x], axis=0)
        if len(polygon_results) > 0:
            x = deepcopy(p_df); x['Video'] = other_vid_name
            polygon_results = pd.concat([polygon_results, x], axis=0)

    rectangle_results = rectangle_results.drop_duplicates(subset=["Video", "Name"], keep="first")
    circle_results = circle_results.drop_duplicates(subset=["Video", "Name"], keep="first")
    polygon_results = polygon_results.drop_duplicates(subset=["Video", "Name"], keep="first")

    store = pd.HDFStore(roi_coordinates_path, mode="w")
    store[Keys.ROI_RECTANGLES.value] = rectangle_results
    store[Keys.ROI_CIRCLES.value] = circle_results
    store[Keys.ROI_POLYGONS.value] = polygon_results
    store.close()
    stdout_success(msg=f"ROIs for {video_name} applied to a further {len(other_video_file_paths)} videos (Duplicated rectangles count: {len(r_df)}, circles: {len(c_df)}, polygons: {len(p_df)}).")
    print('\nNext, click on "draw" to modify ROI location(s) or click on "reset" to remove ROI drawing(s)')

#multiply_ROIs(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini", filename=r"C:\troubleshooting\mitra\project_folder\videos\501_MA142_Gi_CNO_0514.mp4")
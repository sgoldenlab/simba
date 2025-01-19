import os
from tkinter import *
from typing import Union

import pandas as pd

from simba.ui.tkinter_functions import TwoOptionQuestionPopUp
from simba.utils.checks import (check_file_exist_and_readable,
                                check_valid_dataframe)
from simba.utils.enums import ConfigKey, Keys, Links, Paths
from simba.utils.errors import NoROIDataError
from simba.utils.printing import stdout_trash
from simba.utils.read_write import get_fn_ext, read_config_file, remove_files


def reset_video_ROIs(config_path: Union[str, os.PathLike],
                     filename: Union[str, os.PathLike]) -> None:

    """
    Delete drawn ROIs for a single video in a SimBA project.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file.
    :param Union[str, os.PathLike] filename: Path to video in project for which ROIs should be duplicated in the other videos in the project
    :return: None. The results are stored in the ``/project_folder/logs/measures\ROI_definitions.h5`` of the SimBA project

    :example:
    >>> reset_video_ROIs(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini", filename=r"C:\troubleshooting\mitra\project_folder\videos\501_MA142_Gi_CNO_0514.mp4")
    """

    check_file_exist_and_readable(file_path=config_path)
    check_file_exist_and_readable(file_path=filename)
    _, video_name, video_ext = get_fn_ext(filename)
    config = read_config_file(config_path=config_path)
    project_path = config.get(ConfigKey.GENERAL_SETTINGS.value, ConfigKey.PROJECT_PATH.value)
    roi_coordinates_path = os.path.join(project_path, "logs", Paths.ROI_DEFINITIONS.value)
    if not os.path.isfile(roi_coordinates_path):
        raise NoROIDataError(msg=f"Cannot reset/delete ROI definitions: no ROI definitions exist in SimBA project. Could find find a file at expected location {roi_coordinates_path}. Create ROIs before deleting ROIs.", source=reset_video_ROIs.__name__)
    with pd.HDFStore(roi_coordinates_path) as hdf: roi_data_keys = [x[1:] for x in hdf.keys()]
    missing_keys = [x for x in roi_data_keys if x not in [Keys.ROI_RECTANGLES.value, Keys.ROI_CIRCLES.value, Keys.ROI_POLYGONS.value]]
    if len(missing_keys) > 0:
        raise NoROIDataError(msg=f'The ROI data file {roi_coordinates_path} is corrupted. Missing the following keys: {missing_keys}', source=reset_video_ROIs.__name__)

    rectangles_df = pd.read_hdf(path_or_buf=roi_coordinates_path, key=Keys.ROI_RECTANGLES.value)
    circles_df = pd.read_hdf(path_or_buf=roi_coordinates_path, key=Keys.ROI_CIRCLES.value)
    polygon_df = pd.read_hdf(path_or_buf=roi_coordinates_path, key=Keys.ROI_POLYGONS.value)
    check_valid_dataframe(df=rectangles_df, source=f'{reset_video_ROIs.__name__} rectangles_df', required_fields=['Video'])
    check_valid_dataframe(df=circles_df, source=f'{reset_video_ROIs.__name__} circles_df', required_fields=['Video'])
    check_valid_dataframe(df=polygon_df, source=f'{reset_video_ROIs.__name__} polygon_df', required_fields=['Video'])
    video_rectangle_roi_records = rectangles_df[rectangles_df["Video"] == video_name]
    video_circle_roi_records = circles_df[circles_df["Video"] == video_name]
    video_polygon_roi_records = polygon_df[polygon_df["Video"] == video_name]
    video_roi_cnt = len(video_rectangle_roi_records) + len(video_circle_roi_records) + len(video_polygon_roi_records)
    if video_roi_cnt == 0:
        raise NoROIDataError(msg=f"Cannot delete ROIs for video {video_name}: no ROI records exist for {video_name}. Create ROIs for for video {video_name} first", source=reset_video_ROIs.__name__)

    store = pd.HDFStore(roi_coordinates_path, mode="w")
    store[Keys.ROI_RECTANGLES.value] = rectangles_df[rectangles_df["Video"] != video_name]
    store[Keys.ROI_CIRCLES.value] = circles_df[circles_df["Video"] != video_name]
    store[Keys.ROI_POLYGONS.value] = polygon_df[polygon_df["Video"] != video_name]
    store.close()
    stdout_trash(msg=f"Deleted ROI records for video {video_name}. Deleted rectangle count: {len(video_rectangle_roi_records)}, circles: {len(video_circle_roi_records)}, polygons: {len(video_polygon_roi_records)}.")

def delete_all_ROIs(config_path: Union[str, os.PathLike]) -> None:
    """
    Launches a pop-up asking if to delete all SimBA roi definitions. If click yes, then the ``/project_folder/logs/measures\ROI_definitions.h5`` of the SimBA project is deleted.

    :param config_path: Path to SimBA project config file.
    :return: None

    :example:
    >>> delete_all_ROIs(config_path=r"C:\troubleshooting\ROI_movement_test\project_folder\project_config.ini")
    """

    question = TwoOptionQuestionPopUp(title="WARNING!", question="Do you want to delete all defined ROIs in the project?", option_one="YES", option_two="NO", link=Links.ROI.value)
    if question.selected_option == "YES":
        check_file_exist_and_readable(file_path=config_path)
        config = read_config_file(config_path=config_path)
        project_path = config.get(ConfigKey.GENERAL_SETTINGS.value, ConfigKey.PROJECT_PATH.value)
        roi_coordinates_path = os.path.join(project_path, "logs", Paths.ROI_DEFINITIONS.value)
        if not os.path.isfile(roi_coordinates_path):
            raise NoROIDataError(msg=f"Cannot delete ROI definitions: no ROI definitions exist in SimBA project. Could find find a file at expected location {roi_coordinates_path}. Create ROIs before deleting ROIs.", source=reset_video_ROIs.__name__)
        else:
            remove_files(file_paths=[roi_coordinates_path], raise_error=True)
            stdout_trash(msg=f"Deleted all ROI records for video for the SimBA project (Deleted file {roi_coordinates_path}). USe the Define ROIs menu to create new ROIs.")
    else:
        pass


#delete_all_ROIs(config_path=r"C:\troubleshooting\ROI_movement_test\project_folder\project_config.ini")
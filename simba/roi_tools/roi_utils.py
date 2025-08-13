import math
import os
import warnings
from copy import copy, deepcopy
from tkinter import *
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import ImageTk
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon

from simba.utils.checks import (check_file_exist_and_readable, check_int,
                                check_str, check_valid_array,
                                check_valid_dataframe, check_valid_tuple,
                                check_video_and_data_frm_count_align)
from simba.utils.enums import (ROI_SETTINGS, ConfigKey, Formats, Keys, Options,
                               Paths)
from simba.utils.errors import (InvalidInputError, NoROIDataError,
                                NotDirectoryError)
from simba.utils.printing import stdout_success, stdout_trash
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, get_video_meta_data,
                                    read_config_file, read_df, read_roi_data)
from simba.utils.warnings import (FrameRangeWarning, NoFileFoundWarning,
                                  VideoFileWarning)
from simba.video_processors.roi_selector import ROISelector
from simba.video_processors.roi_selector_circle import ROISelectorCircle
from simba.video_processors.roi_selector_polygon import ROISelectorPolygon

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

DRAW_FRAME_NAME = "DEFINE SHAPE"

def create_rectangle_entry(rectangle_selector: ROISelector, video_name: str, shape_name: str, clr_name: str, clr_bgr: Tuple[int, int, int], thickness: int, ear_tag_size: int, px_conversion_factor: float):
    return {'Video':             video_name,
            'Shape_type':        ROI_SETTINGS.RECTANGLE.value,
            'Name':              shape_name,
            'Color name':        clr_name,
            'Color BGR':         clr_bgr,
            'Thickness':         thickness,
            'Center_X':          rectangle_selector.center[0],
            'Center_Y':          rectangle_selector.center[1],
            'topLeftX':          rectangle_selector.top_left[0],
            'topLeftY':          rectangle_selector.top_left[1],
            'Bottom_right_X':    rectangle_selector.bottom_right[0],
            'Bottom_right_Y':    rectangle_selector.bottom_right[1],
            'width':             rectangle_selector.width,
            'height':            rectangle_selector.height,
            'width_cm':          round((rectangle_selector.width / px_conversion_factor) / 10, 2),
            'height_cm':         round((rectangle_selector.height / px_conversion_factor) / 10, 2),
            'area_cm':           round((round((rectangle_selector.width / px_conversion_factor) / 10, 2) * round((rectangle_selector.height / px_conversion_factor) / 10, 2)), 2),
            "Tags":             {"Center tag": rectangle_selector.center,
                                 "Top left tag": rectangle_selector.top_left,
                                 "Bottom right tag": rectangle_selector.bottom_right,
                                 "Top right tag": rectangle_selector.top_right_tag,
                                 "Bottom left tag": rectangle_selector.bottom_left_tag,
                                 "Top tag": rectangle_selector.top_tag,
                                 "Right tag": rectangle_selector.right_tag,
                                 "Left tag": rectangle_selector.left_tag,
                                 "Bottom tag": rectangle_selector.bottom_tag},
            'Ear_tag_size':      ear_tag_size}

def create_circle_entry(circle_selector: ROISelectorCircle,  video_name: str, shape_name: str, clr_name: str, clr_bgr: Tuple[int, int, int], thickness: int, ear_tag_size: int, px_conversion_factor: float):
    return {'Video':             video_name,
            'Shape_type':        ROI_SETTINGS.CIRCLE.value,
            'Name':              shape_name,
            'Color name':        clr_name,
            'Color BGR':         clr_bgr,
            'Thickness':         thickness,
            'centerX':           circle_selector.circle_center[0],
            'centerY':           circle_selector.circle_center[1],
            'radius':            circle_selector.circle_radius,
            'radius_cm':         round((circle_selector.circle_radius / px_conversion_factor) / 10, 2),
            'area_cm':           round(math.pi * (round((circle_selector.circle_radius / px_conversion_factor) / 10, 2) **2), 2),
            "Tags":             {"Center tag": circle_selector.circle_center,
                                 "Border tag": circle_selector.left_border_tag},
            'Ear_tag_size':      ear_tag_size}


def create_polygon_entry(polygon_selector: ROISelectorPolygon, video_name: str, shape_name: str, clr_name: str, clr_bgr: Tuple[int, int, int], thickness: int, ear_tag_size: int, px_conversion_factor: float) -> dict:
    return {'Video':                video_name,
            'Shape_type':           ROI_SETTINGS.POLYGON.value,
            'Name':                 shape_name,
            'Color name':           clr_name,
            'Color BGR':            clr_bgr,
            'Thickness':            thickness,
            'Center_X':             polygon_selector.polygon_centroid[0],
            'Center_Y':             polygon_selector.polygon_centroid[1],
            'vertices':             polygon_selector.polygon_arr,
            'center':               tuple(polygon_selector.polygon_centroid),
            'area':                 polygon_selector.polygon_area,
            'max_vertice_distance': polygon_selector.max_vertice_distance,
            "area_cm":              round(polygon_selector.polygon_area / px_conversion_factor, 2),
            'Tags':                 polygon_selector.tags,
            'Ear_tag_size':         ear_tag_size}

def create_duplicated_rectangle_entry(shape_entry: dict, jump_size: int) -> dict:
    shape_entry['Name'] = f'{shape_entry["Name"]}_duplicated'
    shape_entry['Center_X'] = shape_entry['Center_X'] + jump_size
    shape_entry['Center_Y'] = shape_entry['Center_Y'] + jump_size
    shape_entry['topLeftX'] = shape_entry['topLeftX'] + jump_size
    shape_entry['topLeftY'] = shape_entry['topLeftY'] + jump_size
    shape_entry['Bottom_right_X'] = shape_entry['Bottom_right_X'] + jump_size
    shape_entry['Bottom_right_Y'] = shape_entry['Bottom_right_Y'] + jump_size
    new_shape_tags = {}
    for tag_name, tag_value in shape_entry['Tags'].items():
        new_shape_tags[tag_name] = (int(tag_value[0] + jump_size), int(tag_value[1] + jump_size))
    shape_entry['Tags'] = new_shape_tags
    return shape_entry


def create_duplicated_circle_entry(shape_entry: dict, jump_size: int) -> dict:
    out = copy(shape_entry)
    out['Name'] = f'{out["Name"]}_duplicated'
    out['centerX'] = out['centerX'] + jump_size
    out['centerY'] = out['centerY'] + jump_size
    out['Tags'] = {'Center tag': ((out['centerX']), int(out['centerY'])),
                   'Border tag': (int(out['Tags']['Border tag'][0] + jump_size), int(out['Tags']['Border tag'][1] + jump_size))}
    return out

def create_duplicated_polygon_entry(shape_entry: dict, jump_size: int) -> dict:
    shape_entry['Name'] = f'{shape_entry["Name"]}_duplicated'
    shape_entry['Center_X'] = shape_entry['Center_X'] + jump_size
    shape_entry['Center_Y'] = shape_entry['Center_Y'] + jump_size
    shape_entry['vertices'] = shape_entry['vertices'] + jump_size
    new_shape_tags = {}
    for tag_name, tag_value in shape_entry['Tags'].items():
        new_shape_tags[tag_name] = (int(tag_value[0] + jump_size), int(tag_value[1] + jump_size))
    shape_entry['Tags'] = new_shape_tags
    return shape_entry

def get_rectangle_df_headers():
    return ['Video', 'Shape_type', 'Name', 'Color name', 'Color BGR', 'Thickness', 'Center_X', 'Center_Y', 'topLeftX', 'topLeftY', 'Bottom_right_X', 'Bottom_right_Y', 'width', 'height', 'width_cm', 'height_cm', 'area_cm', "Tags", 'Ear_tag_size']

def get_circle_df_headers():
    return ['Video', 'Shape_type', 'Name', 'Color name', 'Color BGR', 'Thickness', 'centerX', 'centerY', 'radius', 'radius_cm', 'area_cm', "Tags", 'Ear_tag_size']

def get_polygon_df_headers():
    return ['Video', 'Shape_type', 'Name', 'Color name', 'Color BGR', 'Thickness', 'Center_X', 'Center_Y', 'vertices', 'center', 'area', 'max_vertice_distance', 'area_cm', "Tags", 'Ear_tag_size']


def set_roi_metric_sizes(roi_dict: dict, px_conversion_factor: Union[int, float]) -> dict:
    """
    Helper to update the metric attributes of a shape dictionary.
    """
    out = {}
    for name, data in roi_dict.items():
        if data['Shape_type'].lower() == ROI_SETTINGS.RECTANGLE.value:
            data['width_cm'] = round((data['width'] / px_conversion_factor) / 10, 2)
            data['height_cm'] = round((data['height'] / px_conversion_factor) / 10, 2)
            data['area_cm'] =  round((round((data['width'] / px_conversion_factor) / 10, 2) * round((data['height'] / px_conversion_factor) / 10, 2)), 2)
        elif data['Shape_type'].lower() == ROI_SETTINGS.CIRCLE.value:
            data['radius_cm'] = round((data['radius'] / px_conversion_factor) / 10, 2)
            data['area_cm'] = round(math.pi * (round((data['radius'] / px_conversion_factor) / 10, 2) ** 2), 2)
        elif data['Shape_type'].lower() == ROI_SETTINGS.POLYGON.value:
            data['area'] = Polygon(data['vertices']).simplify(tolerance=20, preserve_topology=True).area
            data['area_cm'] = round(data['area'] / px_conversion_factor, 2)
            data['max_vertice_distance'] =  np.max(cdist(data['vertices'], data['vertices']).astype(np.int32))
        out[name] = data
    return out


def get_roi_df_from_dict(roi_dict: dict, video_name_nesting: Optional[bool] = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Helper create DataFrames from a shape dictionary.

    If nesting is True, then the roi_dict has video name, and shape name keys.
    """
    rectangles_df, circles_df, polygon_df = pd.DataFrame(columns=get_rectangle_df_headers()), pd.DataFrame(columns=get_circle_df_headers()), pd.DataFrame(columns=get_polygon_df_headers())
    if not video_name_nesting:
        for shape_name, shape_data in roi_dict.items():
            if shape_data['Shape_type'].lower() == ROI_SETTINGS.RECTANGLE.value:
                rectangles_df = pd.concat([rectangles_df, pd.DataFrame([shape_data])], ignore_index=True)
            elif shape_data['Shape_type'].lower() == ROI_SETTINGS.CIRCLE.value:
                circles_df = pd.concat([circles_df, pd.DataFrame([shape_data])], ignore_index=True)
            elif shape_data['Shape_type'].lower() == ROI_SETTINGS.POLYGON.value:
                polygon_df = pd.concat([polygon_df, pd.DataFrame([shape_data])], ignore_index=True)
    else:
        for video_name, video_data in roi_dict.items():
            for shape_name, shape_data in video_data.items():
                if shape_data['Shape_type'].lower() == ROI_SETTINGS.RECTANGLE.value:
                    rectangles_df = pd.concat([rectangles_df, pd.DataFrame([shape_data])], ignore_index=True)
                elif shape_data['Shape_type'].lower() == ROI_SETTINGS.CIRCLE.value:
                    circles_df = pd.concat([circles_df, pd.DataFrame([shape_data])], ignore_index=True)
                elif shape_data['Shape_type'].lower() == ROI_SETTINGS.POLYGON.value:
                    polygon_df = pd.concat([polygon_df, pd.DataFrame([shape_data])], ignore_index=True)

    return (rectangles_df, circles_df, polygon_df)


def get_roi_dict_from_dfs(rectangle_df: pd.DataFrame,
                          circle_df: pd.DataFrame,
                          polygon_df: pd.DataFrame,
                          video_name_nesting: Optional[bool] = False) -> dict:
    """
    Helper create dict from a shape dataframes.
    """

    out = {}
    for idx, row in rectangle_df.iterrows():
        if not video_name_nesting:
            out[row['Name']] = row.to_dict()
        else:
            if row['Video'] not in list(out.keys()):
                out[row['Video']] = {}
            out[row['Video']][row['Name']] = row.to_dict()
    for idx, row in circle_df.iterrows():
        if not video_name_nesting:
            out[row['Name']] = row.to_dict()
        else:
            if row['Video'] not in list(out.keys()):
                out[row['Video']] = {}
            out[row['Video']][row['Name']] = row.to_dict()
    for idx, row in polygon_df.iterrows():
        if not video_name_nesting:
            out[row['Name']] = row.to_dict()
        else:
            if row['Video'] not in list(out.keys()):
                out[row['Video']] = {}
            out[row['Video']][row['Name']] = row.to_dict()
    return out



def get_roi_data(roi_path: Union[str, os.PathLike], video_name: str) -> tuple:
    """ Helper to read in ROI data from disk"""
    rectangles_df, circles_df, polygon_df = pd.DataFrame(columns=get_rectangle_df_headers()), pd.DataFrame(columns=get_circle_df_headers()), pd.DataFrame(columns=get_polygon_df_headers())
    other_roi_dict = {}
    roi_names, roi_dict = [''], {}
    other_video_names_w_rois = ['']

    if os.path.isfile(roi_path):
        in_rectangles_df, in_circles_df, in_polygon_df = read_roi_data(roi_path=roi_path)
        other_video_names_w_rois = list(set(list(in_rectangles_df['Video'].unique()) + list(in_circles_df['Video'].unique()) + list(in_polygon_df['Video'].unique())))
        other_video_names_w_rois = [x for x in other_video_names_w_rois if x != video_name]
        if len(other_video_names_w_rois) == 0: other_video_names_w_rois = ['']
        rectangles_df = in_rectangles_df[in_rectangles_df['Video'] == video_name].reset_index(drop=True)
        circles_df = in_circles_df[in_circles_df['Video'] == video_name].reset_index(drop=True)
        polygon_df = in_polygon_df[in_polygon_df['Video'] == video_name].reset_index(drop=True)
        other_rectangles_df = in_rectangles_df[in_rectangles_df['Video'] != video_name].reset_index(drop=True)
        other_circles_df = in_circles_df[in_circles_df['Video'] != video_name].reset_index(drop=True)
        other_polygon_df = in_polygon_df[in_polygon_df['Video'] != video_name].reset_index(drop=True)
        other_roi_dict = get_roi_dict_from_dfs(rectangle_df=other_rectangles_df, circle_df=other_circles_df, polygon_df=other_polygon_df, video_name_nesting=True)
        if len(rectangles_df) + len(circles_df) + len(polygon_df) > 0:
            roi_names = list(set(list(rectangles_df['Name'].unique()) + list(circles_df['Name'].unique()) + list(polygon_df['Name'].unique())))
            roi_dict = get_roi_dict_from_dfs(rectangle_df=rectangles_df, circle_df=circles_df, polygon_df=polygon_df, video_name_nesting=False)

    return (rectangles_df, circles_df, polygon_df, roi_dict, roi_names, other_roi_dict, other_video_names_w_rois)


def get_roi_data_for_video_name(roi_path: Union[str, os.PathLike], video_name: str):
    in_rectangles_df, in_circles_df, in_polygon_df = read_roi_data(roi_path=roi_path)
    rectangles_df = in_rectangles_df[in_rectangles_df['Video'] == video_name].reset_index(drop=True)
    circles_df = in_circles_df[in_circles_df['Video'] == video_name].reset_index(drop=True)
    polygon_df = in_polygon_df[in_polygon_df['Video'] == video_name].reset_index(drop=True)
    roi_dict = get_roi_dict_from_dfs(rectangle_df=rectangles_df, circle_df=circles_df, polygon_df=polygon_df, video_name_nesting=False)
    return roi_dict



def get_video_roi_data_from_dict(roi_dict: dict, video_name: str) -> dict:
    out = {}
    for video_key, data in roi_dict.items():
        if video_key == video_name:
            for shape_name, shape_data in data.items():
                out[shape_name] = shape_data
    return out


def change_roi_dict_video_name(roi_dict: dict, video_name: str) -> dict:
    out = {}
    for shape_name, shape_data in roi_dict.items():
        new_shape_data = copy(shape_data)
        new_shape_data['Video'] = video_name
        out[shape_name] = new_shape_data
    return out




def get_ear_tags_for_rectangle(center: Tuple[int, int], width: int, height: int) -> Dict[str, Union[int, Tuple[int, int]]]:
    """
    Knowing the center, width, and height of rectangle, return its vertices.

    :param Tuple[int, int] center: The center x and y coordinates of the rectangle
    :param int width: The width of the rectangle in pixels.
    :param Tuple[int, int] width: The width of the rectangle in pixels.
    :return: Dictionary holding the name of the tag as key and coordinates as value.
    :rtype: Dict[str, Union[int, Tuple[int, int]]]
    """

    check_valid_tuple(x=center, source=get_ear_tags_for_rectangle.__name__, accepted_lengths=(2,), valid_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name='width', value=width, min_value=1)
    check_int(name='height', value=height, min_value=1)
    tags = {}
    tags['Top left tag'] = (round((center[1] - (width/2))), round(center[0] - (height/2)))
    tags['Bottom right tag'] = (round(center[1] + (width/2)), round(center[0] + (height/2)))
    tags['Top right tag'] = (round(center[1] + (width/2)), round(center[0] - (height/2)))
    tags['Bottom left tag'] = (round(center[1] - (width / 2)), round(center[0] + (height / 2)))
    tags['Top tag'] = (round(center[1]), round(center[0] - (height / 2)))
    tags['Right tag'] = (round(center[1] + (width / 2)), round(center[0]))
    tags['Left tag'] = (round(center[1] - (width / 2)), round(center[0]))
    tags['Bottom tag'] = (round(center[1]), round(center[0] + (height / 2)))
    return tags



def get_vertices_hexagon(center: Tuple[int, int],
                         radius: int) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """
    Generates the vertices of a regular hexagon centered at a given point with a specified radius.

    :param Tuple[int, int] center:  A tuple (x, y) representing the center coordinates of the hexagon. Must contain exactly two numeric values.
    :param int radius: The radius of the hexagon, which represents the distance from the center to any of the vertices. Must be a positive integer.
    :return: A tuple containing vertices as array and dict.
    :rtype: Tuple[np.ndarray, Dict[str, Tuple[int, int]]]
    """

    check_valid_tuple(x=center, source=get_vertices_hexagon.__name__, accepted_lengths=(2,), valid_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name='radius', value=radius, min_value=1)
    vertices = []
    x_c, y_c = center
    for i in range(6):
        angle_rad = np.deg2rad(60 * i)
        x_i = x_c + radius * np.cos(angle_rad)
        y_i = y_c + radius * np.sin(angle_rad)
        vertices.append((x_i, y_i))

    vertices_dict = {"Center_tag": (center[0], center[1])}
    for tag_id, tag in enumerate(vertices):
        vertices_dict[f"Tag_{tag_id}"] = (round(tag[0]), round(tag[1]))
    return (np.round(np.array(vertices)).astype(np.int32), vertices_dict)


def get_half_circle_vertices(center: Tuple[int, int],
                             radius: int,
                             direction: str,
                             n_points: Optional[int] = 50) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:

    """
    Generates vertices for a half-circle with a given radius and direction, centered at a specific point.

    :param Tuple[int, int] center: A tuple (x, y) representing the center coordinates of the half-circle. Must contain exactly two numeric values.
    :param int radius: The radius of the half-circle. Must be a positive integer.
    :param str direction: The direction in which the half-circle is oriented.
    :param Optional[int] n_points: The number of vertices used to approximate the half-circle. Defaults to 50.
    :return: A tuple containing vertices as array and dict.
    :rtype: Tuple[np.ndarray, Dict[str, Tuple[int, int]]]
    """

    check_valid_tuple(x=center, source=get_vertices_hexagon.__name__, accepted_lengths=(2,), valid_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name='radius', value=radius, min_value=1)
    check_str(name='direction', options=['NORTH', 'SOUTH', 'WEST', 'EAST', 'NORTH-EAST', 'NORTH-WEST', 'SOUTH-EAST', 'SOUTH-WEST'], value=direction)
    x_c, y_c = center
    if direction == "WEST":
        a = np.linspace(np.pi / 2, 3 * np.pi / 2, n_points)
    elif direction == "EAST":
        a = np.linspace(-np.pi / 2, np.pi / 2, n_points)
    elif direction == "SOUTH":
        a = np.linspace(0, np.pi, n_points)
    elif direction == 'SOUTH-WEST':
        a = np.linspace(np.pi / 4, 5 * np.pi / 4, n_points)
    elif direction == 'SOUTH-EAST':
        a = np.linspace(-np.pi / 4, 3 * np.pi / 4, n_points)
    elif direction == 'NORTH-WEST':
        a = np.linspace(3 * np.pi / 4, 7 * np.pi / 4, n_points)
    elif direction == 'NORTH-EAST':
        a = np.linspace(5 * np.pi / 4, 9 * np.pi / 4, n_points)
    else:
        a = np.linspace(np.pi, 2 * np.pi, n_points)
    x, y = x_c + radius * np.cos(a), y_c + radius * np.sin(a)
    vertices = np.round(np.column_stack((x, y))).astype(np.int32)
    shape_center = np.round(np.array(Polygon(vertices).centroid.coords)[0]).astype(np.int32)
    vertices_dict = {"Center_tag": (shape_center[0], shape_center[1])}

    for tag_id in range(vertices.shape[0]):
        vertices_dict[f"Tag_{tag_id}"] = (vertices[tag_id][0], vertices[tag_id][1])
    return (np.round(np.array(vertices)).astype(np.int32), vertices_dict)



def get_triangle_vertices(center: Tuple[int, int], side_length: int, direction: int) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """
    Find equilateral triangle vertices knowing the center, direction and length side.

    :param Tuple[int, int] center: A tuple (x, y) representing the center coordinates of the hexagon. Must contain exactly two numeric values.
    :param int side_length: The length of each of the three sides in pixels.
    :param int direction: The direction to point the vertices.
    :return: 2-part tuple containing vertices as array and dict.
    :rtype: Tuple[np.ndarray, Dict[str, Tuple[int, int]]]
    """
    check_valid_tuple(x=center, source=get_vertices_hexagon.__name__, accepted_lengths=(2,), valid_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name='side_length', value=side_length, min_value=1)
    check_int(name='direction', value=direction, min_value=0, max_value=360)

    direction_radians = np.radians(direction)
    radius = side_length / np.sqrt(3)

    top_vertex = (center[0] + radius * np.cos(direction_radians), center[1] + radius * np.sin(direction_radians))
    vertex2 = (center[0] + radius * np.cos(direction_radians + np.radians(120)), center[1] + radius * np.sin(direction_radians + np.radians(120)))
    vertex3 = (center[0] + radius * np.cos(direction_radians + np.radians(-120)), center[1] + radius * np.sin(direction_radians + np.radians(-120)))
    vertices = np.round(np.array([top_vertex, vertex2, vertex3, top_vertex])).astype(np.int32)
    vertices_dict = {"Center_tag": (int(center[0]), int(center[1])), 'Tag_0':  (int(top_vertex[0]), int(top_vertex[1])), 'Tag_1': (int(vertex2[0]), int(vertex2[1])), 'Tag_2': (int(vertex3[0]), int(vertex3[1]))}
    return (vertices, vertices_dict)



def multiply_ROIs(filename: Union[str, os.PathLike],
                  config_path: Optional[Union[str, os.PathLike]] = None,
                  roi_coordinates_path: Optional[Union[str, os.PathLike]] = None,
                  videos_dir: Optional[Union[str, os.PathLike]] = None) -> None:

    """
    Reproduce ROIs in one video to all other videos in SimBA project.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file.
    :param Union[str, os.PathLike] filename: Path to video in project for which ROIs should be duplicated in the other videos in the project
    :return: None. The results are stored in the ``/project_folder/logs/measures\ROI_definitions.h5`` of the SimBA project

    :example:
    >>> multiply_ROIs(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini", filename=r"C:\troubleshooting\mitra\project_folder\videos\501_MA142_Gi_CNO_0514.mp4")
    """

    if config_path is not None:
        check_file_exist_and_readable(file_path=config_path)
        config = read_config_file(config_path=config_path)
        project_path = config.get(ConfigKey.GENERAL_SETTINGS.value, ConfigKey.PROJECT_PATH.value)
        videos_dir = os.path.join(project_path, "videos")
        roi_coordinates_path = os.path.join(project_path, "logs", Paths.ROI_DEFINITIONS.value)

    check_file_exist_and_readable(file_path=filename)
    _, video_name, video_ext = get_fn_ext(filename)

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

    r_df = [pd.DataFrame(columns=get_rectangle_df_headers()) if video_name not in videos_w_rectangles else rectangles_df[rectangles_df["Video"] == video_name]][0]
    c_df = [pd.DataFrame(columns=get_circle_df_headers()) if video_name not in videos_w_circles else circles_df[circles_df["Video"] == video_name]][0]
    p_df = [pd.DataFrame(columns=get_polygon_df_headers()) if video_name not in videos_w_polygons else polygon_df[polygon_df["Video"] == video_name]][0]

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
    #print('Next, click on "DRAW" to modify ROI location(s) or click on "RESET" to remove ROI drawing(s)')

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

def get_image_from_label(tk_lbl: Label):
    """ Given a tkinter label with an image, retrieve image in array format"""

    if not hasattr(tk_lbl, 'image'):
        raise InvalidInputError(msg=f'The label {tk_lbl} does not have a valid image')
    else:
        tk_img = tk_lbl.image
        pil_image = ImageTk.getimage(tk_img)
        img = np.asarray(pil_image)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_pose_for_roi_ui(pose_path: Union[str, os.PathLike],
                        video_path: Union[str, os.PathLike]) -> Union[None, np.ndarray]:
    video_meta_data = get_video_meta_data(video_path=video_path, raise_error=False)
    if video_meta_data is None:
        VideoFileWarning(msg=f'Cannot plot pose on ROI as cannot read meta data for video {video_path}.', source=get_pose_for_roi_ui.__name__)
        return None
    file_readable = check_file_exist_and_readable(file_path=pose_path, raise_error=False)
    if not file_readable:
        NoFileFoundWarning(msg=f'Cannot plot tracking in ROI window: {pose_path} is unreadable.', source=get_pose_for_roi_ui.__name__)
        return None
    pose_df = read_df(file_path=pose_path, file_type='csv')
    data_align = check_video_and_data_frm_count_align(video=video_path, data=pose_path, raise_error=False)
    if not data_align:
        FrameRangeWarning(msg=f'Cannot plot tracking in ROI window: The data contains {len(pose_path)} frames and the video has {video_meta_data["frame_count"]} frames', source=get_pose_for_roi_ui.__name__)
        return None
    pose_df = pose_df.drop(pose_df.columns[2::3], axis=1)
    pose_data = pose_df.values.reshape(len(pose_df), round(len(pose_df.columns) / 2), 2).astype(np.int32)
    valid_pose = check_valid_array(data=pose_data, source=f'{get_pose_for_roi_ui.__name__} pose_data', accepted_ndims=(3,), accepted_axis_0_shape=[video_meta_data['frame_count']], accepted_dtypes=Formats.INTEGER_DTYPES.value, raise_error=False)
    if not valid_pose:
        FrameRangeWarning(msg=f'Cannot plot tracking in ROI window: The pose data from path {pose_path} is not a 3D numeric array with length {video_meta_data["frame_count"]}', source=get_pose_for_roi_ui.__name__)
    return pose_data


def insert_gridlines_on_roi_img(img: np.ndarray,
                                grid: List[Polygon],
                                color: Tuple[int, int, int],
                                thickness: int) -> np.ndarray:

    if grid is None or len(grid) == 0:
        return img
    else:
        try:
            for polygon in grid:
                cords = np.round(np.array(polygon.exterior.coords)).astype(np.int32)
                img = cv2.polylines(img=img, pts=[cords], isClosed=True, color=color, thickness=thickness, lineType=8)
            return img
        except Exception as e:
            msg = f'Cannot draw gridlines: {e.args}'
            raise InvalidInputError(msg=msg, source=f'{insert_gridlines_on_roi_img.__name__} draw')
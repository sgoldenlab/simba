__author__ = "Simon Nilsson"

import glob
import os
import re
import struct
import sys
from multiprocessing import Lock, Process, Value
from typing import Dict, List, Tuple, Union

import pandas as pd
from matplotlib import cm

import simba
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.enums import Methods, Paths
from simba.utils.read_write import get_fn_ext


class SharedCounter(object):
    """Counter that can be shared across processes on different cores"""

    def __init__(self, initval=0):
        self.val = Value("i", initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


def get_body_part_configurations() -> Dict[str, Union[str, os.PathLike]]:
    """
    Return dict with named body-part schematics of pose-estimation schemas in SimBA installation as keys,
    and paths to the images representing those body-part schematics as values.
    """

    lookup = {}
    simba_dir = os.path.dirname(simba.__file__)
    img_dir = os.path.join(simba_dir, Paths.SCHEMATICS.value)
    names_path = os.path.join(simba_dir, Paths.PROJECT_POSE_CONFIG_NAMES.value)
    check_file_exist_and_readable(file_path=names_path)
    check_if_dir_exists(in_dir=img_dir)
    names_lst = list(pd.read_csv(names_path, header=None)[0])
    img_paths = glob.glob(img_dir + "/*.png")
    img_paths.sort(
        key=lambda v: [
            int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", v)
        ]
    )
    for name, img_path in zip(names_lst, img_paths):
        lookup[name] = {}
        lookup[name]["img_path"] = img_path
    return lookup


def get_bp_config_codes() -> Dict[str, str]:
    """
    Helper to match SimBA project_config.ini [create ensemble settings][pose_estimation_body_parts] to string names.

    """

    return {
        "1 animal; 4 body-parts": "4",
        "1 animal; 7 body-parts": "7",
        "1 animal; 8 body-parts": "8",
        "1 animal; 9 body-parts": "9",
        "2 animals; 8 body-parts": "8",
        "2 animals; 14 body-parts": "14",
        "2 animals; 16 body-parts": "16",
        "MARS": Methods.USER_DEFINED.value,
        "Multi-animals; 4 body-parts": "8",
        "Multi-animals; 7 body-parts": "14",
        "Multi-animals; 8 body-parts": "16",
        "3D tracking": "3D_user_defined",
        "AMBER": "AMBER",
    }


def get_bp_config_code_class_pairs() -> Dict[str, object]:
    """
    Helper to match SimBA project_config.ini [create ensemble settings][pose_estimation_body_parts] setting to feature extraction module class.
    """

    from simba.feature_extractors.amber_feature_extractor import \
        AmberFeatureExtractor
    from simba.feature_extractors.feature_extractor_4bp import \
        ExtractFeaturesFrom4bps
    from simba.feature_extractors.feature_extractor_7bp import \
        ExtractFeaturesFrom7bps
    from simba.feature_extractors.feature_extractor_8bp import \
        ExtractFeaturesFrom8bps
    from simba.feature_extractors.feature_extractor_8bps_2_animals import \
        ExtractFeaturesFrom8bps2Animals
    from simba.feature_extractors.feature_extractor_9bp import \
        ExtractFeaturesFrom9bps
    from simba.feature_extractors.feature_extractor_14bp import \
        ExtractFeaturesFrom14bps
    from simba.feature_extractors.feature_extractor_16bp import \
        ExtractFeaturesFrom16bps
    from simba.feature_extractors.feature_extractor_user_defined import \
        UserDefinedFeatureExtractor

    return {
        "16": ExtractFeaturesFrom16bps,
        "14": ExtractFeaturesFrom14bps,
        "9": ExtractFeaturesFrom9bps,
        "8": {1: ExtractFeaturesFrom8bps, 2: ExtractFeaturesFrom8bps2Animals},
        "7": ExtractFeaturesFrom7bps,
        "4": ExtractFeaturesFrom4bps,
        "user_defined": UserDefinedFeatureExtractor,
        "AMBER": AmberFeatureExtractor,
    }


def get_icons_paths() -> Dict[str, Union[str, os.PathLike]]:
    """
    Helper to get dictionary with icons with the icon names as keys (grabbed from file-name) and their
    file paths as values.
    """

    simba_dir = os.path.dirname(simba.__file__)
    icons_dir = os.path.join(simba_dir, Paths.ICON_ASSETS.value)
    icon_paths = glob.glob(icons_dir + "/*.png")
    icons = {}
    for icon_path in icon_paths:
        _, icon_name, _ = get_fn_ext(icon_path)
        icons[icon_name] = {}
        icons[icon_name]["icon_path"] = icon_path
    return icons


def get_third_party_appender_file_formats() -> Dict[str, str]:
    """
    Helper to get dictionary that maps different third-party annotation tools with different file formats.
    """

    return {
        "BORIS": "csv",
        "ETHOVISION": "xlsx",
        "OBSERVER": "xlsx",
        "SOLOMON": "csv",
        "DEEPETHOGRAM": "csv",
        "BENTO": "annot",
    }


def get_emojis() -> Dict[str, str]:
    """
    Helper to get dictionary of emojis with names as keys and emojis as values. Note, the same emojis are
    represented differently in different python versions.
    """
    python_version = str(f"{sys.version_info.major}.{sys.version_info.minor}")
    if python_version == "3.6":
        return {
            "thank_you": "".join(
                chr(x) for x in struct.unpack(">2H", "\U0001f64f".encode("utf-16be"))
            ),
            "relaxed": "".join(
                chr(x) for x in struct.unpack(">2H", "\U0001F600".encode("utf-16be"))
            ),
            "error": "".join(
                chr(x) for x in struct.unpack(">2H", "\U0001F6A8".encode("utf-16be"))
            ),
            "complete": "".join(
                chr(x) for x in struct.unpack(">2H", "\U0001F680".encode("utf-16be"))
            ),
            "warning": "".join(
                chr(x) for x in struct.unpack(">2H", "\u2757\uFE0F".encode("utf-16be"))
            ),
            "trash": "".join(
                chr(x) for x in struct.unpack(">2H", "\U0001F5D1".encode("utf-16be"))
            ),
        }

    elif python_version == "3.10" or python_version == "3.9":
        return {
            "thank_you": "\U0001f64f".encode("utf-8", "replace").decode(),
            "relaxed": "\U0001F600".encode("utf-8", "replace").decode(),
            "warning": "\u2757\uFE0F".encode("utf-8", "replace").decode(),
            "error": "\U0001F6A8".encode("utf-8", "replace").decode(),
            "complete": "\U0001F680".encode("utf-8", "replace").decode(),
            "trash": "\U0001F5D1".encode("utf-8", "replace").decode(),
        }

    elif python_version == "3.7":
        return {
            "thank_you": "\U0001f64f".encode("utf16", errors="surrogatepass").decode(
                "utf16"
            ),
            "relaxed": "\U0001F600".encode("utf16", errors="surrogatepass").decode(
                "utf16"
            ),
            "error": "\U0001F6A8".encode("utf16", errors="surrogatepass").decode(
                "utf16"
            ),
            "complete": "\U0001F680".encode("utf16", errors="surrogatepass").decode(
                "utf16"
            ),
            "warning": "\u2757\uFE0F".encode("utf16", errors="surrogatepass").decode(
                "utf16"
            ),
            "trash": "\U0001F5D1F".encode("utf16", errors="surrogatepass").decode(
                "utf16"
            ),
        }

    else:
        return {
            "thank_you": "\U0001f64f",
            "relaxed": "\U0001F600",
            "error": "\U0001F6A8",
            "complete": "\U0001F680",
            "trash": "\U0001F5D1",
        }


def get_meta_data_file_headers() -> List[str]:
    """
    Get List of headers for SimBA classifier metadata output.

    :return List[str]:
    """

    return [
        "Classifier_name",
        "RF_criterion",
        "RF_max_features",
        "RF_min_sample_leaf",
        "RF_n_estimators",
        "compute_feature_permutation_importance",
        "generate_classification_report",
        "generate_example_decision_tree",
        "generate_features_importance_bar_graph",
        "generate_features_importance_log",
        "generate_precision_recall_curves",
        "generate_rf_model_meta_data_file",
        "generate_sklearn_learning_curves",
        "learning_curve_data_splits",
        "learning_curve_k_splits",
        "n_feature_importance_bars",
        "over_sample_ratio",
        "over_sample_setting",
        "train_test_size",
        "train_test_split_type",
        "under_sample_ratio",
        "under_sample_setting",
        "class_weight",
    ]


def get_cmaps() -> List[str]:
    """
    Get list of named matplotlib color palettes.
    """
    return [
        "spring",
        "summer",
        "autumn",
        "cool",
        "Wistia",
        "Pastel1",
        "Set1",
        "winter",
        "afmhot",
        "gist_heat",
        "copper",
    ]


def get_categorical_palettes():
    return [
        "Pastel1",
        "Pastel2",
        "Paired",
        "Accent",
        "Dark2",
        "Set1",
        "Set2",
        "Set3",
        "tab10",
        "tab20",
        "tab20b",
        "tab20c",
    ]


def get_color_dict() -> Dict[str, Tuple[int]]:
    """
    Get dict of color names as keys and RGB tuples as values
    """

    return {
        "Grey": (220, 200, 200),
        "Red": (0, 0, 255),
        "Dark-red": (0, 0, 139),
        "Maroon": (0, 0, 128),
        "Orange": (0, 165, 255),
        "Dark-orange": (0, 140, 255),
        "Coral": (80, 127, 255),
        "Chocolate": (30, 105, 210),
        "Yellow": (0, 255, 255),
        "Green": (0, 128, 0),
        "Dark-grey": (105, 105, 105),
        "Light-grey": (192, 192, 192),
        "Pink": (178, 102, 255),
        "Lime": (204, 255, 229),
        "Purple": (255, 51, 153),
        "Cyan": (255, 255, 102),
        "White": (255, 255, 255),
        "Black": (0, 0, 0),
        "Darkgoldenrod": (184, 134, 11),
        "Olive": (109, 113, 46),
        "Seagreen": (46, 139, 87),
        "Dodgerblue": (30, 144, 255),
        "Springgreen": (0, 255, 127),
        "Firebrick": (178, 34, 34),
        "Indigo": (63, 15, 183),
    }


def get_named_colors() -> List[str]:
    """
    Get list of named matplotlib colors.
    """
    return [
        "red",
        "pink",
        "lime",
        "gold",
        "coral",
        "lavender",
        "sienna",
        "tomato",
        "grey",
        "azure",
        "crimson",
        "lightgrey",
        "aqua",
        "plum",
        "blue",
        "teal",
        "maroon",
        "green",
        "black",
        "deeppink",
        "darkgoldenrod",
        "purple",
        "olive",
        "seagreen",
        "dodgerblue",
        "springgreen",
        "firebrick",
        "indigo",
        "white",
    ]


def create_color_palettes(no_animals: int, map_size: int) -> List[List[int]]:
    """
    Create list of lists of bgr colors, one for each animal. Each list is pulled from a different palette
    matplotlib color map.

    :param int no_animals: Number of different palette lists
    :param int map_size: Number of colors in each created palette.
    :return List[List[int]]:  BGR colors

    :example:
    >>> create_color_palettes(no_animals=2, map_size=2)
    >>> [[[255.0, 0.0, 255.0], [0.0, 255.0, 255.0]], [[102.0, 127.5, 0.0], [102.0, 255.0, 255.0]]]
    """
    colorListofList = []
    cmaps = [
        "spring",
        "summer",
        "autumn",
        "cool",
        "Wistia",
        "Pastel1",
        "Set1",
        "winter",
        "afmhot",
        "gist_heat",
        "copper",
        "viridis",
        "Set3",
        "Set2",
        "Paired",
        "seismic",
        "prism",
        "ocean",
    ]

    for colormap in range(no_animals):
        if hasattr(cm, "cmap_d") and colormap in cm.cmap_d:
            currColorMap = cm.get_cmap(cmaps[colormap], map_size)
        else:
            currColorMap = cm.get_cmap("spring", map_size)
        currColorList = []
        for i in range(currColorMap.N):
            rgb = list((currColorMap(i)[:3]))
            rgb = [i * 255 for i in rgb]
            rgb.reverse()
            currColorList.append(rgb)
        colorListofList.append(currColorList)
    return colorListofList


def cardinality_to_integer_lookup() -> Dict[str, int]:
    """
    Create dictionary that maps cardinal compass directions to integers.

    :example:
    >>> data = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    >>> [cardinality_to_integer_lookup()[d] for d in data]
    >>> [0, 1, 2, 3, 4, 5, 6, 7]
    """

    return {"N": 0, "NE": 1, "E": 2, "SE": 3, "S": 4, "SW": 5, "W": 6, "NW": 7}


def integer_to_cardinality_lookup():
    """
    Create dictionary that maps integers to cardinal compass directions.
    """
    return {0: "N", 1: "NE", 2: "E", 3: "SE", 4: "S", 5: "SW", 6: "W", 7: "NW"}


def percent_to_crf_lookup() -> Dict[str, int]:
    """
    Create dictionary that matches human-readable percent values to FFmpeg Constant Rate Factor (CRF)
    values that regulates video quality in CPU codecs. Higher CRF values translates to lower video quality and reduced
    file sizes.
    """
    return {
        "10": 37,
        "20": 34,
        "30": 31,
        "40": 28,
        "50": 25,
        "60": 22,
        "70": 19,
        "80": 16,
        "90": 13,
        "100": 10,
    }

def percent_to_qv_lk():
    """
    Create dictionary that matches human-readable percent values to FFmpeg regulates video quality in CPU codecs.
    Higher FFmpeg quality scores maps to smaller, lower quality videos. Used in some AVI codecs such as 'divx' and 'mjpeg'.
    """
    return {100: 3,
            90: 5,
            80: 7,
            70: 9,
            60: 11,
            50: 13,
            40: 15,
            30: 17,
            20: 19,
            10: 21}

def get_ffmpeg_crossfade_methods():
    return ['fade',
            'fadeblack',
            'fadewhite',
            'distance',
            'wipeleft',
            'wiperight',
            'wipeup',
            'wipedown',
            'sideleft',
            'sideright',
            'sideup',
            'sidedown',
            'smoothleft',
            'smoothright',
            'smoothup',
            'smoothdown',
            'circlecrop',
            'rectcrop',
            'circleclose',
            'circleopen',
            'horzclose',
            'horzopen',
            'vertclose',
            'vertopen',
            'diagbl',
            'diagbr',
            'diagtl',
            'diagtr',
            'hlslice',
            'hrslice',
            'vuslice',
            'vdslice',
            'dissolve',
            'pixelize',
            'radial',
            'hblur',
            'wipetl',
            'wipetr',
            'wipebl',
            'wipebr',
            'fadegrays',
            'squeezev',
            'squeezeh',
            'zoomin',
            'hlwind',
            'hrwind',
            'vuwind',
            'vdwind',
            'coverleft',
            'coverright',
            'cobverup',
            'coverdown',
            'revealleft',
            'revealright',
            'revealup',
            'revealdown']


def video_quality_to_preset_lookup() -> Dict[str, str]:
    """
    Create dictionary that matches human-readable video quality settings to FFmpeg presets for GPU codecs.
    """
    return {"Low": "fast", "Medium": "medium", "High": "slow"}


def get_log_config():
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s|%(name)s||%(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%SZ",
                # "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            }
        },
        "handlers": {
            "file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "default",
                "mode": "a",
                "backupCount": 5,
                "maxBytes": 5000000,
            }
        },
        "loggers": {"": {"level": "INFO", "handlers": ["file_handler"]}},
    }


#
# def rao_spacing_critical_values():
#     {4.0: 186.45,
#      5.0: 183.44,
#      6.0: 180.65,
#      7.0: 177.83,
#      8.0: 175.68,
#      9.0: 173.68,
#      10.0: 171.98,
#      11.0: 170.45,
#      12.0: 169.09,
#      13.0: 167.87,
#      14.0: 166.76,
#      15.0: 165.75,
#      16.0: 164.83,
#      17.0: 163.98,
#      18.0: 163.2,
#      19.0: 162.47,
#      20.0: 161.79,
#      21.0: 161.16,
#      22.0: 160.56,
#      23.0: 160.01,
#      24.0: 159.48,
#      25.0: 158.99,
#      26.0: 158.52,
#      27.0: 158.07,
#      28.0: 157.65,
#      29.0: 157.25,
#      30.0: 156.87,
#      35.0: 155.19,
#      40.0: 153.82,
#      45.0: 152.68,
#      50.0: 151.7,
#      75.0: 148.34,
#      100.0: 146.29,
#      150.0: 143.83,
#      200.0: 142.35,
#      300.0: 140.57,
#      400.0: 139.5,
#      500.0: 138.77,
#      600.0: 138.23,
#      700.0: 137.8,
#      800.0: 137.46,
#      900.0: 137.18,
#      1000.0: 136.94}

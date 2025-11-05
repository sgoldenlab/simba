__author__ = "Simon Nilsson"

import glob
import os
import platform
import random
import re
import struct
import subprocess
import sys
import tkinter as tk
from copy import copy
from datetime import datetime
from multiprocessing import Lock, Value
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import matplotlib.font_manager
import pandas as pd
import psutil
import pyglet
from matplotlib import cm
from tabulate import tabulate

import simba
from simba.utils.checks import (check_ffmpeg_available,
                                check_file_exist_and_readable,
                                check_if_dir_exists, check_int, check_str,
                                check_valid_dict, check_valid_tuple, check_instance)
from simba.utils.enums import (OS, UML, Defaults, FontPaths, Formats, Methods,
                               Options, Paths, Keys)
from simba.utils.errors import FFMPEGNotFoundError, NoFilesFoundError, InvalidInputError
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, get_video_meta_data)
from simba.utils.warnings import NoDataFoundWarning

if platform.system() == OS.WINDOWS.value:
    from pyglet.libs.win32 import constants
    constants.COINIT_MULTITHREADED = 0x2  # 0x2 = COINIT_APARTMENTTHREADED


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
        "SimBA BLOB Tracking": Methods.SIMBA_BLOB.value,
        "FaceMap": Methods.FACEMAP.value,
        "SuperAnimal-TopView": Methods.SUPER_ANIMAL_TOPVIEW.value
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

def load_simba_fonts():
    """ Load fonts defined in simba.utils.enums.FontPaths into memory"""
    simba_dir = os.path.dirname(simba.__file__)
    font_enum = {i.name: i.value for i in FontPaths}
    for k, v in font_enum.items():
        pyglet.font.add_file(os.path.join(simba_dir, v))

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
        return {"thank_you": "".join(chr(x) for x in struct.unpack(">2H", "\U0001f64f".encode("utf-16be"))),
                "relaxed": "".join(chr(x) for x in struct.unpack(">2H", "\U0001F600".encode("utf-16be"))),
                "error": "".join(chr(x) for x in struct.unpack(">2H", "\U0001F6A8".encode("utf-16be"))),
                "complete": "".join(chr(x) for x in struct.unpack(">2H", "\U0001F680".encode("utf-16be"))),
                "warning": "".join(chr(x) for x in struct.unpack(">2H", "\u2757\uFE0F".encode("utf-16be"))),
                "trash": "".join(chr(x) for x in struct.unpack(">2H", "\U0001F5D1".encode("utf-16be"))),
                "information": "".join(chr(x) for x in struct.unpack(">2H", "\U0001F4DD".encode("utf-16be")))}  # 📝 memo

    elif python_version == "3.10" or python_version == "3.9":
        return {
            "thank_you": "\U0001f64f".encode("utf-8", "replace").decode(),
            "relaxed": "\U0001F600".encode("utf-8", "replace").decode(),
            "warning": "\u2757\uFE0F".encode("utf-8", "replace").decode(),
            "error": "\U0001F6A8".encode("utf-8", "replace").decode(),
            "complete": "\U0001F680".encode("utf-8", "replace").decode(),
            "trash": "\U0001F5D1".encode("utf-8", "replace").decode(),
            "information": "\U0001F4DD".encode("utf-8", "replace").decode(),  # 📝 memo
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
            "trash": "\U0001F5D1".encode("utf16", errors="surrogatepass").decode(
                "utf16"
            ),
            "information": "\U0001F4DD".encode("utf16", errors="surrogatepass").decode(
                "utf16"
            ),  # 📝 memo
        }

    else:
        return {
            "thank_you": "\U0001f64f",
            "relaxed": "\U0001F600",
            "error": "\U0001F6A8",
            "complete": "\U0001F680",
            "warning": "\u2757\uFE0F",
            "trash": "\U0001F5D1",
            "information": "\U0001F4DD",
             # 📝 memo
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
        'rf_max_depth',
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


def get_color_dict() -> Dict[str, Tuple[int, int, int]]:
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



def get_random_color_palette(n_colors: int):
    """ Get a random color palette with N random colors."""
    check_int(name=f'{get_random_color_palette.__name__} n_colors', value=n_colors, min_value=1, raise_error=True)
    return [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(n_colors)]

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


def gpu_quality_to_cpu_quality_lk():
    return {'fast': 34,
            'medium': 23,
            'slow': 13}

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


def get_labelling_img_kbd_bindings() -> dict:
    """
    Returns dictionary of tkinter keyboard bindings.

    .. note::
        Change ``kbd`` values to change keyboard shortcuts. For example:

        Some possible examples:
            <Key>, <KeyPress>, <KeyRelease>: Binds to any key press or release.
            <KeyPress-A>, <Key-a>: Binds to the 'a' key press (case sensitive).
            <Up>, <Down>, <Left>, <Right>: Binds to the arrow keys.
            <Control-KeyPress-A>, <Control-a>: Binds to Ctrl + A or Ctrl + a
    """
    return \
        {'frame+1': # MOVE FORWARD 1 FRAME
             {'label': 'Right Arrow = +1 frame',
              'kbd': "<Right>"},
         'frame-1': # MOVE BACK 1 FRAME
             {'label': 'Left Arrow = -1 frame',
              'kbd': "<Left>"},
         'save': # SAVE CURRENT ANNOTATIONS STATS TO DISK
             {'label': 'Ctrl + s = Save annotations file',
              'kbd': "<Control-s>"},
         'frame+1_keep_choices': # MOVE FORWARD 1 FRAME AND KEEP ANNOTATION SELECTIONS OF THE CURRENT FRAME
             {'label': 'Ctrl + a = +1 frame and keep choices',
              'kbd': "<Control-a>"},
         'frame-1_keep_choices':  # MOVE BACKWARDS 1 FRAME AND KEEP ANNOTATION SELECTIONS OF THE CURRENT FRAME
             {'label': 'Ctrl + q = -1 frame and keep choices',
              'kbd': "<Control-q>"},
         'print_annotation_statistic': # PRINT ANNOTATION STATISTICS
             {'label': 'Ctrl + p = Show annotation statistics',
              'kbd': "<Control-p>"},
         'last_frame':  # SHOW LAST FRAME
             {'label': 'Ctrl + l = Last frame',
              'kbd': "<Control-l>"},
         'first_frame':  # SHOW FIRT FRAME
             {'label': 'Ctrl + o = First frame',
              'kbd': "<Control-o>"}
         }

def get_labelling_video_kbd_bindings() -> dict:
    """
    Returns a dictionary of OpenCV-compatible keyboard bindings for video labeling.

    Notes:
        - Change the `kbd` values to customize keyboard shortcuts.
        - OpenCV key codes differ from Tkinter bindings (see `get_labelling_img_kbd_bindings`).
        - Use either single-character strings (e.g. 'p') or integer ASCII codes (e.g. 32 for space bar).

    Examples:
        Remap space bar to Pause/Play:
            {'Pause/Play': {'label': 'Space = Pause/Play', 'kbd': 32}}
    """

    bindings = {
        'Pause/Play': {
            'label': 'p = Pause/Play',
            'kbd': 'p'
        },
        'forward_two_frames': {
            'label': 'o = +2 frames',
            'kbd': 'o'
        },
        'forward_ten_frames': {
            'label': 'e = +10 frames',
            'kbd': 'e'
        },
        'forward_one_second': {
            'label': 'w = +1 second',
            'kbd': 'w'
        },
        'backwards_two_frames': {
            'label': 't = -2 frames',
            'kbd': 't'
        },
        'backwards_ten_frames': {
            'label': 's = -10 frames',
            'kbd': 's'
        },
        'backwards_one_second': {
            'label': 'x = -1 second',
            'kbd': 'x'
        },
        'close_window': {
            'label': 'q = Close video window',
            'kbd': 'q'
        },
    }


    #PERFORM CHECKS THAT BINDINGS ARE DEFINED CORRECTLY.
    check_valid_dict( x=bindings, valid_key_dtypes=(str,), valid_values_dtypes=(dict,), source=f'{get_labelling_video_kbd_bindings.__name__} bindings')
    cleaned_bindings = {}
    for action, config in bindings.items():
        check_valid_dict(x=config, valid_key_dtypes=(str,), valid_values_dtypes=(str, int), required_keys=('label', 'kbd'))
        kbd_val = config['kbd']
        check_str(value=config['label'], allow_blank=False, raise_error=True, name=f'{get_labelling_video_kbd_bindings.__name__} action')
        if check_int(name=f'{action} kbd', value=kbd_val, raise_error=False)[0]:
            new_config = copy(config)
            new_config['kbd'] = int(kbd_val)
            cleaned_bindings[action] = new_config
        else:
            cleaned_bindings[action] = config

    return cleaned_bindings


def get_fonts():
    """ Returns a dictionary with all fonts available in OS, with the font name as key and font path as value"""
    font_dict = {f.name: f.fname for f in matplotlib.font_manager.fontManager.ttflist if not f.name.startswith('.')}
    if len(font_dict) == 0:
        NoDataFoundWarning(msg='No fonts found on disk using matplotlib.font_manager', source=get_fonts.__name__)
    if platform.system() == OS.WINDOWS.value:
        font_dict = {key: str(Path(value.replace('C:', '')).as_posix()) for key, value in font_dict.items()}
    return font_dict

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

def get_model_names():
    model_names_dir = os.path.join(os.path.dirname(simba.__file__), Paths.UNSUPERVISED_MODEL_NAMES.value)
    return list(pd.read_parquet(model_names_dir)[UML.NAMES.value])

def win_to_wsl_path(win_path: Union[str, os.PathLike]) -> str:
    """Helper to convert a windows path name, to a WSL path name"""
    result = subprocess.run(["wsl.exe", "wslpath", win_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"WSL path conversion failed: {result.stderr}")
    return result.stdout.strip()


def get_available_ram():
    total = psutil.virtual_memory().total
    available = psutil.virtual_memory().available
    total_mb = total / (1024 ** 2)
    available_mb = available / (1024 ** 2)

    results = {}
    results["bytes"] = total
    results["available_bytes"] = available
    results["megabytes"] = total_mb
    results["available_mb"] = available_mb
    results["gigabytes"] = total_mb / 1024
    results["available_gb"] = available_mb / 1024

    return results


def get_current_time():
    return datetime.now().strftime("%H:%M:%S")


def get_display_resolution() -> Tuple[int, int]:
    """
    Helper to get main monitor / display resolution.

    .. note::
       May return the virtual geometry in multi-display setups. To return the resolution of each available monitor in mosaic, see :func:`simba.utils.lookups.get_monitor_info`.

    """
    root = tk.Tk()
    root.withdraw()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return (width, height)


def get_img_resize_info(img_size: Tuple[int ,int],
                        display_resolution: Optional[Tuple[int, int]] = None,
                        max_height_ratio: float = 0.5,
                        max_width_ratio: float = 0.5,
                        min_height_ratio: float = 0.0,
                        min_width_ratio: float = 0.0) -> Tuple[int, int, float, float]:
    """
    Calculates the new dimensions and scaling factors needed to resize an image while preserving its
    aspect ratio so that it fits within a given portion of the display resolution.

    :param Tuple[int, int] img_size : The original size of the image as (width, height).
    :param Optional[Tuple[int, int]] display_resolution: Optional resolution of the display as (width, height). If none, then grabs the resolution of the main monitor.
    :param float max_height_ratio: The maximum allowed height of the image as a fraction of the display height (default is 0.5).
    :param float max_width_ratio: The maximum allowed width of the image as a fraction of the display width (default is 0.5).
    :returns: Length 4 tuple with resized width, resized height, downscale factor, and upscale factor
    :rtype: Tuple[int, int, float, float]
    """

    if display_resolution is None:
        _, display_resolution = get_monitor_info()
    max_width = round(display_resolution[0] * max_width_ratio)
    max_height = round(display_resolution[1] * max_height_ratio)
    min_width = round(display_resolution[0] * min_width_ratio)
    min_height = round(display_resolution[1] * min_height_ratio)

    if img_size[1] > max_width or img_size[0] > max_height:
        width_ratio = max_width / img_size[0]
        height_ratio = max_height / img_size[1]
        downscale_factor = min(width_ratio, height_ratio)
        upscale_factor = 1 / downscale_factor
        new_width = round(img_size[0] * downscale_factor)
        new_height = round(img_size[1] * downscale_factor)
        return new_width, new_height, downscale_factor, upscale_factor


    elif img_size[1] < min_width or img_size[0] < min_height:
        width_ratio = min_width / img_size[0]
        height_ratio = min_height / img_size[1]
        scale = max(width_ratio, height_ratio)  # ensures both dimensions meet or exceed min
        new_width = round(round(img_size[0] * scale))
        new_height = round(round(img_size[1] * scale))
        return new_width, new_height, scale, 1 / scale

    else:
        return img_size[0], img_size[1], 1, 1

def is_running_in_ide():
    return hasattr(sys, 'ps1') or sys.flags.interactive


def get_monitor_info() -> Tuple[Dict[int, Dict[str, Union[int, bool]]], Tuple[int, int]]:
    """
    Helper to get main monitor / display resolution.

    .. note::
       Returns dict containing the resolution of each available monitor. To get the virtual geometry, see :func:`simba.utils.lookups.get_display_resolution`, and tuple of main monitor width and height.
    """
    monitors = pyglet.canvas.get_display().get_screens()
    results = {}
    for monitor_cnt, monitor_info in enumerate(monitors):
        primary = True if monitor_info.x == 0 and monitor_info.y == 0 else False
        results[monitor_cnt] = {'width': monitor_info.width,
                                 'height': monitor_info.height,
                                 'primary': primary}

    main_monitor = next(({'width': v['width'], 'height': v['height']} for v in results.values() if v.get('primary')), {'width': next(iter(results.values()))['width'], 'height': next(iter(results.values()))['height']})

    return results, (int(main_monitor['width']), int(main_monitor['height']))



def get_table(data: Dict[str, Any],
              headers: Optional[Tuple[str, str]] = ("SETTING", "VALUE"),
              tablefmt: str = "grid") -> str:
    """
     Create a formatted table string from dictionary data using the tabulate library.

     Converts a dictionary into a formatted table string suitable for display
     or printing. Each key-value pair in the dictionary becomes a row in the table.

     :param Dict[str, Any] data: Dictionary containing the data to be formatted as a table.  Keys become the first column, values become the second column.
     :param Optional[Tuple[str, str]] headers: Tuple of two strings representing the column headers. Default is ("SETTING", "VALUE").
     :param Literal["grid"] tablefmt: Table format style. For options, see simba.utils.enums.Formats.VALID_TABLEFMT
     :return str: Formatted table string ready for display or printing.

     :example:
     >>> data = {"fps": 30, "width": 1920, "height": 1080, "frame_count": 3000}
     >>> table = get_table(data=data, headers=("PARAMETER", "VALUE"))
     """

    check_valid_dict(x=data, valid_key_dtypes=(str,), min_len_keys=1, source=f'{get_table.__name__} data')
    check_valid_tuple(x=headers, source=f'{get_table.__name__} data', accepted_lengths=(2,), valid_dtypes=(str,))
    check_str(name=f'{get_table.__name__} tablefmt', value=tablefmt, options=Formats.VALID_TABLEFMT.value, raise_error=True)
    table_view = [[key, data[key]] for key in data]
    return tabulate(table_view, headers=headers, tablefmt=tablefmt)




def print_video_meta_data(data_path: Union[str, os.PathLike]) -> None:
    """
    Print video metadata as formatted tables to the console.

    This function reads video metadata from either a single video file or all video files
    in a directory, then prints the metadata as formatted tables.

    .. seealso::
       To get video metadata as a dictionary without printing, use :func:`simba.utils.read_write.get_video_meta_data`.
       To get video metadata as a table without printing, use :func:`simba.utils.lookups.get_table`.

    :param Union[str, os.PathLike] data_path: Path to video file or directory containing videos.
    :return: None. Video metadata is printed as formatted tables in the main console.
    """

    if os.path.isfile(data_path):
        video_meta_data = [get_video_meta_data(video_path=data_path)]
    elif os.path.isdir(data_path):
        video_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=False)
        video_meta_data = [get_video_meta_data(video_path=x) for x in video_paths]
    else:
        raise NoFilesFoundError(msg=f'{data_path} is not a valid file or directory path', source=print_video_meta_data.__name__)
    for video_meta in video_meta_data:
        table = get_table(data=video_meta, headers=('VIDEO PARAMETER', 'VALUE'), tablefmt='grid')
        print(f"{table} {Defaults.STR_SPLIT_DELIMITER.value}TABLE")


def get_ffmpeg_encoders(raise_error: bool = True) -> List[str]:
    """
    Get a list of all available FFmpeg encoders.

    :param bool raise_error: If True, raises an exception when FFmpeg is not available or the command fails. If False, returns an empty list on error. Default: True.
    :return: List of encoder names (e.g., ['libx264', 'aac', 'libvpx', ...]). Returns empty list if FFmpeg is unavailable and raise_error=False.
    :rtype: List[str]

    :example:
    >>> codecs = get_ffmpeg_encoders()
    >>> print(Formats.BATCH_CODEC.value in codecs)
    """

    check_ffmpeg_available(raise_error=True)
    try:
        proc = subprocess.Popen(['ffmpeg', '-encoders'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if isinstance(stdout, bytes):
            stdout = stdout.decode('utf-8')
    except Exception as e:
        if raise_error:
            raise FFMPEGNotFoundError(msg=str(e.args))
        else:
            return []
    encoders = []
    lines = stdout.split('\n')

    for line in lines:
        if re.match(r'^\s*[VAS]', line):
            parts = line.split()
            if len(parts) >= 2:
                encoder_name = parts[1]
                encoders.append(encoder_name)
    return encoders


def find_closest_string(target: str,
                        string_list: List[str],
                        case_sensitive: bool = False,
                        token_based: bool = True) -> Optional[Tuple[str, Union[int, float]]]:
    """
    Find the closest string in a list to a target string using hybrid similarity matching.

    This function uses a combination of token-based matching and Levenshtein distance to find
    the best match. Token-based matching is particularly useful for strings like body part names
    where word order may vary (e.g., "Left_ear" vs "Ear_left").

    :param str target: The target string to match against.
    :param List[str] string_list: List of strings to search through.
    :param bool case_sensitive: If True, comparison is case-sensitive. If False (default), comparison is case-insensitive.
    :param bool token_based: If True (default), uses hybrid token-based and Levenshtein matching which handles word reordering better. If False, uses pure Levenshtein distance only.
    :return: Tuple of (closest_string, distance) or None if string_list is empty. When token_based=True, distance is a float score (lower is better). When token_based=False, distance is integer edit distance.
    :rtype: Optional[Tuple[str, Union[int, float]]]

    :example:
    >>> find_closest_string("cat", ["dog", "car", "bat"])
    >>> ('car', 0.33)
    >>> find_closest_string("Left_ear", ["Ear_left", "Right_ear", "Nose"])
    >>> ('Ear_left', 0.0)
    >>> find_closest_string("CAT", ["dog", "car", "bat"], case_sensitive=False)
    >>> ('car', 0.33)
    >>> find_closest_string("CAT", ["dog", "car", "bat"], case_sensitive=True, token_based=False)
    >>> ('car', 3)
    """

    check_str(name=f'{find_closest_string.__name__} target', value=target, allow_blank=False, raise_error=True)
    check_instance(source=f'{find_closest_string.__name__} string_list', instance=string_list, accepted_types=(list,), raise_error=True)
    if len(string_list) == 0:
        return None
    for i in string_list:
        check_str(name=f'{find_closest_string.__name__} string_list entry', value=i, allow_blank=False, raise_error=True)

    def levenshtein(s1: str, s2: str) -> int:
        if s1 == s2: return 0
        if not s1: return len(s2)
        if not s2: return len(s1)
        if len(s1) > len(s2): s1, s2 = s2, s1
        prev_row = list(range(len(s1) + 1))
        for i, c2 in enumerate(s2):
            curr_row = [i + 1]
            for j, c1 in enumerate(s1):
                cost = 0 if c1 == c2 else 1
                curr_row.append(min(prev_row[j + 1] + 1, curr_row[j] + 1, prev_row[j] + cost))
            prev_row = curr_row
        return prev_row[-1]

    def tokenize(s: str) -> List[str]:
        """Split string by common delimiters and return sorted tokens"""
        tokens = re.split(r'[_\-\s]+', s)
        return sorted([t for t in tokens if t])

    def token_sort_similarity(s1: str, s2: str) -> float:
        """
        Hybrid similarity score combining token matching with character-level Levenshtein.
        Returns a score where 0.0 = perfect match, higher = worse match.
        """
        tokens1 = tokenize(s1)
        tokens2 = tokenize(s2)
        
        # Token set matching
        set1, set2 = set(tokens1), set(tokens2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            token_score = 1.0
        else:
            token_score = 1.0 - (intersection / union)  # Jaccard distance

        sorted_s1 = '_'.join(tokens1)
        sorted_s2 = '_'.join(tokens2)
        max_len = max(len(sorted_s1), len(sorted_s2))
        if max_len == 0:
            lev_score = 0.0
        else:
            lev_score = levenshtein(sorted_s1, sorted_s2) / max_len
        
        # Weighted combination: token matching (70%) + order similarity (30%)
        return token_score * 0.7 + lev_score * 0.3

    # Prepare strings for comparison
    if not case_sensitive:
        target_cmp = target.lower()
        string_list_cmp = [s.lower() for s in string_list]
    else:
        target_cmp = target
        string_list_cmp = string_list

    # Find closest match
    if token_based:
        scores = [token_sort_similarity(target_cmp, s) for s in string_list_cmp]
        closest_idx = min(range(len(scores)), key=lambda i: scores[i])
        closest = string_list[closest_idx]
        distance = scores[closest_idx]
    else:
        distances = [levenshtein(target_cmp, s) for s in string_list_cmp]
        closest_idx = min(range(len(distances)), key=lambda i: distances[i])
        closest = string_list[closest_idx]
        distance = distances[closest_idx]
    
    return closest, distance



def create_directionality_cords(bp_dict: dict,
                                left_ear_name: str,
                                nose_name: str,
                                right_ear_name: str) -> dict:
    """
    Helper to create a dictionary mapping animal body-parts (nose, left ear, right ear) to their X and Y coordinate
    column names for directionality analysis.

    :param dict bp_dict: Dictionary with animal names as keys and body-part coordinate information as values. Expected to contain 'X_bps' and 'Y_bps' keys with lists of column names.
    :param str left_ear_name: Name of the left ear body-part to search for in coordinate column names.
    :param str nose_name: Name of the nose body-part to search for in coordinate column names.
    :param str right_ear_name: Name of the right ear body-part to search for in coordinate column names.
    :return: Nested dictionary with animal names as keys, body-part types (nose, ear_left, ear_right) as second-level keys, and coordinate types (X_bps, Y_bps) as third-level keys with corresponding column names as values.
    :rtype: dict
    :raises InvalidInputError: If any required body-part or coordinate cannot be found in the input dictionary.

    :example:
    >>> bp_dict = {'Animal_1': {'X_bps': ['Animal_1_Nose_x', 'Animal_1_Ear_left_x', 'Animal_1_Ear_right_x'], 'Y_bps': ['Animal_1_Nose_y', 'Animal_1_Ear_left_y', 'Animal_1_Ear_right_y']}}
    >>> create_directionality_cords(bp_dict=bp_dict, left_ear_name='Ear_left', nose_name='Nose', right_ear_name='Ear_right')
    >>> {'Animal_1': {'nose': {'X_bps': 'Animal_1_Nose_x', 'Y_bps': 'Animal_1_Nose_y'}, 'ear_left': {'X_bps': 'Animal_1_Ear_left_x', 'Y_bps': 'Animal_1_Ear_left_y'}, 'ear_right': {'X_bps': 'Animal_1_Ear_right_x', 'Y_bps': 'Animal_1_Ear_right_y'}}}
    """

    NOSE, EAR_LEFT, EAR_RIGHT = Keys.NOSE.value, Keys.EAR_LEFT.value, Keys.EAR_RIGHT.value

    results = {}
    for animal in bp_dict.keys():
        results[animal] = {NOSE: {}, EAR_LEFT: {}, EAR_RIGHT: {}}
        for dimension in ["X_bps", "Y_bps"]:
            for cord in bp_dict[animal][dimension]:
                if (nose_name.lower() in cord.lower()) and ("x" in cord.lower()):
                    results[animal][NOSE]["X_bps"] = cord
                elif (nose_name.lower() in cord.lower()) and ("y" in cord.lower()):
                    results[animal][NOSE]["Y_bps"] = cord
                elif (left_ear_name.lower() in cord.lower()) and ("x" in cord.lower()):
                    results[animal][EAR_LEFT]["X_bps"] = cord
                elif (left_ear_name.lower() in cord.lower()) and ("y" in cord.lower()):
                    results[animal][EAR_LEFT]["Y_bps"] = cord
                elif (right_ear_name.lower() in cord.lower()) and ("x" in cord.lower()):
                    results[animal][EAR_RIGHT]["X_bps"] = cord
                elif (right_ear_name.lower() in cord.lower()) and ("y" in cord.lower()):
                    results[animal][EAR_RIGHT]["Y_bps"] = cord

    for animal_name, animal_bps in results.items():
        for bp_name, bp_values in animal_bps.items():
           if len(bp_values.keys()) == 0:
               raise InvalidInputError(msg=f'Could not detect a body-part for animal {animal_name}, body-part {bp_name} in SimBA project. Make sure the body-part configuration file at {Paths.BP_NAMES.value} lists the appropriate body-parts', source=create_directionality_cords.__name__)
           for cord_key, cord_value in bp_values.items():
                if cord_value == '':
                    raise InvalidInputError(msg=f'Could not detect a body-part for animal {animal_name}, body-part {bp_name} and coordinate {cord_key} in SimBA project. MAke sure the body-part configuration file at {Paths.BP_NAMES.value} lists the appropriate body-parts. Passed values: {left_ear_name, nose_name, right_ear_name}', source=create_directionality_cords.__name__)
    return results




#
# display_resolution = get_display_resolution()
# img_size = (3000, 600)
# _get_img_resize_info(img_size=img_size, display_resolution=display_resolution)
#

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

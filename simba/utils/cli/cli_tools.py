import glob
import os
from typing import Tuple, Union

import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.utils.checks import check_float, check_str
from simba.utils.enums import ConfigKey, Dtypes, Paths
from simba.utils.lookups import get_bp_config_code_class_pairs
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (get_fn_ext, read_config_entry,
                                    read_config_file,
                                    read_project_path_and_file_type,
                                    read_video_info_csv)


def feature_extraction_runner(config_path: Union[str, os.PathLike]) -> None:
    """
    Helper to run feature extraction from CLI.

    :param config_path: Path to SimBA project config file in ini format.
    """

    config = read_config_file(config_path=config_path)
    pose_setting = read_config_entry(
        config=config,
        section=ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
        option=ConfigKey.POSE_SETTING.value,
        data_type=Dtypes.STR.value,
    )
    animal_cnt = read_config_entry(
        config=config,
        section=ConfigKey.GENERAL_SETTINGS.value,
        option=ConfigKey.ANIMAL_CNT.value,
        data_type=Dtypes.INT.value,
    )
    print(
        f"Pose-estimation body part setting for feature extraction: {animal_cnt} animals {animal_cnt} body-parts"
    )
    feature_extractor_classes = get_bp_config_code_class_pairs()
    if pose_setting == "8":
        feature_extractor = feature_extractor_classes[pose_setting][animal_cnt](
            config_path=config_path
        )
    else:
        feature_extractor = feature_extractor_classes[pose_setting](
            config_path=config_path
        )
    feature_extractor.run()


def set_video_parameters(
    config_path: Union[str, os.PathLike],
    px_per_mm: float,
    fps: float,
    resolution: Tuple[int, int],
) -> None:
    """
    Helper to batch set the video_info.csv from CLI. Requires all videos to have the same pixels per millimeter,
    fps, and resolution.

    :param str config_path: Path to SimBA project config file in ini format.
    :param float px_per_mm: Pixels per millimeter in all the video files.
    :param int fps: FPS of the video files.
    :param tuple resolution: Resolution of the video files.
    """

    timer = SimbaTimer(start=True)
    config = read_config_file(config_path=config_path)
    project_path, file_type = read_project_path_and_file_type(config=config)
    video_info_path = os.path.join(project_path, Paths.VIDEO_INFO.value)
    if not os.path.isfile(video_info_path):
        df = pd.DataFrame(
            columns=[
                "Video",
                "fps",
                "Resolution_width",
                "Resolution_height",
                "Distance_in_mm",
                "pixels/mm",
            ]
        ).set_index("Video")
        df.to_csv(video_info_path)
    video_info = read_video_info_csv(os.path.join(project_path, Paths.VIDEO_INFO.value))
    data_paths = glob.glob(
        os.path.join(project_path, Paths.OUTLIER_CORRECTED.value) + "/*." + file_type
    )
    for file_path in data_paths:
        _, video_name, _ = get_fn_ext(file_path)
        if video_name not in list(video_info["Video"]):
            video_info.loc[len(video_info)] = [
                video_name,
                fps,
                resolution[0],
                resolution[1],
                0,
                px_per_mm,
            ]
    video_info.reset_index(drop=True).set_index("Video").to_csv(
        os.path.join(project_path, Paths.VIDEO_INFO.value)
    )
    timer.stop_timer()
    stdout_success(
        msg="Video parameters set",
        elapsed_time=timer.elapsed_time_str,
        source=set_video_parameters.__name__,
    )


def set_outlier_correction_criteria_cli(
    config_path: Union[str, os.PathLike],
    movement_criterion: float,
    location_criterion: float,
    aggregation: Literal["mean", "median"],
    body_parts: dict,
):
    """Helper to set outlier settings in a SimBA project_config.ini from command line"""
    timer = SimbaTimer(start=True)
    config = read_config_file(config_path=config_path)
    check_float(name="MOVEMENT CRITERION", value=movement_criterion, min_value=0.1)
    check_float(name="LOCATION CRITERION", value=location_criterion, min_value=0.1)
    check_str(name="AGGREGATION METHOD", value=aggregation, options=("mean", "median"))
    config.set(
        section="Outlier settings",
        option="movement_criterion",
        value=str(movement_criterion),
    )
    config.set(
        section="Outlier settings",
        option="location_criterion",
        value=str(location_criterion),
    )
    config.set(section="Outlier settings", option="mean_or_median", value=aggregation)
    for cnt, (k, v) in enumerate(body_parts.items()):
        config.set(
            section="Outlier settings",
            option=f"movement_bodypart1_mouse{cnt+1}",
            value=v["Movement"][0],
        )
        config.set(
            section="Outlier settings",
            option=f"movement_bodypart2_mouse{cnt+1}",
            value=v["Movement"][1],
        )
        config.set(
            section="Outlier settings",
            option=f"location_bodypart1_mouse{cnt+1}",
            value=v["Location"][0],
        )
        config.set(
            section="Outlier settings",
            option=f"location_bodypart2_mouse{cnt+1}",
            value=v["Location"][1],
        )
    with open(config_path, "w") as file:
        config.write(file)

    timer.stop_timer()
    stdout_success(
        "Outlier parameters set",
        elapsed_time=timer.elapsed_time_str,
        source=set_outlier_correction_criteria_cli.__name__,
    )


# DEFINITIONS
# from simba.outlier_tools.outlier_corrector_movement import OutlierCorrecterMovement
# from simba.outlier_tools.outlier_corrector_location import OutlierCorrecterLocation
#
# CONFIG_PATH = '/Users/simon/Desktop/envs/troubleshooting/DLC_2_Black_animals/project_folder/project_config.ini'
# AGGREGATION_METHOD = 'mean'
# BODY_PARTS = {'Animal_1': {'Movement': ['Nose_1', 'Tail_base_1'],
#                            'Location': ['Nose_1', 'Tail_base_1']},
#               'Animal_2': {'Movement': ['Nose_2', 'Tail_base_2'],
#                            'Location': ['Nose_2', 'Tail_base_2']}}
# MOVEMENT_CRITERION = 0.7
# LOCATION_CRITERION = 2.0
#
# set_outlier_correction_criteria_cli(config_path=CONFIG_PATH,
#                                     aggregation=AGGREGATION_METHOD,
#                                     body_parts=BODY_PARTS,
#                                     movement_criterion=MOVEMENT_CRITERION,
#                                     location_criterion=LOCATION_CRITERION)
#
#
# _ = OutlierCorrecterMovement(config_path=CONFIG_PATH).run()
# _ = OutlierCorrecterLocation(config_path=CONFIG_PATH).run()


# set_video_parameters(config_path='/Users/simon/Desktop/envs/troubleshooting/notebook_example/project_folder/project_config.ini', px_per_mm=5.6, fps=25, resolution=(400, 400))

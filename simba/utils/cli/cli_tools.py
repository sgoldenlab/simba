import glob
import os
from typing import Tuple, Union

import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict, check_int,
                                check_str, check_valid_boolean,
                                check_valid_dict)
from simba.utils.enums import ConfigKey, Dtypes, Paths
from simba.utils.lookups import get_bp_config_code_class_pairs
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_core_cnt, get_fn_ext,
                                    get_video_meta_data, read_config_entry,
                                    read_config_file, read_json,
                                    read_project_path_and_file_type,
                                    read_video_info_csv)
from simba.video_processors.blob_tracking_executor import BlobTrackingExecutor


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
    video_info = read_video_info_csv(file_path=os.path.join(project_path, Paths.VIDEO_INFO.value), raise_error=False)
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


def blob_tracker(config_path: Union[str, os.PathLike]) -> None:
    """
    Method to access blob detection through CLI or notebook

    .. note::
       For an example blob detection config file, see `https://github.com/sgoldenlab/simba/blob/master/misc/blob_definitions_ex.json <https://github.com/sgoldenlab/simba/blob/master/misc/blob_definitions_ex.json>`__.

    :param Union[str, os.PathLike] config_path: Path to json file holding blob detection setting
    :returns: None. The blob detection data is saved at the location specified in the ``config_path``.
    :rtype: None

    :example:
    >>> blob_tracker('/Users/simon/Downloads/result_bg/blob_definitions.json')
    """

    REQUIRED_KEYS = ['video_data', 'input_dir', 'output_dir', 'gpu', 'save_bg_videos', 'core_cnt', 'core_cnt',
                     'vertice_cnt', 'open_iterations', 'close_iterations', 'video_data']
    VIDEO_KEYS = ['video_path', 'threshold', 'smoothing_time', 'buffer_size', "reference", 'close_kernel',
                  'open_kernel']
    check_file_exist_and_readable(file_path=config_path)
    data = read_json(x=config_path)
    check_if_keys_exist_in_dict(data=data, key=REQUIRED_KEYS, name=config_path, raise_error=True)
    check_if_dir_exists(in_dir=data['input_dir'], source=f'{config_path} input_dir')
    check_if_dir_exists(in_dir=data['output_dir'], source=f'{config_path} output_dir')
    check_valid_boolean(value=data['gpu'], source=f'{config_path} gpu')
    check_valid_boolean(value=data['save_bg_videos'], source=f'{config_path} save_bg_videos')
    check_int(name=f'{config_path} core_cnt', value=data['core_cnt'], min_value=1, max_value=find_core_cnt()[0])
    check_int(name=f'{config_path} vertice_cnt', value=data['vertice_cnt'], min_value=4)
    check_int(name=f'{config_path} close_iterations', value=data['close_iterations'], min_value=1)
    check_int(name=f'{config_path} open_iterations', value=data['open_iterations'], min_value=1)

    check_valid_dict(x=data['video_data'], valid_key_dtypes=(str,))
    for video_name, video_data in data['video_data'].items():
        check_if_keys_exist_in_dict(data=video_data, key=VIDEO_KEYS, name=f'{config_path} {video_name}', raise_error=True)
        check_file_exist_and_readable(file_path=video_data['video_path'])
        check_file_exist_and_readable(file_path=video_data['reference'])
        video_meta_data = get_video_meta_data(video_path=video_data['video_path'])
        max_dim = max(video_meta_data['width'], video_meta_data['height'])
        check_int(name=f'{video_name} threshold', value=video_data['threshold'], min_value=1, max_value=100)
        data['video_data'][video_name]['threshold'] = int((video_data['threshold'] / 100) * 255)
        if video_data['smoothing_time'] is not None:
            check_float(name=f'{video_name} smoothing_time', value=video_data['smoothing_time'], min_value=0.0)
            data['video_data'][video_name]['smoothing_time'] = int(float(video_data['smoothing_time']) * 1000)
        if video_data['buffer_size'] is not None:
            check_float(name=f'{video_name} buffer_size', value=video_data['buffer_size'], min_value=0.0)
        if video_data['close_kernel'] is not None:
            check_float(name=f'{video_name} close_kernel', value=video_data['close_kernel'], min_value=0.0)
            w = ((max_dim * float(video_data['close_kernel'])) / 100) / 4
            h = ((max_dim * float(video_data['close_kernel'])) / 100) / 4
            k = (int(max(h, 1)), int(max(w, 1)))
            data['video_data'][video_name]['close_kernel'] = tuple(k)
        if video_data['open_kernel'] is not None:
            check_float(name=f'{video_name} open_kernel', value=video_data['open_kernel'], min_value=0.0)
            w = ((max_dim * float(video_data['open_kernel'])) / 100) / 4
            h = ((max_dim * float(video_data['open_kernel'])) / 100) / 4
            k = (int(max(h, 1)), int(max(w, 1)))
            data['video_data'][video_name]['open_kernel'] = tuple(k)

    tracker = BlobTrackingExecutor(data=data)
    tracker.run()

#blob_tracker(r'C:\troubleshooting\blob_track\blob_definitions_ex.json')



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


#set_video_parameters(config_path=r'C:\troubleshooting\ethan_alyssa\project_folder\project_config.ini', px_per_mm=5.6, fps=25, resolution=(400, 400))

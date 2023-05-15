import os, glob
import pandas as pd
from typing import Union

from simba.utils.enums import ConfigKey, Dtypes, Paths
from simba.utils.read_write import read_config_file, read_config_entry, read_project_path_and_file_type, read_video_info_csv, get_fn_ext
from simba.utils.lookups import get_bp_config_code_class_pairs
from simba.utils.printing import SimbaTimer, stdout_success

def feature_extraction_runner(config_path: Union[str, os.PathLike]) -> None:
    """
    Helper to run feature extraction from CLI.

    :param config_path: Path to SimBA project config file in ini format.
    """

    config = read_config_file(config_path=config_path)
    pose_setting = read_config_entry(config=config, section=ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, option=ConfigKey.POSE_SETTING.value, data_type=Dtypes.STR.value)
    animal_cnt = read_config_entry(config=config, section=ConfigKey.GENERAL_SETTINGS.value, option=ConfigKey.ANIMAL_CNT.value, data_type=Dtypes.INT.value)
    print(f'Pose-estimation body part setting for feature extraction: {animal_cnt} animals {animal_cnt} body-parts')
    feature_extractor_classes = get_bp_config_code_class_pairs()
    if pose_setting == '8':
        feature_extractor = feature_extractor_classes[pose_setting][animal_cnt](config_path=config_path)
    else:
        feature_extractor = feature_extractor_classes[pose_setting](config_path=config_path)
    feature_extractor.run()


def set_video_parameters(config_path: str,
                         px_per_mm: float,
                         fps: float,
                         resolution: tuple) -> None:
    """
    Helper to batch set the video_info.csv from CLI

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
        df = pd.DataFrame(columns=['Video', 'fps', 'Resolution_width', 'Resolution_height', 'Distance_in_mm', 'pixels/mm']).set_index('Video')
        df.to_csv(video_info_path)
    video_info = read_video_info_csv(os.path.join(project_path, Paths.VIDEO_INFO.value))
    data_paths = glob.glob(os.path.join(project_path, Paths.OUTLIER_CORRECTED.value) + '/*.' + file_type)
    for file_path in data_paths:
        _, video_name, _ = get_fn_ext(file_path)
        if video_name not in list(video_info['Video']):
            video_info.loc[len(video_info)] = [video_name, fps, resolution[0], resolution[1], 0, px_per_mm]
    video_info.reset_index(drop=True).set_index('Video').to_csv(os.path.join(project_path, Paths.VIDEO_INFO.value))
    timer.stop_timer()
    stdout_success(msg='Video parameters set', elapsed_time=timer.elapsed_time_str)

#set_video_parameters(config_path='/Users/simon/Desktop/envs/troubleshooting/notebook_example/project_folder/project_config.ini', px_per_mm=5.6, fps=25, resolution=(400, 400))





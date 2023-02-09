import os, glob
from simba.misc_tools import get_fn_ext
from simba.read_config_unit_tests import read_config_file, read_config_entry
from simba.extract_frames_fast import video_to_frames

def split_file_path(path: str):
    """
    Split path into a list of the path directory components.
    """
    path_parts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            path_parts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            path_parts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            path_parts.insert(0, parts[1])
    return path_parts

def extract_frames_for_all_videos_in_folder(in_directory: str,
                                            config_path: str):
    """
    Extracts all frames for all videos in avi and mp4 format in a directory.
    """

    video_paths = []
    file_paths_in_folder = [f for f in glob.glob(in_directory + '/*') if os.path.isfile(f)]
    for file_cnt, file_path in enumerate(file_paths_in_folder):
        _, file_name, file_ext = get_fn_ext(file_path)
        if (file_ext.lower() == '.mp4') or (file_ext.lower() == '.avi'):
            video_paths.append(file_name + file_ext)
    if len(video_paths) == 0:
        print('SIMBA ERROR: No video files in mp4 or avi format found in {} directory'.format(in_directory))
        raise ValueError
    config = read_config_file(config_path)
    project_path = read_config_entry(config, 'General settings', 'project_path', data_type='folder_path')
    for video_cnt, video_path in video_paths:
        _, file_name, _ = get_fn_ext(video_path)
        save_dir = os.path.join(project_path, 'frames', 'input', file_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            video_to_frames(video_path, save_dir, overwrite=True, every=1, chunk_size=1000)
            print('Video {} complete {}/{}...'.format(file_name, str(video_cnt+1), str(len(video_paths))))
        else:
            print('Frame directory for video {} already exist. SimBA is skipping video {}...'.format(file_name, file_name))
            pass
    print('Frame extraction of {} videos complete.'.format(str(len(video_paths))))



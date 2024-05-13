import pytest
import os, shutil, glob
from simba.utils.read_write import get_fn_ext
from simba.utils.checks import check_file_exist_and_readable
from simba.video_processors.batch_process_create_ffmpeg_commands import FFMPEGCommandCreator
from simba.video_processors.video_processing import (change_img_format,
                                                     extract_frame_range,
                                                     change_single_video_fps,
                                                     change_fps_of_multiple_videos,
                                                     convert_video_powerpoint_compatible_format,
                                                     convert_to_mp4,
                                                     video_to_greyscale,
                                                     superimpose_frame_count,
                                                     clip_video_in_range,
                                                     downsample_video,
                                                     gif_creator,
                                                     batch_convert_video_format,
                                                     batch_create_frames,
                                                     extract_frames_single_video,
                                                     multi_split_video,
                                                     video_concatenator)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

@pytest.fixture(params=['tests/data/test_projects/two_c57/project_folder/videos/Together_1.avi'])
def config_video_path_args(request):
    return request


@pytest.fixture(params=['tests/data/test_projects/two_c57/project_folder/videos'])
def config_video_dir_args(request):
    return request

def copy_file(file_path):
    test_file_copy_path = os.path.join(os.path.dirname(file_path), 'test.mp4')
    if os.path.isfile(test_file_copy_path): os.remove(test_file_copy_path)
    shutil.copy(file_path, test_file_copy_path)
    return test_file_copy_path

@pytest.mark.parametrize('json_path', ['tests/data/test_projects/two_c57/video_processing/batch_process_log.json'])
def test_ffmpeg_command_creator(json_path):
    ffmpeg_executor = FFMPEGCommandCreator(json_path=json_path)
    ffmpeg_executor.crop_videos()
    ffmpeg_executor.clip_videos()
    ffmpeg_executor.downsample_videos()
    ffmpeg_executor.apply_fps()
    ffmpeg_executor.apply_grayscale()
    ffmpeg_executor.apply_frame_count()
    ffmpeg_executor.apply_clahe()
    ffmpeg_executor.move_all_processed_files_to_output_folder()
    
 
@pytest.mark.parametrize('directory, file_type_in, file_type_out', [['tests/data/test_projects/two_c57/video_processing/test_imgs', 'png', 'bmp']])
def test_change_img_format(directory, file_type_in, file_type_out):
    test_dir_copy_path = os.path.join(os.path.dirname(directory), 'test_out')
    if os.path.isdir(test_dir_copy_path): shutil.rmtree(test_dir_copy_path)
    shutil.copytree(directory, test_dir_copy_path)
    change_img_format(directory=test_dir_copy_path, file_type_in=file_type_in, file_type_out=file_type_out)


@pytest.mark.parametrize('start_frame, end_frame', [[1, 5]])
def test_extract_frame_range(config_video_path_args, start_frame, end_frame):
    extract_frame_range(file_path=config_video_path_args.param, start_frame=start_frame, end_frame=end_frame)
    directory, video_name, _ = get_fn_ext(config_video_path_args.param)
    saved_files = glob.glob(os.path.join(directory, f'{video_name}_frames') + '/*.png')
    assert len(saved_files) == (end_frame - start_frame) + 1
    for i in saved_files: check_file_exist_and_readable(i)

@pytest.mark.parametrize('time',  [1])
def test_remove_beginning_of_video(config_video_path_args, time):
    copied_file_path = copy_file(file_path=config_video_path_args.param)
    #remove_beginning_of_video(file_path=copied_file_path, time=time)

@pytest.mark.parametrize('fps', [10])
def test_change_single_video_fps(config_video_path_args, fps):
    copied_file_path = copy_file(file_path=config_video_path_args.param)
    change_single_video_fps(file_path=copied_file_path, fps=fps)

@pytest.mark.parametrize('directory, fps', [['tests/data/test_projects/two_c57/project_folder/videos/', 10]])
def test_change_fps_of_multiple_videos(directory, fps):
    test_dir_copy_path = os.path.join(os.path.dirname(directory), 'test_out')
    if os.path.isdir(test_dir_copy_path): shutil.rmtree(test_dir_copy_path)
    shutil.copytree(directory, test_dir_copy_path)
    change_fps_of_multiple_videos(directory=test_dir_copy_path, fps=fps)

# def test_convert_video_powerpoint_compatible_format(config_video_path_args):
#     copied_file_path = copy_file(file_path=config_video_path_args.param)
#     convert_video_powerpoint_compatible_format(file_path=copied_file_path)

# def test_convert_to_mp4(config_video_path_args):
#     copied_file_path = copy_file(file_path=config_video_path_args.param)
#     convert_to_mp4(file_path=copied_file_path)

def test_video_to_greyscale(config_video_path_args):
    copied_file_path = copy_file(file_path=config_video_path_args.param)
    video_to_greyscale(file_path=copied_file_path)

def test_superimpose_frame_count(config_video_path_args):
    copied_file_path = copy_file(file_path=config_video_path_args.param)
    superimpose_frame_count(file_path=copied_file_path)

@pytest.mark.parametrize('start_time, end_time', [['00:00:01', '00:00:05']])
def test_clip_video_in_range(config_video_path_args, start_time, end_time):
    copied_file_path = copy_file(file_path=config_video_path_args.param)
    clip_video_in_range(file_path=copied_file_path, start_time=start_time, end_time=end_time)

@pytest.mark.parametrize('width, height', [[600, 300]])
def test_downsample_video(config_video_path_args, width, height):
    copied_file_path = copy_file(file_path=config_video_path_args.param)
    downsample_video(file_path=copied_file_path, video_height=height, video_width=width)

@pytest.mark.parametrize('start_time, duration, width', [[1, 3, 600]])
def test_gif_creator(config_video_path_args, start_time, duration, width):
    copied_file_path = copy_file(file_path=config_video_path_args.param)
    gif_creator(file_path=copied_file_path, width=width, start_time=start_time, duration=duration)

@pytest.mark.parametrize('start_time, duration, width', [[1, 3, 600]])
def test_gif_creator(config_video_path_args, start_time, duration, width):
    copied_file_path = copy_file(file_path=config_video_path_args.param)
    gif_creator(file_path=copied_file_path, width=width, start_time=start_time, duration=duration)

@pytest.mark.parametrize('input_format, output_format', [['avi', 'mp4']])
def test_batch_convert_video_format(config_video_dir_args, input_format, output_format):
    test_dir_copy_path = os.path.join(os.path.dirname(config_video_dir_args.param), 'test_out')
    if os.path.isdir(test_dir_copy_path): shutil.rmtree(test_dir_copy_path)
    shutil.copytree(config_video_dir_args.param, test_dir_copy_path)
    batch_convert_video_format(directory=test_dir_copy_path, input_format=input_format, output_format=output_format)

@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="LONG RUNNING TIME.")
def test_batch_create_frames(config_video_dir_args):
    test_dir_copy_path = os.path.join(os.path.dirname(config_video_dir_args.param), 'test_out')
    if os.path.isdir(test_dir_copy_path): shutil.rmtree(test_dir_copy_path)
    shutil.copytree(config_video_dir_args.param, test_dir_copy_path)
    batch_create_frames(directory=test_dir_copy_path)

@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="LONG RUNNING TIME.")
def test_extract_frames_single_video(config_video_path_args):
    copied_file_path = copy_file(file_path=config_video_path_args.param)
    extract_frames_single_video(file_path=copied_file_path)

@pytest.mark.parametrize('start_times, end_times', [(['00:00:01', '00:00:03'], ['00:00:04', '00:00:05'])])
def test_multi_split_video(config_video_path_args, start_times, end_times):
    copied_file_path = copy_file(file_path=config_video_path_args.param)
    multi_split_video(file_path=copied_file_path, start_times=start_times, end_times=end_times)

@pytest.mark.parametrize('horizontal', [True, False])
def test_video_concatenator(config_video_path_args, horizontal):
    copied_file_path = copy_file(file_path=config_video_path_args.param)
    _, video_name, _ = get_fn_ext(filepath=copied_file_path)
    video_concatenator(video_one_path=copied_file_path, video_two_path=copied_file_path, resolution=video_name, horizontal=horizontal)
    



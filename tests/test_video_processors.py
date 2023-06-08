import pytest
from simba.video_processors.batch_process_create_ffmpeg_commands import FFMPEGCommandCreator

@pytest.mark.parametrize('json_path', ['/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/two_c57/video_processing/batch_process_log.json'])
def test_ffmpeg_command_creator(json_path):
    print(json_path)
    ffmpeg_executor = FFMPEGCommandCreator(json_path=json_path)
    ffmpeg_executor.crop_videos()
    ffmpeg_executor.clip_videos()
    ffmpeg_executor.downsample_videos()
    ffmpeg_executor.apply_fps()
    ffmpeg_executor.apply_grayscale()
    ffmpeg_executor.apply_frame_count()
    ffmpeg_executor.apply_clahe()
    ffmpeg_executor.move_all_processed_files_to_output_folder()


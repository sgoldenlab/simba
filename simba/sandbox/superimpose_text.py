from typing import Union, Optional
try:
    from typing import Literal
except:
    from typing_extensions import Literal
import os
import subprocess
from simba.utils.read_write import get_fn_ext, find_all_videos_in_directory, get_video_meta_data
from simba.video_processors.roi_selector import ROISelector
from simba.utils.checks import check_ffmpeg_available, check_float, check_if_dir_exists, check_file_exist_and_readable, check_str, check_int
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.errors import InvalidInputError


def superimpose_video_names(video_path: Union[str, os.PathLike],
                            font_size: Optional[int] = 30,
                            font_color: Optional[str] = 'white',
                            font_border_color: Optional[str] = 'black',
                            font_border_width: Optional[int] = 2,
                            position: Optional[Literal['top_left', 'top_right', 'bottom_left', 'bottom_right', 'middle_top', 'middle_bottom']] = 'top_left',
                            save_dir: Optional[Union[str, os.PathLike]] = None) -> None:

    check_ffmpeg_available(raise_error=True)
    timer = SimbaTimer(start=True)
    POSITIONS = ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'top_middle', 'bottom_middle']
    check_str(name=f'{superimpose_video_names.__name__} position', value=position, options=POSITIONS)
    check_int(name=f'{superimpose_video_names.__name__} font_size', value=font_size, min_value=1)
    check_int(name=f'{superimpose_video_names.__name__} font_border_width', value=font_border_width, min_value=1)
    font_color = ''.join(filter(str.isalnum, font_color)).lower()
    font_border_color = ''.join(filter(str.isalnum, font_border_color)).lower()
    if os.path.isfile(video_path):
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_paths = list(find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True).values())
    else:
        raise InvalidInputError(msg=f'{video_path} is not a valid file path or a valid directory path', source=superimpose_video_names.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir = os.path.dirname(video_paths[0])
    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, ext = get_fn_ext(video_path)
        print(f'Superimposing video name on {video_name} (Video {file_cnt + 1}/{len(video_paths)})...')
        save_path = os.path.join(save_dir, f'{video_name}_video_name_superimposed{ext}')
        if position == POSITIONS[0]:
            cmd = f"ffmpeg -i '{video_path}' -vf \"drawtext=fontfile=Arial.ttf:text={video_name}:x=5:y=5:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy '{save_path}' -loglevel error -stats -hide_banner -y"
        elif position == POSITIONS[1]:
            cmd = f"ffmpeg -i '{video_path}' -vf \"drawtext=fontfile=Arial.ttf:text={video_name}:x=(w-tw-5):y=5:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy '{save_path}' -loglevel error -stats -hide_banner -y"
        elif position == POSITIONS[2]:
            cmd = f"ffmpeg -i '{video_path}' -vf \"drawtext=fontfile=Arial.ttf:text={video_name}:x=5:y=(h-th-5):fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy '{save_path}' -loglevel error -stats -hide_banner -y"
        elif position == POSITIONS[3]:
            cmd = f"ffmpeg -i '{video_path}' -vf \"drawtext=fontfile=Arial.ttf:text={video_name}:x=(w-tw-5):y=(h-th-5):fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy '{save_path}' -loglevel error -stats -hide_banner -y"
        elif position == POSITIONS[4]:
            cmd = f"ffmpeg -i '{video_path}' -vf \"drawtext=fontfile=Arial.ttf:text={video_name}:x=(w-tw)/2:y=10:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy '{save_path}' -loglevel error -stats -hide_banner -y"
        else:
            cmd = f"ffmpeg -i '{video_path}' -vf \"drawtext=fontfile=Arial.ttf:text={video_name}:x=(w-tw)/2:y=(h-th-10):fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy '{save_path}' -loglevel error -stats -hide_banner -y"
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f'Super-imposed video name on {len(video_paths)} video(s), saved in {save_dir}', elapsed_time=timer.elapsed_time_str)



def superimpose_freetext(video_path: Union[str, os.PathLike],
                         text: str,
                         font_size: Optional[int] = 30,
                         font_color: Optional[str] = 'white',
                         font_border_color: Optional[str] = 'black',
                         font_border_width: Optional[int] = 2,
                         position: Optional[Literal['top_left', 'top_right', 'bottom_left', 'bottom_right', 'middle_top', 'middle_bottom']] = 'top_left',
                         save_dir: Optional[Union[str, os.PathLike]] = None) -> None:

    check_ffmpeg_available(raise_error=True)
    timer = SimbaTimer(start=True)
    POSITIONS = ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'top_middle', 'bottom_middle']
    check_str(name=f'{superimpose_freetext.__name__} position', value=position, options=POSITIONS)
    check_int(name=f'{superimpose_freetext.__name__} font_size', value=font_size, min_value=1)
    check_int(name=f'{superimpose_freetext.__name__} font_border_width', value=font_border_width, min_value=1)
    check_str(name=f'{superimpose_freetext.__name__} text', value=text)
    font_color = ''.join(filter(str.isalnum, font_color)).lower()
    font_border_color = ''.join(filter(str.isalnum, font_border_color)).lower()
    if os.path.isfile(video_path):
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_paths = list(find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True).values())
    else:
        raise InvalidInputError(msg=f'{video_path} is not a valid file path or a valid directory path', source=superimpose_video_names.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir = os.path.dirname(video_paths[0])
    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, ext = get_fn_ext(video_path)
        print(f'Superimposing video name on {video_name} (Video {file_cnt + 1}/{len(video_paths)})...')
        save_path = os.path.join(save_dir, f'{video_name}_text_superimposed{ext}')
        if position == POSITIONS[0]:
            cmd = f"ffmpeg -i '{video_path}' -vf \"drawtext=fontfile=Arial.ttf:text={text}:x=5:y=5:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy '{save_path}' -loglevel error -stats -hide_banner -y"
        elif position == POSITIONS[1]:
            cmd = f"ffmpeg -i '{video_path}' -vf \"drawtext=fontfile=Arial.ttf:text={text}:x=(w-tw-5):y=5:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy '{save_path}' -loglevel error -stats -hide_banner -y"
        elif position == POSITIONS[2]:
            cmd = f"ffmpeg -i '{video_path}' -vf \"drawtext=fontfile=Arial.ttf:text={text}:x=5:y=(h-th-5):fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy '{save_path}' -loglevel error -stats -hide_banner -y"
        elif position == POSITIONS[3]:
            cmd = f"ffmpeg -i '{video_path}' -vf \"drawtext=fontfile=Arial.ttf:text={text}:x=(w-tw-5):y=(h-th-5):fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy '{save_path}' -loglevel error -stats -hide_banner -y"
        elif position == POSITIONS[4]:
            cmd = f"ffmpeg -i '{video_path}' -vf \"drawtext=fontfile=Arial.ttf:text={text}:x=(w-tw)/2:y=10:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy '{save_path}' -loglevel error -stats -hide_banner -y"
        else:
            cmd = f"ffmpeg -i '{video_path}' -vf \"drawtext=fontfile=Arial.ttf:text={text}:x=(w-tw)/2:y=(h-th-10):fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy '{save_path}' -loglevel error -stats -hide_banner -y"
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f'Super-imposed text on {len(video_paths)} video(s), saved in {save_dir}', elapsed_time=timer.elapsed_time_str)

superimpose_freetext(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/open_field_below/project_folder/videos/raw_clip1_fps_5_progress_bar.mp4', position='top_left', text='DUCK')
import os
from datetime import datetime
from typing import Union, Optional
import subprocess
try:
    from typing import Literal
except:
    from typing_extensions import Literal
from simba.utils.checks import check_instance, check_ffmpeg_available, check_str, check_if_dir_exists, check_int, check_nvidea_gpu_available
from simba.utils.read_write import find_files_of_filetypes_in_directory, get_fn_ext, find_files_of_filetypes_in_directory
from simba.utils.enums import Options
from simba.utils.errors import InvalidInputError, FFMPEGCodecGPUError, NotDirectoryError
from simba.utils.lookups import percent_to_crf_lookup, percent_to_qv_lk
from simba.utils.printing import stdout_success, SimbaTimer
import cv2

def frames_to_movie(directory: Union[str, os.PathLike],
                    fps: int,
                    quality: int,
                    out_format: Literal['mp4', 'avi', 'webm', 'mov'] = 'mp4',
                    gpu: Optional[bool] = False) -> None:
    """
    Merge all image files in a folder to a mp4 video file. Video file is stored in the same directory as the
    input directory sub-folder.

    .. note::
       The Image files have to have ordered numerical names e.g., ``1.png``, ``2.png`` etc...

    :parameter str directory: Directory containing the images.
    :parameter int fps: The frame rate of the output video.
    :parameter int bitrate: The bitrate of the output video (e.g., 32000).
    :parameter str img_format: The format of the input image files (e.g., ``png``).
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = frames_to_movie(directory_path='project_folder/video_img', fps=15, bitrate=32000, img_format='png')
    """

    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None", source=frames_to_movie.__name__)
    check_if_dir_exists(in_dir=directory, source=frames_to_movie.__name__)
    check_int(name="FPS", value=fps, min_value=1)
    check_int(name="quality", value=quality, min_value=1)
    crf_lk = percent_to_crf_lookup()
    crf = crf_lk[str(quality)]
    img_paths = find_files_of_filetypes_in_directory(directory=directory, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, raise_error=True)
    _, _, ext = get_fn_ext(filepath=img_paths[0])
    img = cv2.imread(img_paths[0])
    img_h, img_w = int(img.shape[0]), int(img.shape[1])
    ffmpeg_fn = os.path.join(directory, f"%d.{ext}")
    save_path = os.path.join(os.path.dirname(directory), f"{os.path.basename(directory)}.{out_format}")
    if not gpu:
        cmd = f'ffmpeg -y -r {fps} -f image2 -s {img_h}x{img_w} -i "{ffmpeg_fn}" -c:v libx265 -crf {crf} "{save_path}" -loglevel error -stats -hide_banner -y'
    else:
        cmd = f'ffmpeg -y -r {fps} -f image2 -s {img_h}x{img_w} -i "{ffmpeg_fn}" -c:v h264_nvenc -crf {crf} "{save_path}" -loglevel error -stats -hide_banner -y'
    subprocess.call(cmd, shell=True)
    stdout_success(msg=f"Video created at {save_path}", source=frames_to_movie.__name__)





# if gpu:
    # else:
    #     command = f'ffmpeg -y -r {fps} -f image2 -s {img_h}x{img_w} -i "{ffmpeg_fn}" -vcodec libx264 -b {bitrate}k "{save_path}" -y'
    # print(
    #     f"Creating {os.path.basename(save_path)} from {len(img_paths_in_folder)} images..."
    # )
    # subprocess.call(command, shell=True)
    # stdout_success(msg=f"Video created at {save_path}", source=frames_to_movie.__name__)


# _ = frames_to_movie(directory='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT_downsampled', fps=15, bitrate=32000, img_format='png')











def convert_to_mp4(path: Union[str, os.PathLike],
                   codec: Literal['libx265', 'libx264', 'powerpoint'] = 'libx265',
                   save_dir: Optional[Union[str, os.PathLike]] = None,
                   quality: Optional[int] = 60) -> None:
    """
    Convert a directory containing videos, or a single video, to MP4 format using passed quality and codec.

    :param Union[str, os.PathLike] path: Path to directory or file.
    :param Literal['libx265', 'libx264', 'powerpoint'] codec:
    :param Optional[Optional[Union[str, os.PathLike]]] save_dir: Directory where to save the converted videos. If None, then creates a directory in the same directory as the input.
    :param Optional[int] quality: Integer representing the quality: 10, 20, 30.. 100.
    :return: None.

    :example:
    >>> convert_to_mp4(path='/Users/simon/Desktop/video_test', quality="100")
    """

    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)
    check_str(name=f'{convert_to_mp4.__name__} codec', value=codec, options=('libx265', 'libx264'))
    check_instance(source=f'{convert_to_mp4.__name__} path', instance=path, accepted_types=(str,))
    check_int(name=f'{convert_to_mp4.__name__} quality', value=quality)
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    crf_lk = percent_to_crf_lookup()
    crf = crf_lk[str(quality)]
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=convert_to_mp4.__name__)
    if os.path.isfile(path):
        file_paths = [path]
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(path), f'mp4_{datetime_}')
            os.makedirs(save_dir)
    elif os.path.isdir(path):
        file_paths = find_files_of_filetypes_in_directory(directory=path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        if save_dir is None:
            save_dir = os.path.join(path, f'mp4_{datetime_}')
            os.makedirs(save_dir)
    else:
        raise InvalidInputError(msg=f'Paths is not a valid file or directory path.', source=convert_to_mp4.__name__)
    for file_cnt, file_path in enumerate(file_paths):
        _, video_name, _ = get_fn_ext(filepath=file_path)
        print(f'Converting video {video_name} to MP4 (Video {file_cnt+1}/{len(file_paths)})...')
        _ = get_video_meta_data(video_path=file_path)
        out_path = os.path.join(save_dir, f'{video_name}.mp4')
        if codec != 'powerpoint':
            cmd = f'ffmpeg -i "{file_path}" -c:v {codec} -crf {crf} -c:a copy -an "{out_path}" -loglevel error -stats -hide_banner -y'
        else:
            cmd = f'ffmpeg -i "{file_path}" -c:v libx264 -preset slow -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf {crf} -c:v libx264 -codec:a aac "{out_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"{len(file_paths)} video(s) converted to MP4 and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=convert_to_mp4.__name__,)


def convert_to_avi(path: Union[str, os.PathLike],
                   codec: Literal['xvid', 'divx', 'mjpeg'] = 'divx',
                   save_dir: Optional[Union[str, os.PathLike]] = None,
                   quality: Optional[int] = 60) -> None:

    """
    Convert a directory containing videos, or a single video, to AVI format using passed quality and codec.

    :param Union[str, os.PathLike] path: Path to directory or file.
    :param Literal['xvid', 'divx', 'mjpeg'] codec: Method to encode the AVI format. Default: xvid.
    :param Optional[Optional[Union[str, os.PathLike]]] save_dir: Directory where to save the converted videos. If None, then creates a directory in the same directory as the input.
    :param Optional[int] quality: Integer representing the quality: 10, 20, 30.. 100.
    :return: None.
    """


    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)
    check_str(name=f'{convert_to_avi.__name__} codec', value=codec, options=('xvid', 'divx', 'mjpeg'))
    check_instance(source=f'{convert_to_avi.__name__} path', instance=path, accepted_types=(str,))
    check_int(name=f'{convert_to_avi.__name__} quality', value=quality)
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    crf_lk = percent_to_crf_lookup()
    cv_lk = percent_to_qv_lk()
    crf = crf_lk[str(quality)]
    qv = cv_lk[int(quality)]
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=convert_to_avi.__name__)
    if os.path.isfile(path):
        file_paths = [path]
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(path), f'mp4_{datetime_}')
            os.makedirs(save_dir)
    elif os.path.isdir(path):
        file_paths = find_files_of_filetypes_in_directory(directory=path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        if save_dir is None:
            save_dir = os.path.join(path, f'avi_{datetime_}')
            os.makedirs(save_dir)
    else:
        raise InvalidInputError(msg=f'Paths is not a valid file or directory path.', source=convert_to_avi.__name__)
    for file_cnt, file_path in enumerate(file_paths):
        _, video_name, _ = get_fn_ext(filepath=file_path)
        print(f'Converting video {video_name} to avi (Video {file_cnt+1}/{len(file_paths)})...')
        _ = get_video_meta_data(video_path=file_path)
        out_path = os.path.join(save_dir, f'{video_name}.avi')
        if codec == 'divx':
            cmd = f'ffmpeg -i "{file_path}" -c:v mpeg4 -crf {crf} -vtag DIVX "{out_path}" -loglevel error -stats -hide_banner -y'
        elif codec == 'xvid':
            cmd = f'ffmpeg -i "{file_path}" -c:v libxvid -q:v {qv} "{out_path}" -loglevel error -stats -hide_banner -y'
        else:
            cmd = f'ffmpeg -i "{file_path}" -c:v mjpeg -q:v {qv} "{out_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"{len(file_paths)} video(s) converted to AVI and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=convert_to_avi.__name__,)


def convert_to_webm(path: Union[str, os.PathLike],
                    codec: Literal['vp8', 'vp9', 'av1'] = 'vp9',
                    save_dir: Optional[Union[str, os.PathLike]] = None,
                    quality: Optional[int] = 60) -> None:

    """
    Convert a directory containing videos, or a single video, to WEBM format using passed quality and codec.

    :param Union[str, os.PathLike] path: Path to directory or file.
    :param Literal['vp8', 'vp9', 'av1'] codec: Method to encode the WEBM format. Default: vp9.
    :param Optional[Optional[Union[str, os.PathLike]]] save_dir: Directory where to save the converted videos. If None, then creates a directory in the same directory as the input.
    :param Optional[int] quality: Integer representing the quality: 10, 20, 30.. 100.
    :return: None.
    """

    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)
    check_str(name=f'{convert_to_webm.__name__} codec', value=codec, options=('vp8', 'vp9', 'av1'))
    check_instance(source=f'{convert_to_webm.__name__} path', instance=path, accepted_types=(str,))
    check_int(name=f'{convert_to_webm.__name__} quality', value=quality)
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    crf_lk = percent_to_crf_lookup()
    crf = crf_lk[str(quality)]
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=convert_to_webm.__name__)
    if os.path.isfile(path):
        file_paths = [path]
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(path), f'webm_{datetime_}')
            os.makedirs(save_dir)
    elif os.path.isdir(path):
        file_paths = find_files_of_filetypes_in_directory(directory=path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        if save_dir is None:
            save_dir = os.path.join(path, f'webm_{datetime_}')
        os.makedirs(save_dir)
    else:
        raise InvalidInputError(msg=f'Paths is not a valid file or directory path.', source=convert_to_webm.__name__)
    for file_cnt, file_path in enumerate(file_paths):
        _, video_name, _ = get_fn_ext(filepath=file_path)
        print(f'Converting video {video_name} to WEBM (Video {file_cnt+1}/{len(file_paths)})...')
        _ = get_video_meta_data(video_path=file_path)
        out_path = os.path.join(save_dir, f'{video_name}.webm')
        if codec == 'vp8':
            cmd = f'ffmpeg -i "{file_path}" -c:v libvpx -crf {crf} "{out_path}" -loglevel error -stats -hide_banner -y'
        elif codec == 'vp9':
            cmd = f'ffmpeg -i "{file_path}" -c:v libvpx-vp9 -crf {crf} "{out_path}" -loglevel error -stats -hide_banner -y'
        else:
            cmd = f'ffmpeg -i "{file_path}" -c:v libaom-av1 -crf {crf} "{out_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"{len(file_paths)} video(s) converted to WEBM and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=convert_to_webm.__name__,)


def convert_to_mov(path: Union[str, os.PathLike],
                    codec: Literal['prores', 'animation', 'dnxhd', 'cineform'] = 'prores',
                    save_dir: Optional[Union[str, os.PathLike]] = None,
                    quality: Optional[int] = 60) -> None:
    """
    Convert a directory containing videos, or a single video, to MOV format using passed quality and codec.

    :param Union[str, os.PathLike] path: Path to directory or file.
    :param Literal['prores', 'animation'] codec: Method to encode the MOV format. Default: prores.
    :param Optional[Optional[Union[str, os.PathLike]]] save_dir: Directory where to save the converted videos. If None, then creates a directory in the same directory as the input.
    :param Optional[int] quality: Integer representing the quality: 10, 20, 30.. 100.
    :return: None.
    """

    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)
    check_str(name=f'{convert_to_mov.__name__} codec', value=codec, options=('prores', 'animation', 'cineform', 'dnxhd'))
    check_instance(source=f'{convert_to_mov.__name__} path', instance=path, accepted_types=(str,))
    check_int(name=f'{convert_to_mov.__name__} quality', value=quality)
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=convert_to_mov.__name__)
    if os.path.isfile(path):
        file_paths = [path]
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(path), f'mp4_{datetime_}')
            os.makedirs(save_dir)
    elif os.path.isdir(path):
        file_paths = find_files_of_filetypes_in_directory(directory=path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        if save_dir is None:
            save_dir = os.path.join(path, f'mov_{datetime_}')
            os.makedirs(save_dir)
    else:
        raise InvalidInputError(msg=f'Paths is not a valid file or directory path.', source=convert_to_mov.__name__)
    for file_cnt, file_path in enumerate(file_paths):
        _, video_name, _ = get_fn_ext(filepath=file_path)
        print(f'Converting video {video_name} to MOV (Video {file_cnt + 1}/{len(file_paths)})...')
        _ = get_video_meta_data(video_path=file_path)
        out_path = os.path.join(save_dir, f'{video_name}.mov')
        if codec == 'prores':
            cmd = f'ffmpeg -i "{file_path}" -c:v prores_ks -profile:v 1 "{out_path}" -loglevel error -stats -hide_banner -y'
        elif codec == 'dnxhd':
            cmd = f'ffmpeg -i "{file_path}" -c:v dnxhd -profile:v dnxhr_mq "{out_path}" -loglevel error -stats -hide_banner -y'
        elif codec == "cineform":
            cmd = f'ffmpeg -i "{file_path}" -c:v cfhd -compression_level 5 -q:v 3 "{out_path}" -loglevel error -stats -hide_banner -y'
        else:
            cmd = f'ffmpeg -i "{file_path} -c:v qtrle "{out_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"{len(file_paths)} video(s) converted to MOV and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=convert_to_mov.__name__, )

#convert_to_mov(path='/Users/simon/Desktop/video_test', quality=0)


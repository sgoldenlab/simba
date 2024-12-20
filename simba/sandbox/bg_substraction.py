
from typing import Union, Optional, Tuple, Dict, Any
import cv2
import multiprocessing
from datetime import datetime
import functools
import os
import numpy as np
from copy import deepcopy
from simba.utils.checks import check_file_exist_and_readable, check_int, check_that_hhmmss_start_is_before_end, check_if_string_value_is_valid_video_timestamp, check_if_dir_exists
from simba.utils.data import find_frame_numbers_from_time_stamp
from simba.utils.read_write import get_video_meta_data, check_if_hhmmss_timestamp_is_valid_part_of_video, get_fn_ext, find_core_cnt, concatenate_videos_in_folder
from simba.utils.errors import InvalidInputError
from simba.utils.enums import Formats
from simba.utils.printing import SimbaTimer, stdout_success

def create_average_frm(video_path: Union[str, os.PathLike],
                       start_frm: Optional[int] = None,
                       end_frm: Optional[int] = None,
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None,
                       save_path: Optional[Union[str, os.PathLike]] = None) -> Union[None, np.ndarray]:

    """
    Create an image representing the average frame of a segment in a video or an entire video.

    .. note::
       Either pass ``start_frm`` and ``end_frm`` OR ``start_time`` and ``end_time`` OR pass all four arguments as None.
       If all are None, then the entire video will be used to create the average frame.

    :param Union[str, os.PathLike] video_path: The path to the video to create the average frame from. Default: None.
    :param Optional[int] start_frm: The first frame in the segment to create the average frame from. Default: None.
    :param Optional[int] end_frm: The last frame in the segment to create the average frame from. Default: None.
    :param Optional[str] start_time: The start timestamp in `HH:MM:SS` format in the segment to create the average frame from. Default: None.
    :param Optional[str] end_time: The end timestamp in `HH:MM:SS` format in the segment to create the average frame from. Default: None.
    :param Optional[Union[str, os.PathLike]] save_path: The path to where to save the average image. If None, then reaturens the average image in np,ndarray format. Default: None.
    :return Union[None, np.ndarray]: The average image (if ``save_path`` is not None) or None if  ``save_path`` is passed.
    """

    if ((start_frm is not None) or (end_frm is not None)) and ((start_time is not None) or (end_time is not None)):
        raise InvalidInputError(msg=f'Pass start_frm and end_frm OR start_time and end_time', source=create_average_frm.__name__)
    elif type(start_frm) != type(end_frm):
        raise InvalidInputError(msg=f'Pass start frame and end frame', source=create_average_frm.__name__)
    elif type(start_time) != type(end_time):
        raise InvalidInputError(msg=f'Pass start time and end time', source=create_average_frm.__name__)
    check_file_exist_and_readable(file_path=video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    cap = cv2.VideoCapture(video_path)
    if (start_frm is not None) and (end_frm is not None):
        check_int(name='start_frm', value=start_frm, min_value=0, max_value=video_meta_data['frame_count'])
        check_int(name='end_frm', value=end_frm, min_value=0, max_value=video_meta_data['frame_count'])
        if start_frm > end_frm:
            raise InvalidInputError(msg=f'Start frame ({start_frm}) has to be before end frame ({end_frm}).', source=create_average_frm.__name__)
        frame_ids = list(range(start_frm, end_frm+1))
    elif (start_time is not None) and (end_time is not None):
        check_if_string_value_is_valid_video_timestamp(value=start_time, name=create_average_frm.__name__)
        check_if_string_value_is_valid_video_timestamp(value=end_time, name=create_average_frm.__name__)
        check_that_hhmmss_start_is_before_end(start_time=start_time, end_time=end_time, name=create_average_frm.__name__)
        check_if_hhmmss_timestamp_is_valid_part_of_video(timestamp=start_time, video_path=video_path)
        frame_ids = find_frame_numbers_from_time_stamp(start_time=start_time, end_time=end_time, fps=video_meta_data['fps'])
    else:
        frame_ids = list(range(0, video_meta_data['frame_count']))
    cap.set(0, frame_ids[0])
    bg_sum, frm_cnt, frm_len = None, 0, len(frame_ids)
    while frm_cnt <= frm_len:
        ret, frm = cap.read()
        if bg_sum is None: bg_sum = np.float32(frm)
        else: cv2.accumulate(frm, bg_sum)
        frm_cnt += 1
    img = cv2.convertScaleAbs(bg_sum / frm_len)
    cap.release()
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=create_average_frm.__name__)
        cv2.imwrite(save_path, img)
    else:
        return img

def _bg_remover_mp(frm_range: Tuple[int, np.ndarray],
                   video_path: Union[str, os.PathLike],
                   bg_frm: np.ndarray,
                   bg_clr: Tuple[int, int, int],
                   fg_clr: Tuple[int, int, int],
                   video_meta_data: Dict[str, Any],
                   temp_dir: Union[str, os.PathLike]):

    batch, frm_range = frm_range[0], frm_range[1]
    start_frm, current_frm, end_frm = frm_range[0], frm_range[0], frm_range[-1]
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    save_path = os.path.join(temp_dir, f"{batch}.mp4")
    cap.set(1, start_frm)
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'], (video_meta_data['width'], video_meta_data['height']))
    bg = np.full_like(bg_frm, bg_clr)
    bg = bg[:, :, ::-1]
    dir, video_name, ext = get_fn_ext(filepath=video_path)
    while current_frm <= end_frm:
        ret, frm = cap.read()
        if not ret:
            break
        diff = cv2.absdiff(frm, bg_frm)
        b, g, r = diff[:, :, 0], diff[:, :, 1], diff[:, :, 2]
        gray_diff = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray_diff = gray_diff.astype(np.uint8)  # Ensure the type is uint8
        mask = np.where(gray_diff > 50, 255, 0).astype(np.uint8)
        if fg_clr is None:
            fg = cv2.bitwise_and(frm, frm, mask=mask)
            result = cv2.add(bg, fg)
        else:
            mask_inv = cv2.bitwise_not(mask)
            fg_clr_img = np.full_like(frm, fg_clr)
            fg_clr_img = fg_clr_img[:, :, ::-1]
            fg_clr_img = cv2.bitwise_and(fg_clr_img, fg_clr_img, mask=mask)
            result = cv2.bitwise_and(bg, bg, mask=mask_inv)
            result = cv2.add(result, fg_clr_img)
        writer.write(result)
        current_frm += 1
        print(f'Background subtraction frame {current_frm}/{video_meta_data["frame_count"]} (Video: {video_name})')
    writer.release()
    cap.release()
    return batch

def video_bg_substraction_mp(video_path: Union[str, os.PathLike],
                             bg_video_path: Optional[Union[str, os.PathLike]] = None,
                             bg_start_frm: Optional[int] = None,
                             bg_end_frm: Optional[int] = None,
                             bg_start_time: Optional[str] = None,
                             bg_end_time: Optional[str] = None,
                             bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
                             fg_color: Optional[Tuple[int, int, int]] = None,
                             save_path: Optional[Union[str, os.PathLike]] = None,
                             core_cnt: Optional[int] = -1) -> None:

    """
    Subtract the background from a video using multiprocessing.

    .. video:: _static/img/video_bg_subtraction.webm
       :width: 900
       :autoplay:
       :nocontrols:
       :loop:

    .. note::
        For single core alternative, see ``simba.video_processors.video_processing.video_bg_subtraction``

       If  ``bg_video_path`` is passed, that video will be used to parse the background. If None, ``video_path`` will be use dto parse background.
       Either pass ``start_frm`` and ``end_frm`` OR ``start_time`` and ``end_time`` OR pass all four arguments as None.
       Those two arguments will be used to slice the background video, and the sliced part is used to parse the background.

       For example, in the scenario where there is **no** animal in the ``video_path`` video for the first 20s, then the first 20s can be used to parse the background.
       In this scenario, ``bg_video_path`` can be passed as ``None`` and bg_start_time and bg_end_time can be ``00:00:00`` and ``00:00:20``, repectively.

       In the scenario where there **is** animal(s) in the entire ``video_path`` video, pass ``bg_video_path`` as a path to a video recording the arena without the animals.

    :param Union[str, os.PathLike] video_path: The path to the video to remove the background from.
    :param Optional[Union[str, os.PathLike]] bg_video_path: Path to the video which contains a segment with the background only. If None, then ``video_path`` will be used.
    :param Optional[int] bg_start_frm: The first frame in the background video to use when creating a representative background image. Default: None.
    :param Optional[int] bg_end_frm: The last frame in the background video to use when creating a representative background image. Default: None.
    :param Optional[str] bg_start_time: The start timestamp in `HH:MM:SS` format in the background video to use to create a representative background image. Default: None.
    :param Optional[str] bg_end_time: The end timestamp in `HH:MM:SS` format in the background video to use to create a representative background image. Default: None.
    :param Optional[Tuple[int, int, int]] bg_color: The RGB color of the moving objects in the output video. Defaults to None, which represents the original colors of the moving objects.
    :param Optional[Tuple[int, int, int]] fg_color: The RGB color of the background output video. Defaults to black (0, 0, 0).
    :param Optional[Union[str, os.PathLike]] save_path: The patch to where to save the output video where the background is removed. If None, saves the output video in the same directory as the input video with the ``_bg_subtracted`` suffix. Default: None.
    :param Optional[int] core_cnt: The number of cores to use. Defaults to -1 representing all available cores.
    :return: None.

    :example:
    >>> video_bg_substraction_mp(video_path='/Users/simon/Downloads/1_LH.mp4', bg_start_time='00:00:00', bg_end_time='00:00:10', bg_color=(0, 0, 0), fg_color=(255, 255, 255))
    """

    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=video_path)
    if bg_video_path is None:
        bg_video_path = deepcopy(video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    dir, video_name, ext = get_fn_ext(filepath=video_path)
    if save_path is None:
        save_path = os.path.join(dir, f'{video_name}_bg_subtracted{ext}')
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    temp_dir = os.path.join(dir, f'temp_{video_name}_{dt}')
    os.makedirs(temp_dir)
    check_int(name=f'{video_bg_substraction_mp.__name__} core_cnt', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0])
    if core_cnt == -1: core_cnt = find_core_cnt()[0]
    bg_frm = create_average_frm(video_path=bg_video_path, start_frm=bg_start_frm, end_frm=bg_end_frm, start_time=bg_start_time, end_time=bg_end_time)
    bg_frm = cv2.resize(bg_frm, (video_meta_data['width'], video_meta_data['height']))
    bg_frm = bg_frm[:, :, ::-1]
    frm_list = np.array_split(list(range(0, video_meta_data['frame_count'])), core_cnt)
    frm_data = []
    for c, i in enumerate(frm_list):
        frm_data.append((c, i))
    with multiprocessing.Pool(core_cnt, maxtasksperchild=9000) as pool:
        constants = functools.partial(_bg_remover_mp,
                                      video_path=video_path,
                                      bg_frm=bg_frm,
                                      bg_clr=bg_color,
                                      fg_clr=fg_color,
                                      video_meta_data=video_meta_data,
                                      temp_dir=temp_dir)
        for cnt, result in enumerate(pool.imap(constants, frm_data, chunksize=1)):
            print(f'Frame batch {result+1} completed...')
        pool.terminate()
        pool.join()

    print(f"Joining {video_name} multiprocessed video...")
    concatenate_videos_in_folder(in_folder=temp_dir, save_path=save_path, video_format=ext[1:], remove_splits=False)
    timer.stop_timer()
    stdout_success(msg=f'Video saved at {save_path}', elapsed_time=timer.elapsed_time_str)

# video_bg_substraction_mp(video_path='/Users/simon/Downloads/1_LH.mp4',
#                          bg_start_time='00:00:00',
#                          bg_end_time='00:00:10',
#                          bg_color=(0, 0, 0),
#                          fg_color=(255, 255, 255))




    # bg = cv2.cvtColor(np.full_like(bg_frm, bg_color), cv2.COLOR_BGR2RGB)
    # cap = cv2.VideoCapture(video_path)
    # frm_cnt = 0
    # while True:
    #     ret, frm = cap.read()
    #     if not ret:
    #         break
    #     diff = cv2.absdiff(frm, bg_frm)
    #     gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #     _, mask = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)
    #     if fg_color is None:
    #         fg = cv2.bitwise_and(frm, frm, mask=mask)
    #         result = cv2.add(bg, fg)
    #     else:
    #         mask_inv = cv2.bitwise_not(mask)
    #         fg_clr = cv2.cvtColor(np.full_like(frm, fg_color), cv2.COLOR_BGR2RGB)
    #         fg_clr = cv2.bitwise_and(fg_clr, fg_clr, mask=mask)
    #         result = cv2.bitwise_and(bg, bg, mask=mask_inv)
    #         result = cv2.add(result, fg_clr)
    #     writer.write(result)
    #     frm_cnt+= 1
    #     print(f'Background subtraction frame {frm_cnt}/{video_meta_data["frame_count"]} (Video: {video_name})')
    #
    # writer.release()
    # cap.release()
    # timer.stop_timer()
    # stdout_success(msg=f'Background subtracted from {video_name} and saved at {save_path}', elapsed_time=timer.elapsed_time)







# def bg_substraction(bg_video: Union[str, os.PathLike, cv2.VideoCapture],
#                     video: Union[str, os.PathLike, cv2.VideoCapture]):
#
#
#     if isinstance(bg_video, str): bg_video = cv2.VideoCapture(bg_video)
#     bg_sum, bg_meta = None, get_video_meta_data(video_path=bg_video)
#     while True:
#         ret, frm = bg_video.read()
#         if ret:
#             if bg_sum is None:
#                 bg_sum = np.float32(frm)
#             else:
#                 cv2.accumulate(frm, bg_sum)
#         else: break
#     background_model = cv2.convertScaleAbs(bg_sum / bg_meta['frame_count'])
#     bg_video.release()
#
#     if isinstance(video, str):
#         video_meta = get_video_meta_data(video_path=bg_video)
#         video = cv2.VideoCapture(video)
#
#     while True:
#         ret, frame = video.read()
#
#
#
# while True:
#
#
#     if not ret:
#         break
#
#     # Resize the background model to match the frame dimensions
#     background_resized = cv2.resize(background_model, (frame.shape[1], frame.shape[0]))
#
#     # Calculate the absolute difference between the frame and the background model
#     diff = cv2.absdiff(frame, background_resized)
#
#     # Convert the difference image to grayscale
#     gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
#
#     # Threshold the grayscale difference image to create a binary mask
#     _, mask = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)
#
#     # Invert the mask to get foreground as white and background as black
#     mask_inv = cv2.bitwise_not(mask)
#
#     # Create a black background
#     black_background = np.zeros_like(frame)
#
#     # Composite the foreground onto the black background using the inverted mask
#     foreground = cv2.bitwise_and(frame, frame, mask=mask)
#
#     # Composite the black background with the foreground
#     result = cv2.add(black_background, foreground)
#
#     # Show the original frame and the frame with background black and foreground with original colors
#     cv2.imshow('Original', frame)
#     cv2.imshow('Frame with Background Black and Foreground Retaining Original Colors', result)
#
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break
#
# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()


__author__ = "Simon Nilsson"

import os
import signal
import sys

import cv2
import numpy as np

from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.enums import TextOptions
from simba.utils.lookups import get_labelling_video_kbd_bindings
from simba.utils.read_write import (get_fn_ext, get_video_meta_data,
                                    read_frm_of_video)
from simba.utils.warnings import FrameRangeWarning


def annotation_video_player():
    """Private methods for playing the video that is being annotated in SimBA GUI"""

    def labelling_log_writer(frame_number: int) -> None:
        f.seek(0)
        f.write(str(frame_number))
        f.truncate()
        f.flush()
        os.fsync(f.fileno())

    video_path = sys.stdin.readline().encode().decode()
    check_file_exist_and_readable(file_path=video_path)
    project_dir = os.path.dirname(os.path.dirname(video_path))
    cap = cv2.VideoCapture(video_path)
    _, video_name, _ = get_fn_ext(filepath=video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    font_size, space_x, spacing_scale = PlottingMixin().get_optimal_font_scales(text='999999', accepted_px_width=int(video_meta_data["width"] / 8), accepted_px_height=int(video_meta_data["width"] / 8))
    kbd_bindings = get_labelling_video_kbd_bindings()
    f = open(os.path.join(project_dir, "labelling_info.txt"), "w")
    time_between_frames = int(1000 / video_meta_data["fps"])
    cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)

    def print_video_txt(frame: np.ndarray,
                        frame_number: int,
                        video_info: dict) -> np.ndarray:

        current_time = round((frame_number / video_info["fps"]), 2)
        frame = PlottingMixin().put_text(img=frame, text=f"F~ {frame_number}", pos=(TextOptions.BORDER_BUFFER_X.value, int((video_info["height"] - spacing_scale))), font_size=font_size, font_thickness=TextOptions.TEXT_THICKNESS.value + 1, text_bg_alpha=0.6, text_color=(255, 255, 255))
        frame = PlottingMixin().put_text(img=frame, text=f"T~ {current_time}", pos=(TextOptions.BORDER_BUFFER_X.value, int((video_info["height"] - spacing_scale * 2))), font_size=font_size, font_thickness=TextOptions.TEXT_THICKNESS.value + 1, text_bg_alpha=0.6, text_color=(255, 255, 255))
        return frame


    pause_key = kbd_bindings['Pause/Play']['kbd']
    f_2_key = kbd_bindings['forward_two_frames']['kbd']
    f_10_key = kbd_bindings['forward_ten_frames']['kbd']
    f_1s_key = kbd_bindings['forward_one_second']['kbd']
    b_2_key = kbd_bindings['backwards_two_frames']['kbd']
    b_10_key = kbd_bindings['backwards_ten_frames']['kbd']
    b_1s_key = kbd_bindings['backwards_one_second']['kbd']
    close = kbd_bindings['close_window']['kbd']



    while True:
        ret, frame = cap.read()
        if frame is None:
            cap.release()
            cv2.destroyAllWindows()
            break
        key = cv2.waitKey(time_between_frames) & 0xFF
        if (isinstance(pause_key, int) and key == pause_key) or (isinstance(pause_key, str) and key == ord(pause_key)):
            while True:
                second_key = cv2.waitKey(1)
                current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                f.seek(0)
                f.write(str(current_video_position - 1))
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
                if (isinstance(b_2_key, int) and second_key == b_2_key) or (isinstance(b_2_key, str) and second_key == ord(b_2_key)):
                    if (current_video_position - 3) < 0:
                        FrameRangeWarning(msg=f"FRAME {current_video_position - 3} CANNOT BE SHOWN", source=annotation_video_player.__name__)
                    else:
                        current_video_position = current_video_position - 3
                        frame = read_frm_of_video(video_path=cap, frame_index=current_video_position)
                        frame = print_video_txt(frame=frame, frame_number=current_video_position, video_info=video_meta_data)
                        cv2.imshow(video_name, frame)
                        labelling_log_writer(frame_number=current_video_position)
                elif (isinstance(b_10_key, int) and second_key == b_10_key) or (isinstance(b_10_key, str) and second_key == ord(b_10_key)):
                    if (current_video_position - 11) < 0:
                        FrameRangeWarning(msg=f"FRAME {current_video_position - 11} CANNOT BE SHOWN", source=annotation_video_player.__name__)
                    else:
                        current_video_position = current_video_position - 11
                        frame = read_frm_of_video(video_path=cap, frame_index=current_video_position)
                        frame = print_video_txt(frame=frame,frame_number=current_video_position, video_info=video_meta_data)
                        cv2.imshow(video_name, frame)
                        labelling_log_writer(frame_number=current_video_position)
                elif (isinstance(b_1s_key, int) and second_key == b_1s_key) or (isinstance(b_1s_key, str) and second_key == ord(b_1s_key)):
                    if (current_video_position - video_meta_data["fps"]) < 0:
                        FrameRangeWarning(msg=f'FRAME {current_video_position - video_meta_data["fps"]} CANNOT BE SHOWN', source=annotation_video_player.__name__)
                    else:
                        current_video_position = int(current_video_position - video_meta_data["fps"])
                        frame = read_frm_of_video(video_path=cap, frame_index=current_video_position)
                        current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        frame = print_video_txt(frame=frame,frame_number=current_video_position, video_info=video_meta_data)
                        cv2.imshow(video_name, frame)
                        labelling_log_writer(frame_number=current_video_position)
                elif (isinstance(f_1s_key, int) and second_key == f_1s_key) or (isinstance(f_1s_key, str) and second_key == ord(f_1s_key)):
                    if (current_video_position + video_meta_data["fps"]) > video_meta_data["frame_count"]:
                        FrameRangeWarning(msg=f'FRAME {current_video_position + video_meta_data["fps"]} CANNOT BE SHOWN', source=annotation_video_player.__name__)
                    else:
                        current_video_position = int(current_video_position + video_meta_data["fps"])
                        frame = read_frm_of_video(video_path=cap, frame_index=current_video_position)
                        frame = print_video_txt(frame=frame,frame_number=current_video_position, video_info=video_meta_data)
                        cv2.imshow(video_name, frame)
                        labelling_log_writer(frame_number=current_video_position)
                elif (isinstance(f_2_key, int) and second_key == f_2_key) or (isinstance(f_2_key, str) and second_key == ord(f_2_key)):
                    if (current_video_position + 1) > video_meta_data["frame_count"]:
                        FrameRangeWarning( msg=f"FRAME {current_video_position + 1} CANNOT BE SHOWN", source=annotation_video_player.__name__)
                    else:
                        current_video_position = int(current_video_position + 1)
                        frame = read_frm_of_video(video_path=cap, frame_index=current_video_position)
                        frame = print_video_txt(frame=frame,frame_number=current_video_position, video_info=video_meta_data)
                        cv2.imshow(video_name, frame)
                        labelling_log_writer(frame_number=current_video_position)
                elif (isinstance(f_10_key, int) and second_key == f_10_key) or (isinstance(f_10_key, str) and second_key == ord(f_10_key)):
                    if (current_video_position + 9) > video_meta_data["frame_count"]:
                        FrameRangeWarning(msg=f"FRAME {current_video_position + 9} CANNOT BE SHOWN", source=annotation_video_player.__name__)
                    else:
                        current_video_position = int(current_video_position + 9)
                        frame = read_frm_of_video(video_path=cap, frame_index=current_video_position)
                        frame = print_video_txt(frame=frame, frame_number=current_video_position, video_info=video_meta_data)
                        cv2.imshow(video_name, frame)
                        labelling_log_writer(frame_number=current_video_position)
                elif (isinstance(pause_key, int) and second_key == pause_key) or (isinstance(pause_key, str) and second_key == ord(pause_key)):
                    break

                if (isinstance(close, int) and second_key == close) or (isinstance(close, str) and second_key == ord(close)) or (cv2.getWindowProperty(video_name, 1) == -1):
                    cap.release()
                    cv2.destroyAllWindows()
                    path = os.path.join(project_dir, "subprocess.txt")
                    txtFile = open(path)
                    line = txtFile.readline()
                    if (isinstance(close, int) and second_key == close) or (isinstance(close, str) and second_key == ord(close)):
                        os.kill(int(line), signal.SIGTERM)
                        break
                    else:
                        try:
                            os.kill(int(line), signal.SIGTERM)
                        except OSError:
                            print("OSError: Cannot save/read latest image file CSV. Please try again")

        current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame = print_video_txt(frame=frame, frame_number=current_video_position, video_info=video_meta_data)
        cv2.imshow(video_name, frame)
        if (isinstance(close, int) and key == close) or (isinstance(close, str) and key == ord(close)):
            break
        if cv2.getWindowProperty(video_name, 1) == -1:
            break

    cap.release()
    f.close()
    cv2.destroyAllWindows()
    path = os.path.join(project_dir, "subprocess.txt")
    txtFile = open(path)
    line = txtFile.readline()
    os.kill(int(line), signal.SIGTERM)


if __name__ == "__main__":
    annotation_video_player()

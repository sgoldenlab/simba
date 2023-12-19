__author__ = "Simon Nilsson"

import os
import signal
import sys

import cv2

from simba.utils.checks import check_file_exist_and_readable
from simba.utils.enums import TextOptions
from simba.utils.read_write import get_video_meta_data
from simba.utils.warnings import FrameRangeWarning


def annotation_video_player():
    """Private methods for playing the video that is being annotated in SimBA GUI"""

    def labelling_log_writer(frame_number: int) -> None:
        f.seek(0)
        f.write(str(frame_number))
        f.truncate()
        f.flush()
        os.fsync(f.fileno())

    def print_video_txt(frame_number: int, video_info: dict) -> None:
        current_time = round((frame_number / video_info["fps"]), 2)
        cv2.putText(
            frame,
            f"F~ {frame_number}",
            (
                TextOptions.BORDER_BUFFER_X.value,
                int((video_info["height"] - spacing_scale)),
            ),
            TextOptions.FONT.value,
            font_size,
            TextOptions.COLOR.value,
            TextOptions.TEXT_THICKNESS.value + 1,
        )
        cv2.putText(
            frame,
            f"T~ {current_time}",
            (
                TextOptions.BORDER_BUFFER_X.value,
                int((video_info["height"] - spacing_scale * 2)),
            ),
            TextOptions.FONT.value,
            font_size,
            TextOptions.COLOR.value,
            TextOptions.TEXT_THICKNESS.value + 1,
        )

    video_path = sys.stdin.readline().encode().decode()
    check_file_exist_and_readable(file_path=video_path)
    project_dir = os.path.dirname(os.path.dirname(video_path))
    cap = cv2.VideoCapture(video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    space_scale, res_scale, font_scale = 60, 1500, 2
    max_dim = max(video_meta_data["width"], video_meta_data["height"])
    font_size = float(font_scale / (res_scale / max_dim))
    spacing_scale = int(space_scale / (res_scale / max_dim))

    f = open(os.path.join(project_dir, "labelling_info.txt"), "w")
    time_between_frames = int(1000 / video_meta_data["fps"])
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        key = cv2.waitKey(time_between_frames) & 0xFF
        if key == ord("p"):  ### THE VIDEO IS PAUSED
            while True:
                second_key = cv2.waitKey(1)
                current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                f.seek(0)
                f.write(str(current_video_position - 1))
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
                if second_key == ord("t"):  ## BACK UP TWO FRAME
                    if (current_video_position - 2) < 0:
                        FrameRangeWarning(
                            msg=f"FRAME {current_video_position - 2} CANNOT BE SHOWN",
                            source=annotation_video_player.__name__,
                        )
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, (current_video_position - 2))
                        ret, frame = cap.read()
                        current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        print_video_txt(
                            frame_number=current_video_position,
                            video_info=video_meta_data,
                        )
                        cv2.imshow("Video", frame)
                        labelling_log_writer(frame_number=current_video_position)
                elif second_key == ord("s"):  ### BACK UP TEN FRAME
                    if (current_video_position - 11) < 0:
                        FrameRangeWarning(
                            msg=f"FRAME {current_video_position - 11} CANNOT BE SHOWN",
                            source=annotation_video_player.__name__,
                        )
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, (current_video_position - 11))
                        ret, frame = cap.read()
                        current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        print_video_txt(
                            frame_number=current_video_position,
                            video_info=video_meta_data,
                        )
                        cv2.imshow("Video", frame)
                        labelling_log_writer(frame_number=current_video_position)
                elif second_key == ord("x"):  ### BACK UP 1s
                    if (current_video_position - video_meta_data["fps"]) < 0:
                        FrameRangeWarning(
                            msg=f'FRAME {current_video_position - video_meta_data["fps"]} CANNOT BE SHOWN',
                            source=annotation_video_player.__name__,
                        )
                    else:
                        cap.set(
                            cv2.CAP_PROP_POS_FRAMES,
                            (current_video_position - video_meta_data["fps"]),
                        )
                        ret, frame = cap.read()
                        current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        print_video_txt(
                            frame_number=current_video_position,
                            video_info=video_meta_data,
                        )
                        cv2.imshow("Video", frame)
                        labelling_log_writer(frame_number=current_video_position)
                elif second_key == ord("w"):  ### FORWARD 1s
                    if (
                        current_video_position + video_meta_data["fps"]
                    ) > video_meta_data["frame_count"]:
                        FrameRangeWarning(
                            msg=f'FRAME {current_video_position + video_meta_data["fps"]} CANNOT BE SHOWN',
                            source=annotation_video_player.__name__,
                        )
                    else:
                        cap.set(
                            cv2.CAP_PROP_POS_FRAMES,
                            (current_video_position + video_meta_data["fps"]),
                        )
                        ret, frame = cap.read()
                        current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        print_video_txt(
                            frame_number=current_video_position,
                            video_info=video_meta_data,
                        )
                        cv2.imshow("Video", frame)
                        labelling_log_writer(frame_number=current_video_position)
                elif second_key == ord("o"):  ### FORWARD TWO FRAMES
                    if (current_video_position + 1) > video_meta_data["frame_count"]:
                        FrameRangeWarning(
                            msg=f"FRAME {current_video_position + 1} CANNOT BE SHOWN",
                            source=annotation_video_player.__name__,
                        )
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, (current_video_position + 1))
                        ret, frame = cap.read()
                        current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        print_video_txt(
                            frame_number=current_video_position,
                            video_info=video_meta_data,
                        )
                        cv2.imshow("Video", frame)
                        labelling_log_writer(frame_number=current_video_position)

                elif second_key == ord("e"):  ### FORWARD TEN FRAMES
                    if (current_video_position + 9) > video_meta_data["frame_count"]:
                        FrameRangeWarning(
                            msg=f"FRAME {current_video_position + 9} CANNOT BE SHOWN",
                            source=annotation_video_player.__name__,
                        )
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, (current_video_position + 9))
                        ret, frame = cap.read()
                        current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        print_video_txt(
                            frame_number=current_video_position,
                            video_info=video_meta_data,
                        )
                        cv2.imshow("Video", frame)
                        labelling_log_writer(frame_number=current_video_position)
                elif second_key == ord("p"):
                    break

                if (second_key == ord("q")) | (cv2.getWindowProperty("Video", 1) == -1):
                    cap.release()
                    cv2.destroyAllWindows()
                    path = os.path.join(project_dir, "subprocess.txt")
                    txtFile = open(path)
                    line = txtFile.readline()
                    if second_key == ord("q"):
                        os.kill(int(line), signal.SIGTERM)
                        break
                    else:
                        try:
                            os.kill(int(line), signal.SIGTERM)
                        except OSError:
                            print(
                                "OSError: Cannot save/read latest image file CSV. Please try again"
                            )

        current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print_video_txt(frame_number=current_video_position, video_info=video_meta_data)
        cv2.imshow("Video", frame)
        if key == ord("q"):
            break
        if cv2.getWindowProperty("Video", 1) == -1:
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

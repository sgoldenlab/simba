__author__ = "Simon Nilsson", "JJ Choong"

import cv2
import sys
import os
import signal
from simba.read_config_unit_tests import check_file_exist_and_readable
from simba.misc_tools import get_video_meta_data
from simba.misc_tools import get_color_dict

def annotation_video_player():

    def labelling_log_writer(frame_number: int) -> None:
        f.seek(0)
        f.write(str(frame_number))
        f.truncate()
        f.flush()
        os.fsync(f.fileno())

    def print_video_txt(frame_number: int, video_info: dict) -> None:
        current_time = round((frame_number / video_info['fps']), 2)
        cv2.putText(frame, 'F~ {}'.format(str(frame_number)), (10, int((video_info['height'] - spacing_scale))), cv2.FONT_HERSHEY_SIMPLEX, font_size, colors['Pink'], 2)
        cv2.putText(frame, 'T~ {}'.format(str(current_time)), (10, int((video_info['height'] - spacing_scale*2))), cv2.FONT_HERSHEY_SIMPLEX, font_size, colors['Pink'], 2)

    colors = get_color_dict()
    video_path = sys.stdin.readline().encode().decode()
    check_file_exist_and_readable(file_path=video_path)
    project_dir = os.path.dirname(os.path.dirname(video_path))
    cap = cv2.VideoCapture(video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    space_scale, res_scale, font_scale = 60, 1500, 2
    max_dim = max(video_meta_data['width'], video_meta_data['height'])
    font_size = float(font_scale / (res_scale / max_dim))
    spacing_scale = int(space_scale / (res_scale / max_dim))

    f = open(os.path.join(project_dir, 'labelling_info.txt'), 'w')
    time_between_frames = int(1000/video_meta_data['fps'])
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        key = cv2.waitKey(time_between_frames) & 0xff
        if key == ord('p'): ### THE VIDEO IS PAUSED
            while True:
                second_key = cv2.waitKey(1)
                current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                f.seek(0)
                f.write(str(current_video_position-1))
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
                if second_key == ord('t'): ## BACK UP TWO FRAME
                    cap.set(cv2.CAP_PROP_POS_FRAMES, (current_video_position - 2))
                    ret, frame = cap.read()
                    current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    print_video_txt(frame_number=current_video_position, video_info=video_meta_data)
                    cv2.imshow('Video', frame)
                    labelling_log_writer(frame_number=current_video_position)
                if second_key == ord('s'): ### BACK UP TEN FRAME
                    cap.set(cv2.CAP_PROP_POS_FRAMES, (current_video_position - 11))
                    ret, frame = cap.read()
                    current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    print_video_txt(frame_number=current_video_position, video_info=video_meta_data)
                    cv2.imshow('Video', frame)
                    labelling_log_writer(frame_number=current_video_position)
                if second_key == ord('x'): ### BACK UP 1s
                    cap.set(cv2.CAP_PROP_POS_FRAMES, (current_video_position - video_meta_data['fps']))
                    ret, frame = cap.read()
                    current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    print_video_txt(frame_number=current_video_position, video_info=video_meta_data)
                    cv2.imshow('Video', frame)
                    labelling_log_writer(frame_number=current_video_position)
                if second_key == ord('w'): ### FORWARD 1s
                    cap.set(cv2.CAP_PROP_POS_FRAMES, (current_video_position + video_meta_data['fps']))
                    ret, frame = cap.read()
                    current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    print_video_txt(frame_number=current_video_position, video_info=video_meta_data)
                    cv2.imshow('Video', frame)
                    labelling_log_writer(frame_number=current_video_position)
                if second_key == ord('o'): ### FORWARD TWO FRAMES
                    cap.set(cv2.CAP_PROP_POS_FRAMES, (current_video_position + 1))
                    ret, frame = cap.read()
                    current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    print_video_txt(frame_number=current_video_position, video_info=video_meta_data)
                    cv2.imshow('Video', frame)
                    labelling_log_writer(frame_number=current_video_position)

                if second_key == ord('e'): ### FORWARD TEN FRAMES
                    cap.set(cv2.CAP_PROP_POS_FRAMES, (current_video_position + 9))
                    ret, frame = cap.read()
                    current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    print_video_txt(frame_number=current_video_position, video_info=video_meta_data)
                    cv2.imshow('Video', frame)
                    labelling_log_writer(frame_number=current_video_position)
                if second_key == ord('p'):
                    break

                if (second_key == ord('q')) | (cv2.getWindowProperty('Video', 1) == -1):
                    cap.release()
                    cv2.destroyAllWindows()
                    path = os.path.join(project_dir,'subprocess.txt')
                    txtFile = open(path)
                    line = txtFile.readline()
                    if second_key == ord('q'):
                        os.kill(int(line), signal.SIGTERM)
                        break
                    else:
                        try:
                            os.kill(int(line), signal.SIGTERM)
                        except OSError:
                            print('OSError: Cannot save/read latest image file CSV. Please try again')

        current_video_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print_video_txt(frame_number=current_video_position, video_info=video_meta_data)
        cv2.imshow('Video', frame)
        if key == ord('q'):
            break
        if cv2.getWindowProperty('Video', 1) == -1:
            break

    cap.release()
    f.close()
    cv2.destroyAllWindows()
    path = os.path.join(project_dir, 'subprocess.txt')
    txtFile = open(path)
    line = txtFile.readline()
    os.kill(int(line), signal.SIGTERM)

if __name__ == "__main__":
    annotation_video_player()




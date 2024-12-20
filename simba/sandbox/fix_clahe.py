import os
import cv2
import numpy as np
from typing import Union
from simba.utils.read_write import get_fn_ext, get_video_meta_data
from simba.utils.enums import Formats
from simba.utils.checks import check_file_exist_and_readable


def clahe_enhance_video(file_path: Union[str, os.PathLike]) -> None:
    """
    Convert a single video file to clahe-enhanced greyscale .avi file. The result is saved with prefix
    ``CLAHE_`` in the same directory as in the input file.

    :parameter Union[str, os.PathLike] file_path: Path to video file.

    :example:
    >>> _ = clahe_enhance_video(file_path: 'project_folder/videos/Video_1.mp4')
    """

    dir, file_name, file_ext = get_fn_ext(filepath=file_path)
    save_path = os.path.join(dir, f"CLAHE_{file_name}.avi")
    video_meta_data = get_video_meta_data(file_path)
    fourcc = cv2.VideoWriter_fourcc(*Formats.AVI_CODEC.value)
    print(f"Applying CLAHE on video {file_name}, this might take awhile...")
    cap = cv2.VideoCapture(file_path)
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]), 0)
    clahe_filter = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
    frm_cnt = 0
    try:
        while True:
            ret, img = cap.read()
            if ret:
                frm_cnt += 1
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe_frm = clahe_filter.apply(img)
                writer.write(clahe_frm)
                print(f"CLAHE converted frame {frm_cnt}/{video_meta_data['frame_count']}")
            else:
                break
        cap.release()
        writer.release()
    except Exception as se:
        print(se.args)
        print(f"CLAHE conversion failed for video {file_name}.")
        cap.release()
        writer.release()
        raise ValueError()



    #cap.release()
    #writer.release()
    #         break
    # except Exception as se:
    #     print(se.args)
    #     print(f"CLAHE conversion failed for video {file_name}")
    #     cap.release()
    #     writer.release()
    #     raise ValueError()
    # print(f'Saved video at {save_path}')


#clahe_enhance_video(file_path=r'/Users/simon/Desktop/envs/simba/simba/tests/data/test_projects/mouse_open_field/project_folder/videos/Video1.mp4')


import glob
import os

from simba.utils.read_write import get_fn_ext
from simba.video_processors.video_processing import video_bg_subtraction_mp

video_paths = glob.glob(r'C:\troubleshooting\mitra\project_folder\videos\additional' + '/*.mp4')
save_dir = r"C:\troubleshooting\mitra\project_folder\videos\additional\bg_removed"

video_paths = [r"C:\troubleshooting\mitra\project_folder\videos\additional\501_MA142_Gi_Saline_0517.mp4"]

for file_cnt, file_path in enumerate(video_paths):
    _, video_name, _ = get_fn_ext(filepath=file_path)
    save_path = os.path.join(save_dir, f'{video_name}.mp4')
    if not os.path.isfile(save_path):
        video_bg_subtraction_mp(video_path=file_path, save_path=save_path, verbose=True, bg_color=(255, 255, 255), gpu=False)







"""
Reproduces the maplight_ri resident-intruder 3D run: lift every video, then render the overhead, POV and 2 x 2 videos
for a few example videos.

Run from a terminal (the __main__ guard is required on Windows, the CPU pool uses spawn):
    python run_maplight.py
"""
import os
import sys
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))   # use the repo, not an installed simba

from simba.sandbox.overhead_3d.overhead_3d_lifter import Overhead3DLifter
from simba.sandbox.overhead_3d.overhead_3d_visualizer import Overhead3DVisualizer, combine_overhead_pov

CONFIG_PATH = r'D:\troubleshooting\maplight_ri\project_folder\project_config.ini'
OUT_DIR = r'D:\troubleshooting\maplight_ri\project_folder\pose_3d_preview'
POSE_3D_DIR = os.path.join(OUT_DIR, 'pose_3d')

# maplight's body-part names are shuffled relative to the anatomy (verified on video): lifter role -> project body-part.
BP_MAP = {'Nose': 'RIGHT_EAR', 'Left_Ear': 'NOSE', 'Right_Ear': 'LEFT_EAR', 'Center': 'LEFT_SIDE', 'Left_Side': 'CENTER'}
CAMERA_HEIGHT_MM = 1500                # not measured: the rim looks only ~5% larger than the floor, so the camera is far (~1-3 m)
ARENA_FLOOR_MM = (300, 174)            # Innovive cage: floor
ARENA_TOP_MM = (355, 215)              # Innovive cage: rim (tapered walls)
WALL_HEIGHT_MM = 127
VIDEOS = ['Trial_10_C24_D1_1', 'Trial_10_C24_D1_3', 'Trial_10_C24_D2_1']

if __name__ == '__main__':
    t = time.time()
    Overhead3DLifter(config_path=CONFIG_PATH, save_dir=POSE_3D_DIR, method='lifter', calibration='camera_height',
                     camera_height_mm=CAMERA_HEIGHT_MM, bp_map=BP_MAP, verbose=True).run()
    print('lifted', round(time.time() - t), 's', flush=True)
    scene = dict(arena_mm=ARENA_FLOOR_MM, arena_top_mm=ARENA_TOP_MM, wall_height_mm=WALL_HEIGHT_MM)
    Overhead3DVisualizer(config_path=CONFIG_PATH, data_dir=POSE_3D_DIR, save_dir=OUT_DIR, video_names=VIDEOS, view='overhead', annotate=False, **scene).run()
    Overhead3DVisualizer(config_path=CONFIG_PATH, data_dir=POSE_3D_DIR, save_dir=OUT_DIR, video_names=VIDEOS, view='pov', pov_fov_deg=180, **scene).run()
    for v in VIDEOS:
        combine_overhead_pov(os.path.join(OUT_DIR, f'{v}.mp4'), os.path.join(OUT_DIR, f'{v}_pov.mp4'), os.path.join(OUT_DIR, f'{v}_2x2.mp4'))
    print('rendered', round(time.time() - t), 's', flush=True)

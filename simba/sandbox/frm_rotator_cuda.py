import numpy as np
from typing import Optional
from simba.utils.read_write import read_img_batch_from_video_gpu, read_df
from simba.utils.data import egocentrically_align_pose_numba



def egocentric_frm_rotator_cuda(imgs: np.ndarray,
                                rotation_matrices: np.ndarray,
                                batch_size: int = 1000):

    pass




DATA_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/data/501_MA142_Gi_Saline_0513.csv"
VIDEO_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/videos/501_MA142_Gi_Saline_0513.mp4"
SAVE_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/rotate_ex/videos/501_MA142_Gi_Saline_0513_rotated.mp4"
ANCHOR_LOC = np.array([300, 300])

df = read_df(file_path=DATA_PATH, file_type='csv')
bp_cols = [x for x in df.columns if not x.endswith('_p')]
data = df[bp_cols].values.reshape(len(df), int(len(bp_cols) / 2), 2).astype(np.int64)
data, centers, rotation_matrices = egocentrically_align_pose_numba(data=data, anchor_1_idx=6, anchor_2_idx=2, anchor_location=ANCHOR_LOC, direction=180)
imgs = read_img_batch_from_video_gpu(video_path=VIDEO_PATH, start_frm=0, end_frm=100)
imgs = np.stack(list(imgs.values()), axis=0)

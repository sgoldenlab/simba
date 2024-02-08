import argparse
import os.path
from copy import deepcopy
from typing import Dict, Optional, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from simba.utils.read_write import read_df
from simba.mixins.config_reader import ConfigReader
from simba.mixins.statistics_mixin import Statistics


CONFIG_PATH = "/Users/simon/Desktop/envs/simba/troubleshooting/RI/project_folder/project_config.ini"
VIDEO_NAME = "RI_01_179"


config = ConfigReader(config_path=CONFIG_PATH, create_logger=False)
data_path = os.path.join(config.features_dir, f"{VIDEO_NAME}.{config.file_type}")
df = read_df(file_path=data_path, file_type=config.file_type)
_, _, fps = config.read_video_info(video_name=VIDEO_NAME)

# COMPARE THE ANGLE DISTRIBUTION OF ONE ANIMAL IN ROLLING 500MS TIME WINDOWS USING INDEPENDENT SAMPLES T-TEST.
# USEFUL TO FIND INSTANCES WHEN ANIMAL RAPIDLY CHANGE ITS TAILBASE-CENTROID-NOSE THREE-POINT ANGLE
data = df['Mouse_1_angle'].values.astype(np.float64)
independent_samples_t = Statistics.rolling_independent_sample_t(data=data, time_window=0.5, fps=fps)
time_most_sig_change = np.argmax(np.abs(independent_samples_t)) / fps # E.G, FIND TIMEPOINT in SECONDS WHEN ANGLE CHANGE IS MOST STATISTICALLY SIGNIFICANT
print(time_most_sig_change)

# COMPUTE THE CORRELATION BETWEEN ANIMAL 1 AND ANIMAL 2 CENTROID MOVEMENT IN THE ENTIRE VIDEO USING PEARSONS R
animal_1_data = df['Movement_mouse_1_centroid'].values.astype(np.float32)
animal_2_data = df['Movement_mouse_2_centroid'].values.astype(np.float32)
pearsons_r = Statistics.pearsons_r(sample_1=animal_1_data, sample_2=animal_2_data)
print(pearsons_r)

# ... OR SPEARMANS RANK
spearmans = Statistics.spearman_rank_correlation(sample_1=animal_1_data, sample_2=animal_2_data)
print(spearmans)

# ... OR COMPUTE THE PEARSON AND SPEARMAN CORRELATION BETWEEN ANIMAL 1 AND ANIMAL 2 CENTROID MOVEMENT IN IN SLIDING 500MS WINDOWS.
pearsons_r = Statistics.sliding_pearsons_r(sample_1=animal_1_data, sample_2=animal_2_data, time_windows=np.array([0.5]), fps=fps)
spearman = Statistics.sliding_spearman_rank_correlation(sample_1=animal_1_data, sample_2=animal_2_data, time_windows=np.array([0.5]), fps=fps)
pearson = np.min(np.abs(pearsons_r[:, 0])) #GET THE PEARSON R WHEN THE CORRELATION IN 500MS WINDOWS ARE THE LOWEST
data = pd.DataFrame([pearsons_r.flatten(), spearman.flatten()]).T
data.columns = ['pearson', 'spearman']
data.plot(y=["pearson", "spearman"])
plt.show()

# PERFORM A CHI SQUARE TEST COMPARING TWO CLASSIFIER COLUMNS
pearsons_r = Statistics.sliding_pearsons_r(sample_1=animal_1_data, sample_2=animal_2_data, time_windows=np.array([0.5]), fps=fps)








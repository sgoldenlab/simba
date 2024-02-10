import argparse
import os.path
import pickle
from copy import deepcopy
from typing import Dict, Optional, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.statistics_mixin import Statistics
from simba.utils.read_write import read_df

CONFIG_PATH = "/Users/simon/Desktop/envs/simba/troubleshooting/RI/project_folder/project_config.ini"
VIDEO_NAME = "RI_01_179"


config = ConfigReader(config_path=CONFIG_PATH, create_logger=False)
data_path = os.path.join(config.features_dir, f"{VIDEO_NAME}.{config.file_type}")
df = read_df(file_path=data_path, file_type=config.file_type)
_, _, fps = config.read_video_info(video_name=VIDEO_NAME)

#
# correlation_coefficient = 0.8
#
# # Covariance matrix based on the correlation coefficient
# covariance_matrix = np.array([[1.0, correlation_coefficient],
#                               [correlation_coefficient, 1.0]])
#
#
# data = np.random.multivariate_normal(mean=[0, 0], cov=covariance_matrix, size=len(df))
# value1 = (data[:, 0] > 0).astype(int)
# value2 = (data[:, 1] > 0).astype(int)
#
# df['White_animal_lateral_threat'] = value1
# df['Black_animal_freezing'] = value2
#
# df.to_csv(data_path)

# COMPARE THE ANGLE DISTRIBUTION OF ONE ANIMAL IN ROLLING 500MS TIME WINDOWS USING INDEPENDENT SAMPLES T-TEST.
# USEFUL TO FIND INSTANCES WHEN ANIMAL RAPIDLY CHANGE ITS TAILBASE-CENTROID-NOSE THREE-POINT ANGLE
data = df["Mouse_1_angle"].values.astype(np.float64)
independent_samples_t = Statistics.rolling_independent_sample_t(
    data=data, time_window=0.5, fps=fps
)
time_most_sig_change = (
    np.argmax(np.abs(independent_samples_t)) / fps
)  # E.G, FIND TIMEPOINT in SECONDS WHEN ANGLE CHANGE IS MOST STATISTICALLY SIGNIFICANT
print(time_most_sig_change)

# COMPUTE THE CORRELATION BETWEEN ANIMAL 1 AND ANIMAL 2 CENTROID MOVEMENT IN THE ENTIRE VIDEO USING PEARSONS R
animal_1_data = df["Movement_mouse_1_centroid"].values.astype(np.float32)
animal_2_data = df["Movement_mouse_2_centroid"].values.astype(np.float32)
pearsons_r = Statistics.pearsons_r(sample_1=animal_1_data, sample_2=animal_2_data)
print(pearsons_r)

# ... OR SPEARMANS RANK
spearmans = Statistics.spearman_rank_correlation(
    sample_1=animal_1_data, sample_2=animal_2_data
)
print(spearmans)

# COMPUTE THE PEARSON AND SPEARMAN CORRELATION BETWEEN ANIMAL 1 AND ANIMAL 2 CENTROID MOVEMENT IN IN SLIDING 500MS WINDOWS.
pearsons_r = Statistics.sliding_pearsons_r(
    sample_1=animal_1_data,
    sample_2=animal_2_data,
    time_windows=np.array([0.5]),
    fps=fps,
)
spearman = Statistics.sliding_spearman_rank_correlation(
    sample_1=animal_1_data,
    sample_2=animal_2_data,
    time_windows=np.array([0.5]),
    fps=fps,
)
pearson = np.min(
    np.abs(pearsons_r[:, 0])
)  # GET THE PEARSON R WHEN THE CORRELATION IN 500MS WINDOWS ARE THE LOWEST
data = pd.DataFrame([pearsons_r.flatten(), spearman.flatten()]).T
data.columns = ["pearson", "spearman"]
data.plot(y=["pearson", "spearman"])
# plt.show()

# ... or the two sample KS test comparing the two distributions
results = Statistics.two_sample_ks(sample_1=animal_1_data, sample_2=animal_2_data)

# ... or one way ANOVA
critical_values = pickle.load(
    open(
        "/Users/simon/Desktop/envs/simba/simba/simba/assets/lookups/critical_values_05.pickle",
        "rb",
    )
)["f"]["one_tail"].values.astype(np.float32)
results = Statistics.one_way_anova(
    sample_1=animal_1_data, sample_2=animal_2_data, critical_values=critical_values
)


# Compares the feature values in current time-window to prior time-windows to find the length in time to the most recent time-window where a significantly different feature value distribution is detected.
# See this image as an example.
animal_data = df["Movement_mouse_1_centroid"].values.astype(np.float32)
critical_values = pickle.load(
    open(
        "/Users/simon/Desktop/envs/simba/simba/simba/assets/lookups/critical_values_05.pickle",
        "rb",
    )
)["independent_t_test"]["one_tail"].values.astype(np.float32)
results = Statistics.sliding_independent_samples_t(
    data=animal_data,
    time_window=0.5,
    slide_time=1.0,
    critical_values=critical_values,
    fps=fps,
)

# Compute the sliding auto-correlations (the correlation of the feature with itself using lagged windows).
white_animal = df["Movement_mouse_1_centroid"].values.astype(np.float32)
black_animal = df["Movement_mouse_2_centroid"].values.astype(np.float32)
autocor_white_animal = Statistics.sliding_autocorrelation(
    data=white_animal, max_lag=0.5, time_window=1.0, fps=fps
)
autocor_black_animal = Statistics.sliding_autocorrelation(
    data=black_animal, max_lag=0.5, time_window=1.0, fps=fps
)

# Calculate sliding Z-scores for a given data array over specified time windows.
z_scores = Statistics.sliding_z_scores(
    data=white_animal, time_windows=np.array([0.5, 1.5]), fps=fps
)

# Next we look at the classifications. And compute the chi square correlation coefficent
lateral_threat = df[["White_animal_lateral_threat"]].values.flatten().astype(np.float32)
freezing = df[["Black_animal_freezing"]].values.flatten().astype(np.float32)
chi_square = Statistics.chi_square(sample_1=lateral_threat, sample_2=freezing)[0]

# ... or yule coefficient
yule = Statistics.yule_coef(
    x=lateral_threat.astype(np.int8), y=freezing.astype(np.int8)
)
print(yule)

# ... sokal sneath, using the pose-estimation probability as weights
p_cols = [x for x in df.columns if x.endswith("_p")]
p_vals = np.array(df[p_cols].mean(axis=1))
sokal_sneath = Statistics.sokal_sneath(
    x=lateral_threat.astype(np.int8), y=freezing.astype(np.int8), w=p_vals
)

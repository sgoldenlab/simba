import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from scipy.signal import find_peaks


def rolling_frequentist_distribution_tests(data: np.array,
                                           feature_name: str,
                                           fps: int):
    """
    Helper to compare feature value distributions in 1s sequential time-bins: Kolmogorov-Smirnov and T-tests
    Compares the feature values against a normal distribution: Lillefors, Shapiro.
    Find the number of peaks in *rolling* 1s long feature window.
    """
    ks_results, = np.full((data.shape[0]), -1.0),
    t_test_results = np.full((data.shape[0]), -1.0)
    lillefors_results = np.full((data.shape[0]), -1.0)
    shapiro_results = np.full((data.shape[0]), -1.0)
    peak_cnt_results = np.full((data.shape[0]), -1.0)

    for i in range(fps, data.shape[0] - fps, fps):
        bin_1_idx, bin_2_idx = [i-fps, i], [i, i+fps]
        bin_1_data, bin_2_data = data[bin_1_idx[0]:bin_1_idx[1]], data[bin_2_idx[0]:bin_2_idx[1]]
        ks_results[i:i+fps+1] = stats.ks_2samp(data1=bin_1_data, data2=bin_2_data).statistic
        t_test_results[i:i+fps+1] = stats.ttest_ind(bin_1_data, bin_2_data).statistic

    for i in range(0, data.shape[0]-fps, fps):
        lillefors_results[i:i+fps+1] = lilliefors(data[i:i+fps])[0]
        shapiro_results[i:i+fps+1] = stats.shapiro(data[i:i+fps])[0]

    rolling_idx = np.arange(fps)[None, :] + 1*np.arange(data.shape[0])[:, None]
    for i in range(rolling_idx.shape[0]):
        bin_start_idx, bin_end_idx = rolling_idx[i][0], rolling_idx[i][-1]
        peaks, _ = find_peaks(data[bin_start_idx:bin_end_idx], height=0)
        peak_cnt_results[i] = len(peaks)

    columns = [f'{feature_name}_KS', f'{feature_name}_TTEST', f'{feature_name}_LILLEFORS', f'{feature_name}_SHAPIRO', f'{feature_name}_PEAK_CNT']
    return pd.DataFrame(np.column_stack((ks_results, t_test_results,lillefors_results, shapiro_results, peak_cnt_results)), columns=columns).round(4)


# features = [f'Convex_hull_mean_25_window']
# DATA_PATH = '/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/csv/features_extracted/SF2.csv'
# data = pd.read_csv(DATA_PATH, index_col=0)
# for feature in features:
#     df = rolling_frequentist_distribution_tests(data=data[feature].values,
#                                                 feature_name=feature, fps=25)
#     df.to_csv('Test.csv')



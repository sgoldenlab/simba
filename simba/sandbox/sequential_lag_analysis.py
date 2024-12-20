import numpy as np
import pandas as pd

from simba.utils.data import detect_bouts
from simba.utils.read_write import read_df
from simba.utils.checks import check_instance, check_str, check_that_column_exist, check_float
from simba.utils.errors import CountError

from statsmodels.sandbox.stats.runs import runstest_1samp

def sequential_lag_analysis(data: pd.DataFrame, criterion: str, target:str, time_window: float, fps: float):
    """
    Perform sequential lag analysis to determine the temporal relationship between two events.

    For every onset of behavior C, count the proportions of behavior T onsets in the time-window preceding the onset
    of behavior C vs the proportion of behavior T onsets in the time-window proceeding the onset of behavior C.

    A value closer to 1.0 indicates that behavior T
    always precede behavior C. A value closer to 0.0 indicates that behavior T follows behavior C. A value of -1.0 indicates
    that behavior T never precede nor proceed behavior C.

    :example:
    >>> df = read_df(file_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/targets_inserted/Together_1.csv', file_type='csv')
    >>> sequential_lag_analysis(data=df, criterion='Attack', target='Sniffing', fps=5, time_window=2.0)

    References
    ----------
    .. [1] Casarrubea et al., Structural analyses in the study of behavior: From rodents to non-human primates, `Frontiers in Psychology`,
           2022.
    """

    check_instance(source=sequential_lag_analysis.__name__, instance=data, accepted_types=(pd.DataFrame))
    check_str(name=f'{sequential_lag_analysis.__name__} criterion', value=criterion)
    check_str(name=f'{sequential_lag_analysis.__name__} target', value=target)
    check_float(name=f'{sequential_lag_analysis.__name__} fps', value=fps, min_value=1.0)
    check_float(name=f'{sequential_lag_analysis.__name__} time-window', value=time_window, min_value=0.01)
    check_that_column_exist(df=data, column_name=[criterion, target], file_name=sequential_lag_analysis.__name__)
    bouts = detect_bouts(data_df=data, target_lst=[criterion, target], fps=fps)
    if len(bouts) == 0:
        raise CountError(msg=f'No events of behaviors {criterion} and {target} detected in data.', source=sequential_lag_analysis)
    criterion_starts = bouts['Start_frame'][bouts['Event'] == criterion].values
    target_starts = bouts['Start_frame'][bouts['Event'] == target].values
    preceding_cnt, proceeding_cnt = 0, 0
    window = int(fps * time_window)
    if window < 1.0: window = 1
    for criterion_start in criterion_starts:
        preceeding_events = target_starts[np.argwhere((target_starts < criterion_start) & (target_starts >= (criterion_start - window)))].flatten()
        preceding_cnt += preceeding_events.shape[0]
        target_starts = np.array([x for x in target_starts if x not in preceeding_events])
        proceeding_events = target_starts[np.argwhere((target_starts > criterion_start) & (target_starts <= (criterion_start + window)))].flatten()
        proceeding_cnt += proceeding_events.shape[0]
        target_starts = np.array([x for x in target_starts if x not in proceeding_events])
    if preceding_cnt == 0 and proceeding_cnt == 0:
        return -1.0
    elif preceding_cnt == 0:
        return 0.0
    elif proceeding_cnt == 0:
        return 1.0
    else:
        return np.round(preceding_cnt / (preceding_cnt + proceeding_cnt), 3)








df = read_df(file_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/targets_inserted/Together_1.csv', file_type='csv')

sequential_lag_analysis(data=df, criterion='Attack', target='Sniffing', fps=5, time_window=2.0)







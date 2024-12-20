import time

import pandas as pd
import numpy as np
from simba.utils.checks import check_valid_dataframe,  check_float, check_int, check_instance, check_if_df_field_is_boolean
from simba.utils.data import detect_bouts

def plug_holes_shortest_bout(data_df: pd.DataFrame,
                            clf_name: str,
                            fps: float,
                            shortest_bout: int) -> pd.DataFrame:

    """
    Removes behavior "bouts" that are shorter than the minimum user-specified length within a dataframe.

    .. note::
      In the initial step the function looks for behavior "interuptions" that are the length of the ``shortest_bout`` or shorter.
      I.e., these are ``0`` sequences that are the length of the ``shortest_bout`` or shorter with trailing **and** leading `1`s.
      These interuptions are filled with `1`s. Next, the behavioral bouts shorter than the `shortest_bout` are removed. This operations are perfomed as it helps in preserving longer sequences of the desired behavior,
      ensuring they aren't fragmented by brief interruptions.

    :param pd.DataFrame data_df: Pandas Dataframe with classifier prediction data.
    :param str clf_name: Name of the classifier field of list of names of classifier fields
    :param int fps: The fps of the input video.
    :param int shortest_bout: The shortest valid behavior boat in milliseconds.
    :return pd.DataFrame data_df: Dataframe where behavior bouts with invalid lengths have been removed (< shortest_bout)

    :example:
    >>>  data_df = pd.DataFrame(data=[1, 0, 1, 1, 1], columns=['target'])
    >>>  plug_holes_shortest_bout(data_df=data_df, clf_name='target', fps=10, shortest_bout=2000)
    >>>         target
    >>>    0       1
    >>>    1       1
    >>>    2       1
    >>>    3       1
    >>>    4       1
    """

    print(data_df)
    check_int(name=f'{plug_holes_shortest_bout.__name__} shortest_bout', value=shortest_bout, min_value=0)
    check_float(name=f'{plug_holes_shortest_bout.__name__} fps', value=fps, min_value=10e-6)
    shortest_bout_frms, shortest_bout_s = int(fps * (shortest_bout / 1000)), (shortest_bout / 1000)
    if shortest_bout_frms <= 1:
        return data_df
    check_instance(source=plug_holes_shortest_bout.__name__, instance=clf_name, accepted_types=(str,))
    check_valid_dataframe(df=data_df, source=f'{plug_holes_shortest_bout.__name__} data_df', required_fields=[clf_name])
    check_if_df_field_is_boolean(df=data_df, field=clf_name, bool_values=(0, 1))

    data = data_df[clf_name].to_frame()
    data[f'{clf_name}_inverted'] = data[clf_name].apply(lambda x: ~x + 2)
    clf_inverted_bouts = detect_bouts(data_df=data, target_lst=[f'{clf_name}_inverted'], fps=fps)
    clf_inverted_bouts = clf_inverted_bouts[clf_inverted_bouts['Bout_time'] < shortest_bout_s]
    if len(clf_inverted_bouts) > 0:
        below_min_inverted = []
        for i, j in zip(clf_inverted_bouts['Start_frame'].values, clf_inverted_bouts['End_frame'].values):
            below_min_inverted.extend(np.arange(i, j+1))
        data.loc[below_min_inverted, clf_name] = 1
        data_df[clf_name] = data[clf_name]

    clf_bouts = detect_bouts(data_df=data_df, target_lst=[clf_name], fps=fps)
    below_min_bouts = clf_bouts[clf_bouts['Bout_time'] <= shortest_bout_s]
    if len(below_min_bouts) == 0:
        return data_df

    result_clf, below_min_frms = data_df[clf_name].values, []
    for i, j in zip(below_min_bouts['Start_frame'].values, below_min_bouts['End_frame'].values):
        below_min_frms.extend(np.arange(i, j+1))
    result_clf[below_min_frms] = 0
    data_df[clf_name] = result_clf
    return data_df

def plug_holes_shortest_bout_old(data_df: pd.DataFrame, clf_name: str, fps: int, shortest_bout: int) -> pd.DataFrame:
    """
    Removes behavior "bouts" that are shorter than the minimum user-specified length within a dataframe.

    :param pd.DataFrame data_df: Pandas Dataframe with classifier prediction data.
    :param str clf_name: Name of the classifier field.
    :param int fps: The fps of the input video.
    :param int shortest_bout: The shortest valid behavior boat in milliseconds.
    :return pd.DataFrame data_df: Dataframe where behavior bouts with invalid lengths have been removed (< shortest_bout)

    :example:
    >>>  data_df = pd.DataFrame(data=[1, 0, 1, 1, 1], columns=['target'])
    >>>  plug_holes_shortest_bout(data_df=data_df, clf_name='target', fps=10, shortest_bout=2000)
    >>>         target
    >>>    0       1
    >>>    1       1
    >>>    2       1
    >>>    3       1
    >>>    4       1
    """

    frames_to_plug = int(int(fps) * int(shortest_bout) / 1000)
    frames_to_plug_lst = list(range(1, frames_to_plug + 1))
    frames_to_plug_lst.reverse()
    patternListofLists, negPatternListofList = [], []
    for k in frames_to_plug_lst:
        zerosInList, oneInlist = [0] * k, [1] * k
        currList = [1]
        currList.extend(zerosInList)
        currList.extend([1])
        currListNeg = [0]
        currListNeg.extend(oneInlist)
        currListNeg.extend([0])
        patternListofLists.append(currList)
        negPatternListofList.append(currListNeg)
    fill_patterns = np.asarray(patternListofLists)
    remove_patterns = np.asarray(negPatternListofList)

    for currPattern in fill_patterns:
        n_obs = len(currPattern)
        data_df["rolling_match"] = (
            data_df[clf_name]
            .rolling(window=n_obs, min_periods=n_obs)
            .apply(lambda x: (x == currPattern).all())
            .mask(lambda x: x == 0)
            .bfill(limit=n_obs - 1)
            .fillna(0)
            .astype(bool)
        )
        data_df.loc[data_df["rolling_match"] == True, clf_name] = 1
        data_df = data_df.drop(["rolling_match"], axis=1)

    print(remove_patterns)
    for currPattern in remove_patterns:
        n_obs = len(currPattern)
        data_df["rolling_match"] = (
            data_df[clf_name]
            .rolling(window=n_obs, min_periods=n_obs)
            .apply(lambda x: (x == currPattern).all())
            .mask(lambda x: x == 0)
            .bfill(limit=n_obs - 1)
            .fillna(0)
            .astype(bool)
        )
        data_df.loc[data_df["rolling_match"] == True, clf_name] = 0
        data_df = data_df.drop(["rolling_match"], axis=1)

    return data_df


# d = pd.DataFrame(data=[1, 0, 1, 1, 1], columns=['target'])
# results_old = plug_holes_shortest_bout_old(data_df=d, clf_name='target', fps=10, shortest_bout=2000)
#print(results_old)
#results_new = plug_holes_shortest_bout(data_df=d, clf_name='target', fps=10, shortest_bout=2000)



#pd.testing.assert_frame_equal(results, pd.DataFrame(data=[1, 1, 1, 1, 1], columns=['target']))


#     pd.testing.assert_frame_equal(results, pd.DataFrame(data=[1, 1, 1, 1, 1], columns=['target']))
#
#
#
# data = pd.read_csv()
#
# data = np.random.randint(0, 2, (100000,))
# data_df = pd.DataFrame(data=data, columns=['target'])
#
# start = time.time()
# df_1 = plug_holes_shortest_bout(data_df=data_df, clf_name='target', fps=16, shortest_bout=10000)
# new_time = time.time() - start
# start = time.time()
# df_2 = plug_holes_shortest_bout_old(data_df=data_df, clf_name='target', fps=16, shortest_bout=10000)
# old_time = time.time() - start
#
# out = pd.DataFrame()
# out['original'] = data
# out['new'] = df_1['target']
# out['old'] = df_2['target']
# out['diff'] = out['new'] - out['old']
#
#
# out.to_csv('test.csv')
#
#
#
# out['diff'].sum()

#assert data['new'] == data['old']
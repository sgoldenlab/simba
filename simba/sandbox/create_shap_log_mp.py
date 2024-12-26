import functools
import multiprocessing
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from simba.mixins.train_model_mixin import TrainModelMixin
from simba.plotting.shap_agg_stats_visualizer import \
    ShapAggregateStatisticsCalculator
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_instance, check_int,
                                check_str, check_valid_array,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_lst)
from simba.utils.enums import Defaults, Formats
from simba.utils.errors import NoDataError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import find_core_cnt
from simba.utils.warnings import NotEnoughDataWarning, ShapWarning


def _create_shap_mp_helper(data: Tuple[int, pd.DataFrame],
                           explainer: shap.TreeExplainer,
                           clf_name: str,
                           verbose: bool) -> Tuple[int, pd.DataFrame]:

    if verbose:
        print(f'Processing SHAP batch {data[0]+1}... ({len(data[1])} observations)')
    _ = data[1].pop(clf_name).values.reshape(-1, 1)
    shap_results = explainer.shap_values(data[1].values, check_additivity=False)[1]
    return shap_results, data[0]


def create_shap_log(rf_clf: RandomForestClassifier,
                    x: Union[pd.DataFrame, np.ndarray],
                    y: Union[pd.DataFrame, pd.Series, np.ndarray],
                    x_names: List[str],
                    clf_name: str,
                    cnt_present: int,
                    cnt_absent: int,
                    verbose: bool = True,
                    plot: bool = True,
                    save_it: Optional[int] = 100,
                    save_dir: Optional[Union[str, os.PathLike]] = None,
                    save_file_suffix: Optional[int] = None) -> Union[None, Tuple[pd.DataFrame]]:

    """
    Compute SHAP values for a random forest classifier.
    This method computes SHAP (SHapley Additive exPlanations) values for a given random forest classifier.
    The SHAP value for feature 'i' in the context of a prediction 'f' and input 'x' is calculated using the following formula:

    .. math::
       \phi_i(f, x) = \\sum_{S \\subseteq F \\setminus {i}} \\frac{|S|!(|F| - |S| - 1)!}{|F|!} (f_{S \cup {i}}(x_{S \\cup {i}}) - f_S(x_S))

    .. note::
       `Documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#train-predictive-classifiers-settings>`_
       Uses TreeSHAP `Documentation <https://shap.readthedocs.io/en/latest/index.html>`_

    .. image:: _static/img/shap.png
       :width: 600
       :align: center

    .. seealso::
       For multicore solution, see :func:`~simba.mixins.train_model_mixins.TrainModelMixin.create_shap_log_mp`
       For GPU method, see :func:`~simba.data_processors.cuda.create_shap_log.create_shap_log`

    :param RandomForestClassifier rf_clf: sklearn random forest classifier
    :param Union[pd.DataFrame, np.ndarray] x: Test features.
    :param Union[pd.DataFrame, pd.Series, np.ndarray] y: Test target.
    :param List[str] x_names: Feature names.
    :param str clf_name: Classifier name.
    :param int cnt_present: Number of behavior-present frames to calculate SHAP values for.
    :param int cnt_absent: Number of behavior-absent frames to calculate SHAP values for.
    :param int save_it: Save iteration cadence. If None, then only saves at completion.
    :param str save_dir: Optional directory where to save output in csv file format. If None, the data is returned.
    :param Optional[int] save_file_suffix: If integer, represents the count of the classifier within a grid search. If none, the classifier is not part of a grid search.

    :example:
    >>> from simba.mixins.train_model_mixin import TrainModelMixin
    >>> x_cols = list(pd.read_csv('/Users/simon/Desktop/envs/simba/simba/tests/data/sample_data/shap_test.csv', index_col=0).columns)
    >>> x = pd.DataFrame(np.random.randint(0, 500, (9000, len(x_cols))), columns=x_cols)
    >>> y = pd.Series(np.random.randint(0, 2, (9000,)))
    >>> rf_clf = TrainModelMixin().clf_define(n_estimators=100)
    >>> rf_clf = TrainModelMixin().clf_fit(clf=rf_clf, x_df=x, y_df=y)
    >>> feature_names = [str(x) for x in list(x.columns)]
    >>> create_shap_log(rf_clf=rf_clf, x=x, y=y, x_names=feature_names, clf_name='test', save_it=10, cnt_present=50, cnt_absent=50, plot=True, save_dir=r'/Users/simon/Desktop/feltz')
    """

    print("Calculating SHAP values (SINGLE CORE)...")
    timer = SimbaTimer(start=True)
    check_instance(source='create_shap_log', instance=rf_clf, accepted_types=(RandomForestClassifier,))
    check_instance(source=f'{create_shap_log.__name__} x', instance=x, accepted_types=(np.ndarray, pd.DataFrame))
    check_instance(source=f'{create_shap_log.__name__} y', instance=y, accepted_types=(np.ndarray, pd.Series, pd.DataFrame))
    if isinstance(x, pd.DataFrame):
        check_valid_dataframe(df=x, source=f'{create_shap_log.__name__} x', valid_dtypes=Formats.NUMERIC_DTYPES.value)
        x = x.values
    else:
        check_valid_array(data=x, source=f'{create_shap_log.__name__} x', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    if isinstance(y, pd.DataFrame):
        check_valid_dataframe(df=y, source=f'{create_shap_log.__name__} y', valid_dtypes=Formats.NUMERIC_DTYPES.value, max_axis_1=1)
        y = y.values
    else:
        if isinstance(y, pd.Series):
            y = y.values
        check_valid_array(data=y, source=f'{create_shap_log.__name__} y', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_lst(data=x_names, source=f'{create_shap_log.__name__} x_names', valid_dtypes=(str,), exact_len=x.shape[1])
    check_str(name=f'{create_shap_log.__name__} clf_name', value=clf_name)
    check_int(name=f'{create_shap_log.__name__} cnt_present', value=cnt_present, min_value=1)
    check_int(name=f'{create_shap_log.__name__} cnt_absent', value=cnt_absent, min_value=1)
    check_instance(source=f'{create_shap_log.__name__} save_it', instance=save_it, accepted_types=(type(None), int))
    if save_it is not None and save_dir is None:
        ShapWarning(msg='Omitting save_it as save_dir is None')
    if save_it is not None:
        check_int(name=f'{create_shap_log.__name__} save_it', value=save_it, min_value=1)
    if save_it is None or save_it > x.shape[0]:
        save_it = x.shape[0]
    if save_file_suffix is not None:
        check_int(name=f'{create_shap_log.__name__} save_it', value=save_it, min_value=0)
    check_valid_lst(data=list(x_names), valid_dtypes=(str,), exact_len=x.shape[1])
    check_valid_boolean(value=[verbose, plot], source=f'{create_shap_log.__name__} verbose, plot')
    df = pd.DataFrame(np.hstack((x, y.reshape(-1, 1))), columns=x_names + [clf_name])
    del x; del y
    present_df, absent_df = df[df[clf_name] == 1], df[df[clf_name] == 0]
    if len(present_df) == 0:
        raise NoDataError(msg=f'Cannot calculate SHAP values: no target PRESENT annotations detected.', source=create_shap_log.__name__)
    elif len(absent_df) == 0:
        raise NoDataError(msg=f'Cannot calculate SHAP values: no target ABSENT annotations detected.', source=create_shap_log.__name__)
    if len(present_df) < cnt_present:
        NotEnoughDataWarning(msg=f"Train data contains {len(present_df)} behavior-present annotations. This is less the number of frames you specified to calculate shap values for ({str(cnt_present)}). SimBA will calculate shap scores for the {len(present_df)} behavior-present frames available", source=create_shap_log.__name__)
        cnt_present = len(present_df)
    if len(absent_df) < cnt_absent:
        NotEnoughDataWarning (msg=f"Train data contains {len(absent_df)} behavior-absent annotations. This is less the number of frames you specified to calculate shap values for ({str(cnt_absent)}). SimBA will calculate shap scores for the {len(absent_df)} behavior-absent frames available", source=create_shap_log.__name__ ,)
        cnt_absent = len(absent_df)
    out_shap_path, out_raw_path, img_save_path, df_save_paths, summary_dfs, img = None, None, None, None, None, None
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
        if save_file_suffix is not None:
            check_int(name=f'{create_shap_log.__name__} save_file_no', value=save_file_suffix, min_value=0)
            out_shap_path = os.path.join(save_dir, f"SHAP_values_{save_file_suffix}_{clf_name}.csv")
            out_raw_path = os.path.join(save_dir, f"RAW_SHAP_feature_values_{save_file_suffix}_{clf_name}.csv")
            df_save_paths = {'PRESENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_PRESENT_{save_file_suffix}.csv"), 'ABSENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_ABSENT_{save_file_suffix}.csv")}
            img_save_path = os.path.join(save_dir, f"SHAP_summary_line_graph_{clf_name}_{save_file_suffix}.png")
        else:
            out_shap_path = os.path.join(save_dir, f"SHAP_values_{clf_name}.csv")
            out_raw_path = os.path.join(save_dir, f"RAW_SHAP_feature_values_{clf_name}.csv")
            df_save_paths = {'PRESENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_PRESENT.csv"), 'ABSENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_ABSENT.csv")}
            img_save_path = os.path.join(save_dir, f"SHAP_summary_line_graph_{clf_name}.png")
    shap_x = pd.concat([present_df.sample(cnt_present, replace=False), absent_df.sample(cnt_absent, replace=False)], axis=0).reset_index(drop=True)
    shap_y = shap_x[clf_name].values.flatten()
    shap_x = shap_x.drop([clf_name], axis=1)
    explainer = TrainModelMixin().define_tree_explainer(clf=rf_clf)
    expected_value = explainer.expected_value[1]
    raw_df = pd.DataFrame(columns=x_names)
    shap_headers = list(x_names) + ["Expected_value", "Sum", "Prediction_probability", clf_name]
    shap_df = pd.DataFrame(columns=shap_headers)
    for cnt, frame in enumerate(range(len(shap_x))):
        shap_frm_timer = SimbaTimer(start=True)
        frame_data = shap_x.iloc[[frame]]
        frame_shap = explainer.shap_values(frame_data, check_additivity=False)[1][0].tolist()
        frame_shap.extend((expected_value, sum(frame_shap), rf_clf.predict_proba(frame_data)[0][1], shap_y[cnt]))
        raw_df.loc[len(raw_df)] = list(shap_x.iloc[frame])
        shap_df.loc[len(shap_df)] = frame_shap
        if ((cnt % save_it == 0) or (cnt == len(shap_x) - 1) and (cnt != 0) and (save_dir is not None)):
            print(f"Saving SHAP data after {cnt} iterations...")
            shap_df.to_csv(out_shap_path)
            raw_df.to_csv(out_raw_path)
        shap_frm_timer.stop_timer()
        print(f"SHAP frame: {cnt + 1} / {len(shap_x)}, elapsed time: {shap_frm_timer.elapsed_time_str}...")
    if plot:
        shap_computer = ShapAggregateStatisticsCalculator(classifier_name=clf_name,
                                                          shap_df=shap_df,
                                                          shap_baseline_value=int(expected_value * 100),
                                                          save_dir=None)
        summary_dfs, img = shap_computer.run()
        if save_dir is not None:
            summary_dfs['PRESENT'].to_csv(df_save_paths['PRESENT'])
            summary_dfs['ABSENT'].to_csv(df_save_paths['ABSENT'])
            cv2.imwrite(img_save_path, img)

    timer.stop_timer()
    if save_dir is not None and verbose:
        shap_df.to_csv(out_shap_path)
        raw_df.to_csv(out_raw_path)
        stdout_success(msg=f"SHAP calculations complete! Results saved at {out_shap_path} and {out_raw_path}", elapsed_time=timer.elapsed_time_str, source=create_shap_log.__name__)

    if not save_dir:
        return shap_df, raw_df, summary_dfs, img

def create_shap_log_mp(rf_clf: RandomForestClassifier,
                       x: Union[pd.DataFrame, np.ndarray],
                       y: Union[pd.DataFrame, pd.Series, np.ndarray],
                       x_names: List[str],
                       clf_name: str,
                       cnt_present: int,
                       cnt_absent: int,
                       core_cnt: int = -1,
                       chunk_size: int = 100,
                       verbose: bool = True,
                       save_dir: Optional[Union[str, os.PathLike]] = None,
                       save_file_suffix: Optional[int] = None,
                       plot: bool = False) -> Union[None, Tuple[pd.DataFrame]]:
    """
    Compute SHAP values using multiprocessing.

    .. seealso::
       `Documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#train-predictive-classifiers-settings>`_
        For single-core solution, see :func:`~simba.mixins.train_model_mixins.TrainModelMixin.create_shap_log`
        For GPU method, see :func:`~simba.data_processors.cuda.create_shap_log.create_shap_log`

    .. image:: _static/img/shap.png
       :width: 400
       :align: center

    :param RandomForestClassifier rf_clf: Fitted sklearn random forest classifier
    :param Union[pd.DataFrame, np.ndarray] x: Test features.
    :param Union[pd.DataFrame, pd.Series, np.ndarray] y_df: Test target.
    :param List[str] x_names: Feature names.
    :param str clf_name: Classifier name.
    :param int cnt_present: Number of behavior-present frames to calculate SHAP values for.
    :param int cnt_absent: Number of behavior-absent frames to calculate SHAP values for.
    :param int chunk_size: How many observations to process in each chunk. Increase value for faster processing if your memory allows.
    :param bool verbose: If True, prints progress.
    :param Optional[Union[str, os.PathLike]] save_dir: Optional directory where to store the results. If None, then the results are returned.
    :param Optional[int] save_file_suffix: Optional suffix to add to the shap output filenames. Useful for gridsearches and multiple shap data output files are to-be stored in the same `save_dir`.
    :param bool plot: If True, create SHAP aggregation and plots.

    :example:
    >>> from simba.mixins.train_model_mixin import TrainModelMixin
    >>> x_cols = list(pd.read_csv('/Users/simon/Desktop/envs/simba/simba/tests/data/sample_data/shap_test.csv', index_col=0).columns)
    >>> x = pd.DataFrame(np.random.randint(0, 500, (9000, len(x_cols))), columns=x_cols)
    >>> y = pd.Series(np.random.randint(0, 2, (9000,)))
    """

    timer = SimbaTimer(start=True)
    check_instance(source=f'{create_shap_log_mp.__name__} rf_clf', instance=rf_clf, accepted_types=(RandomForestClassifier,))
    check_instance(source=f'{create_shap_log_mp.__name__} x', instance=x, accepted_types=(np.ndarray, pd.DataFrame))
    check_instance(source=f'{create_shap_log_mp.__name__} y', instance=y, accepted_types=(np.ndarray, pd.Series, pd.DataFrame))
    if isinstance(x, pd.DataFrame):
        check_valid_dataframe(df=x, source=f'{create_shap_log_mp.__name__} x', valid_dtypes=Formats.NUMERIC_DTYPES.value)
        x = x.values
    else:
        check_valid_array(data=x, source=f'{create_shap_log_mp.__name__} x', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    if isinstance(y, pd.DataFrame):
        check_valid_dataframe(df=y, source=f'{create_shap_log_mp.__name__} y', valid_dtypes=Formats.NUMERIC_DTYPES.value, max_axis_1=1)
        y = y.values
    else:
        if isinstance(y, pd.Series):
            y = y.values
        check_valid_array(data=y, source=f'{create_shap_log_mp.__name__} y', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_lst(data=x_names, source=f'{create_shap_log_mp.__name__} x_names', valid_dtypes=(str,), exact_len=x.shape[1])
    check_str(name=f'{create_shap_log_mp.__name__} clf_name', value=clf_name)
    check_int(name=f'{create_shap_log_mp.__name__} cnt_present', value=cnt_present, min_value=1)
    check_int(name=f'{create_shap_log_mp.__name__} cnt_absent', value=cnt_absent, min_value=1)
    check_int(name=f'{create_shap_log_mp.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
    check_int(name=f'{create_shap_log_mp.__name__} chunk_size', value=chunk_size, min_value=1)
    check_valid_boolean(value=[verbose, plot], source=f'{create_shap_log_mp.__name__} verbose, plot')
    core_cnt = [find_core_cnt()[0] if core_cnt is -1 or core_cnt > find_core_cnt()[0] else core_cnt][0]
    df = pd.DataFrame(np.hstack((x, y.reshape(-1, 1))), columns=feature_names + [clf_name])
    del x; del y
    present_df, absent_df = df[df[clf_name] == 1], df[df[clf_name] == 0]
    if len(present_df) == 0:
        raise NoDataError(msg=f'Cannot calculate SHAP values: no target PRESENT annotations detected.', source=create_shap_log_mp.__name__)
    elif len(absent_df) == 0:
        raise NoDataError(msg=f'Cannot calculate SHAP values: no target ABSENT annotations detected.', source=create_shap_log_mp.__name__)
    if len(present_df) < cnt_present:
        NotEnoughDataWarning(msg=f"Train data contains {len(present_df)} behavior-present annotations. This is less the number of frames you specified to calculate shap values for ({str(cnt_present)}). SimBA will calculate shap scores for the {len(present_df)} behavior-present frames available", source=self.__class__.__name__)
        cnt_present = len(present_df)
    if len(absent_df) < cnt_absent:
        NotEnoughDataWarning (msg=f"Train data contains {len(absent_df)} behavior-absent annotations. This is less the number of frames you specified to calculate shap values for ({str(cnt_absent)}). SimBA will calculate shap scores for the {len(absent_df)} behavior-absent frames available", source=self.__class__.__name__ ,)
        cnt_absent = len(absent_df)
    shap_data = pd.concat([present_df.sample(cnt_present, replace=False), absent_df.sample(cnt_absent, replace=False)], axis=0).reset_index(drop=True)
    batch_cnt = max(1, int(np.ceil(len(shap_data) / chunk_size)))
    shap_data = np.array_split(shap_data, batch_cnt)
    shap_data = [(x, y) for x, y in enumerate(shap_data)]
    explainer = TrainModelMixin().define_tree_explainer(clf=rf_clf)
    expected_value = explainer.expected_value[1]
    shap_results, shap_raw = [], []
    print(f"Computing {cnt_present + cnt_absent} SHAP values. Follow progress in OS terminal... (CORES: {core_cnt}, CHUNK SIZE: {chunk_size})")
    with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.MAXIMUM_MAX_TASK_PER_CHILD.value) as pool:
        constants = functools.partial(_create_shap_mp_helper, explainer=explainer, clf_name=clf_name, verbose=verbose)
        for cnt, result in enumerate(pool.imap_unordered(constants, shap_data, chunksize=1)):
            proba = TrainModelMixin().clf_predict_proba(clf=rf_clf, x_df=shap_data[result[1]][1].drop(clf_name, axis=1), model_name=clf_name).reshape(-1, 1)
            shap_sum = np.sum(result[0], axis=1).reshape(-1, 1)
            batch_shap_results = np.hstack((result[0], np.full((result[0].shape[0]), expected_value).reshape(-1, 1), shap_sum, proba, shap_data[result[1]][1][clf_name].values.reshape(-1, 1))).astype(np.float32)
            shap_results.append(batch_shap_results)
            shap_raw.append(shap_data[result[1]][1].drop(clf_name, axis=1))
            if verbose:
                print(f"Completed SHAP data (Batch {result[1] + 1}/{len(shap_data)}).")

    pool.terminate(); pool.join()
    shap_df = pd.DataFrame(data=np.row_stack(shap_results), columns=list(x_names) + ["Expected_value", "Sum", "Prediction_probability", clf_name])
    raw_df = pd.DataFrame(data=np.row_stack(shap_raw), columns=list(x_names))
    out_shap_path, out_raw_path, img_save_path, df_save_paths, summary_dfs, img = None, None, None, None, None, None
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
        if save_file_suffix is not None:
            check_int(name=f'{create_shap_log_mp.__name__} save_file_no', value=save_file_suffix, min_value=0)
            out_shap_path = os.path.join(save_dir, f"SHAP_values_{save_file_suffix}_{clf_name}.csv")
            out_raw_path = os.path.join(save_dir, f"RAW_SHAP_feature_values_{save_file_suffix}_{clf_name}.csv")
            df_save_paths = {'PRESENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_PRESENT_{save_file_suffix}.csv"), 'ABSENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_ABSENT_{save_file_suffix}.csv")}
            img_save_path = os.path.join(save_dir, f"SHAP_summary_line_graph_{clf_name}_{save_file_suffix}.png")
        else:
            out_shap_path = os.path.join(save_dir, f"SHAP_values_{clf_name}.csv")
            out_raw_path = os.path.join(save_dir, f"RAW_SHAP_feature_values_{clf_name}.csv")
            df_save_paths = {'PRESENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_PRESENT.csv"), 'ABSENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_ABSENT.csv")}
            img_save_path = os.path.join(save_dir, f"SHAP_summary_line_graph_{clf_name}.png")
        shap_df.to_csv(out_shap_path); raw_df.to_csv(out_raw_path)
    if plot:
        shap_computer = ShapAggregateStatisticsCalculator(classifier_name=clf_name,
                                                          shap_df=shap_df,
                                                          shap_baseline_value=int(expected_value * 100),
                                                          save_dir=None)
        summary_dfs, img = shap_computer.run()
        if save_dir is not None:
            summary_dfs['PRESENT'].to_csv(df_save_paths['PRESENT'])
            summary_dfs['ABSENT'].to_csv(df_save_paths['ABSENT'])
            cv2.imwrite(img_save_path, img)

    timer.stop_timer()
    if save_dir and verbose:
        stdout_success(msg=f'SHAP data saved in {save_dir}', source=create_shap_log_mp.__name__, elapsed_time=timer.elapsed_time_str)
    if not save_dir:
        return shap_df, raw_df, summary_dfs, img

# from simba.mixins.train_model_mixin import TrainModelMixin
# x_cols = list(pd.read_csv('/Users/simon/Desktop/envs/simba/simba/tests/data/sample_data/shap_test.csv', index_col=0).columns)
# x = pd.DataFrame(np.random.randint(0, 500, (9000, len(x_cols))), columns=x_cols)
# y = pd.Series(np.random.randint(0, 2, (9000,)))
# rf_clf = TrainModelMixin().clf_define(n_estimators=100)
# rf_clf = TrainModelMixin().clf_fit(clf=rf_clf, x_df=x, y_df=y)
# feature_names = [str(x) for x in list(x.columns)]
# create_shap_log(rf_clf=rf_clf,
#                 x=x,
#                 y=y,
#                 x_names=feature_names,
#                 clf_name='test',
#                 save_it=10,
#                 cnt_present=50,
#                 cnt_absent=50,
#                 plot=True,
#                 save_dir=r'/Users/simon/Desktop/feltz')



#
#
#
#
#

#
#
#
#
#

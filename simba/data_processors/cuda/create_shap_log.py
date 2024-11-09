__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import (check_if_dir_exists, check_instance, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_valid_array, check_valid_dataframe,
                                check_valid_lst)
from simba.utils.enums import Formats
from simba.utils.errors import FFMPEGCodecGPUError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import write_df
from simba.utils.warnings import NotEnoughDataWarning


def create_shap_log(rf_clf: Union[str, os.PathLike, RandomForestClassifier],
                    x: Union[pd.DataFrame, np.ndarray],
                    y: Union[pd.DataFrame, pd.Series, np.ndarray],
                    cnt_present: int,
                    cnt_absent: int,
                    x_names: Optional[List[str]] = None,
                    clf_name: Optional[str] = None,
                    save_dir: Optional[Union[str, os.PathLike]] = None,
                    verbose: Optional[bool] = True) -> Union[None, Tuple[pd.DataFrame, pd.DataFrame, int]]:
    """
    Computes SHAP (SHapley Additive exPlanations) values using a GPU for a RandomForestClassifier,
    based on specified counts of positive and negative samples, and optionally saves the results.

    .. image:: _static/img/create_shap_log_cuda.png
       :width: 500
       :align: center

    :param Union[str, os.PathLike, RandomForestClassifier] rf_clf: Trained RandomForestClassifier model or path to the saved model. Can be a string, os.PathLike object, or an instance of RandomForestClassifier.
    :param Union[pd.DataFrame, np.ndarray] x: Input features used for SHAP value computation. Can be a pandas DataFrame or numpy ndarray.
    :param Union[pd.DataFrame, pd.Series, np.ndarray] y:  Target labels corresponding to the input features. Can be a pandas DataFrame, pandas Series, or numpy ndarray with 0 and 1 values.
    :param int cnt_present: Number of positive samples (label=1) to include in the SHAP value computation.
    :param int cnt_absent: Number of negative samples (label=0) to include in the SHAP value computation.
    :param Optional[List[str]] x_names: Optional list of feature names corresponding to the columns in `x`. If `x` is a DataFrame, this is extracted automatically.
    :param Optional[str] clf_name: Optional name for the classifier, used in naming output files. If not provided, it is extracted from the `y` labels if possible.
    :param Optional[Union[str, os.PathLike]] save_dir:  Optional directory path where the SHAP values and corresponding raw features are saved as CSV files.
    :param Optional[bool] verbose: Optional boolean flag indicating whether to print progress messages. Defaults to True.
    :return Union[None, Tuple[pd.DataFrame, pd.DataFrame, int]]: If `save_dir` is None, returns a tuple containing:
                                                                - V: DataFrame with SHAP values, expected value, sum of SHAP values, prediction probability, and target labels.
                                                                - R: DataFrame containing the raw feature values for the selected samples.
                                                                - expected_value: The expected value from the SHAP explainer.
                                                                 If `save_dir` is provided, the function returns None and saves the output to CSV files in the specified directory.

    :example:
    >>> x = np.random.random((1000, 501)).astype(np.float32)
    >>> y = np.random.randint(0, 2, size=(len(x), 1)).astype(np.int32)
    >>> clf_names = [str(x) for x in range(501)]
    >>> results = create_shap_log(rf_clf=MODEL_PATH, x=x, y=y, cnt_present=int(i/2), cnt_absent=int(i/2), clf_name='TEST', x_names=clf_names, verbose=False)
    """

    timer = SimbaTimer(start=True)
    if verbose:
        print('Computing SHAP values (GPU)...')
    if not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None)",
                                  source=create_shap_log.__name__)
    check_instance(source=f'{create_shap_log.__name__} rf_clf', instance=rf_clf,
                   accepted_types=(str, RandomForestClassifier))
    if isinstance(rf_clf, (str, os.PathLike)):
        rf_clf = TrainModelMixin().read_pickle(file_path=rf_clf)
    check_instance(source=f'{create_shap_log.__name__} x', instance=x, accepted_types=(pd.DataFrame, np.ndarray))
    if isinstance(x, np.ndarray):
        check_valid_lst(data=x_names, source=f'{create_shap_log.__name__} x_names', valid_dtypes=(str,),
                        exact_len=x.shape[1])
        check_valid_array(data=x, source=f'{create_shap_log.__name__} x', accepted_ndims=[2, ],
                          accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    else:
        check_valid_dataframe(df=x, source=f'{create_shap_log.__name__} x',
                              valid_dtypes=Formats.NUMERIC_DTYPES.value)
        x_names = list(x.columns)
        x = x.values
    check_instance(source=f'{create_shap_log.__name__} y', instance=y,
                   accepted_types=(pd.DataFrame, np.ndarray, pd.Series))
    if isinstance(y, np.ndarray):
        check_str(name=f'{create_shap_log.__name__} clf_name', value=clf_name)
        y = y.flatten()
    elif isinstance(y, pd.Series):
        clf_name = y.name
        y = y.values.flatten()
    else:
        check_valid_dataframe(df=y, source=f'{create_shap_log.__name__} y',
                              valid_dtypes=Formats.NUMERIC_DTYPES.value, max_axis_1=1)
        clf_name = list(y.columns)[0]
        y = y.values.flatten()
    save_shap_path, save_raw_path = None, None
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
        save_shap_path = os.path.join(save_dir, f"SHAP_values_{clf_name}.csv")
        save_raw_path = os.path.join(save_dir, f"RAW_SHAP_feature_values_{clf_name}.csv")
    check_valid_array(data=y, source=f'{create_shap_log.__name__} y', accepted_values=[0, 1])
    check_int(name=f'{create_shap_log.__name__} cnt_present', value=cnt_present, min_value=1)
    check_int(name=f'{create_shap_log.__name__} cnt_absent', value=cnt_absent, min_value=1)
    target_cnt = np.sum(y)
    absent_cnt = y.shape[0] - target_cnt

    if cnt_present > target_cnt:
        NotEnoughDataWarning(
            msg=f"Data contains {target_cnt} behavior-present annotations. This is less the number of frames you specified to calculate shap values for ({cnt_present}). SimBA will calculate shap scores for the {target_cnt} behavior-present frames available",
            source=create_shap_log.__name__)
        cnt_present = target_cnt
    if absent_cnt < cnt_absent:
        NotEnoughDataWarning(
            msg=f"Data contains {absent_cnt} behavior-absent annotations. This is less the number of frames you specified to calculate shap values for ({cnt_absent}). SimBA will calculate shap scores for the {absent_cnt} behavior-absent frames available",
            source=create_shap_log.__name__)
        cnt_absent = absent_cnt

    target_idx = np.argwhere(y == 1).flatten()
    absent_idx = np.argwhere(y == 0).flatten()
    target_idx = np.sort(np.random.choice(target_idx, cnt_present))
    absent_idx = np.sort(np.random.choice(absent_idx, cnt_absent))
    target_x = x[target_idx]
    absent_x = x[absent_idx]
    X = np.vstack([target_x, absent_x]).astype(np.float32)
    Y = np.hstack([np.ones(target_x.shape[0]), np.zeros(absent_x.shape[0])]).astype(np.int32)
    explainer = shap.explainers.GPUTree(model=rf_clf, data=None, model_output='raw',
                                        feature_names='tree_path_dependent')
    shap_values = explainer.shap_values(X, check_additivity=True)
    V = pd.DataFrame(shap_values[1], columns=x_names).astype(np.float32)
    sum = V.sum(axis=1)
    expected_value = explainer.expected_value[1]
    p = TrainModelMixin().clf_predict_proba(clf=rf_clf, x_df=X)

    V['EXPECTED_VALUE'] = expected_value.round(4)
    V['SUM'] = sum + V['EXPECTED_VALUE']
    V['PREDICTION_PROBABILITY'] = p.round(4)
    V['SUM'] = V['SUM'].round(4)
    V[clf_name] = Y
    x_idx = np.hstack([target_idx, absent_idx])
    R = pd.DataFrame(x[x_idx, :], columns=x_names)
    timer.stop_timer()
    if save_dir is None:
        if verbose:
            stdout_success(msg=f'Shap values compute complete (GPU) for {len(V)} observations.',  elapsed_time=timer.elapsed_time_str)
        return (V, R, expected_value)
    else:
        write_df(df=V, file_type='csv', save_path=save_shap_path)
        write_df(df=R, file_type='csv', save_path=save_raw_path)
        if verbose:
            stdout_success(msg=f'Shap values compute complete (GPU) for {len(V)} observations, and saved in {save_dir}',  elapsed_time=timer.elapsed_time_str)







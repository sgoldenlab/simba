import os
import time
from copy import deepcopy
from typing import Dict, Optional, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier

try:
    from cuml.ensemble import RandomForestClassifier as cuRF
except:
    from sklearn.ensemble import RandomForestClassifier as cuRF

from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_int,
                                check_valid_array, check_valid_boolean,
                                check_valid_dict)
from simba.utils.enums import Formats
from simba.utils.errors import InvalidInputError
from simba.utils.read_write import find_core_cnt, read_pickle, write_pickle

ACCEPTED_MODELS = RandomForestClassifier, cuRF

class OrdinalClassifier():
    """
    This class implements a strategy for ordinal classification by fitting multiple binary classifiers to predict thresholds between classes.

    It is particularly useful for problems where the target variable has an inherent order but uneven intervals between levels. Thi includes human severity scores, for example, seizures, stereotopy, convulsion, bizarre behavior scores ranging fro 0-5.


    .. warning::
       If larger data sizes (>2m) pass a GPU ``cuml.ensemble.RandomForestClassifier`` object.

    .. note::
       `Modified from sklego <`https://github.com/koaning/scikit-lego/blob/main/sklego/meta/ordinal_classification.py>`__.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/simba.model.ordinal_clf.OrdinalClassifier.csv
       :widths: 10, 90
       :align: center
       :header-rows: 1

    References
    ----------
    .. [1] Frank, Eibe, and Mark Hall. “A Simple Approach to Ordinal Classification.” In Machine Learning: ECML 2001, edited by Luc De Raedt and Peter Flach, 2167:145–56. Lecture Notes in Computer Science. Berlin, Heidelberg: Springer Berlin Heidelberg, 2001. https://doi.org/10.1007/3-540-44795-4_13.
    .. [2] Sabnis, Gautam, Leinani Hession, J. Matthew Mahoney, Arie Mobley, Marina Santos, and Vivek Kumar. “Visual Detection of Seizures in Mice Using Supervised Machine Learning,” May 31, 2024. https://doi.org/10.1101/2024.05.29.596520.
    .. [3] Another implementation / benchmarking by Lee Prevost - `https://github.com/leeprevost/OrdinalClassifier/tree/main <https://github.com/leeprevost/OrdinalClassifier/tree/main>`_.
    .. [4] https://github.com/fabianp/mord
    .. [5] Michael J Wurm, Paul J Rathouz, and Bret M Hanlon. Regularized ordinal regression and the ordinalnet r package. Journal of Statistical Software, 99(6), 2021.

    :example:
    >>> X = np.random.randint(0, 500, (100, 50))
    >>> y = np.random.randint(1, 6, (100))
    >>> rf_mdl = TrainModelMixin().clf_define(cuda=False)
    >>> fitted_mdl = OrdinalClassifier.fit(X, y, rf_mdl, -1)
    >>> y_hat = OrdinalClassifier.predict_proba(X, fitted_mdl)
    >>> y = OrdinalClassifier.predict(X, fitted_mdl)
    >>> save_path = r"/mnt/c/Users/sroni/Downloads/Box4-20191208T1639-1652/ord_mdl/mdl.pickle"
    >>> OrdinalClassifier.save(mdl=fitted_mdl, save_path=save_path)
    >>> rf_mdl = OrdinalClassifier.load(file_path=save_path)
    >>> y_hat = OrdinalClassifier.predict_proba(X, rf_mdl)
    """

    def __init__(self):
        pass

    @staticmethod
    def fit(X: np.ndarray,
            y: np.ndarray,
            clf: Union[ACCEPTED_MODELS],
            core_cnt: int = -1,
            parallel: Optional[bool] = True) -> Dict[int, Union[ACCEPTED_MODELS]]:

        def _fit_binary_estimator(clf: Union[RandomForestClassifier, cuRF],
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  y_label: int):
            #print(f'Fitting model for y label {y_label}...')
            y_bin = (y <= y_label).astype(np.int32)
            return clone(clf).fit(X, y_bin)

        check_valid_boolean(value=parallel, raise_error=True, source=f'{OrdinalClassifier.__name__} fit parallel')
        classes_ = np.sort(np.unique(y))
        check_valid_array(data=classes_, source=f'{__class__.__name__} y', accepted_ndims=(1,), accepted_dtypes=Formats.INTEGER_DTYPES.value)
        if len(classes_) < 3:
            raise InvalidInputError(msg=f'Found {len(classes_)} classes in y [{classes_}], requires at least 3', source=f'{OrdinalClassifier.__name__} fit')
        intervals = [classes_[i] - classes_[i-1] for i in range(1, len(classes_))]
        if len(set(intervals)) != 1:
            raise InvalidInputError(msg=f'The values in y ({classes_}) are not of equal interval.', source=f'{OrdinalClassifier.__name__} fit')
        check_valid_array(data=X, source=f'{__class__.__name__} x', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=(y.shape[0],))
        if not isinstance(clf, (RandomForestClassifier, cuRF)) or ('predict_proba' not in dir(clf)):
            raise InvalidInputError(msg=f'clf is not of valid type: {type(clf)} (accepted: {ACCEPTED_MODELS})', source=f'{OrdinalClassifier.__name__} fit')
        parallel = False if isinstance(clf, cuRF) else parallel
        check_int(name='core_cnt', min_value=-1, unaccepted_vals=[0], value=core_cnt)
        core_cnt = [find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt][0]
        if parallel:
            return dict(zip(classes_[:-1], Parallel(n_jobs=core_cnt)(delayed(_fit_binary_estimator)(clf, X, y, y_label) for y_label in classes_[:-1])))
        else:
            results = {}
            for y_label in classes_[:-1]:
                results[y_label] = _fit_binary_estimator(clf=clf, X=X, y=y, y_label=y_label)
            return results


    @staticmethod
    def predict_proba(X: np.ndarray, mdl: Dict[int, Union[ACCEPTED_MODELS]]) -> np.ndarray:
        OrdinalClassifier._check_valid_mdl_dict(mdls=mdl)
        check_valid_array(data=X, source=f'{__class__.__name__} x', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        n_features = mdl[list(mdl.keys())[0]].n_features_ if hasattr(mdl, 'n_features_') else mdl[list(mdl.keys())[0]].n_features_in_

        if n_features != X.shape[1]:
            raise InvalidInputError(msg=f'Model expects {mdl[list(mdl.keys())[0]].n_features_} features, got {X.shape[1]}.', source=f'{OrdinalClassifier.__name__} predict')
        check_valid_array(data=X, source=f'{__class__.__name__} x', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        raw_proba = np.array([estimator.predict_proba(X)[:, 1] for estimator in mdl.values()]).T
        return np.diff(np.column_stack((np.zeros(X.shape[0]), raw_proba, np.ones(X.shape[0]))), n=1, axis=1)


    @staticmethod
    def predict(X: np.ndarray, mdl: Dict[int, Union[ACCEPTED_MODELS]]) -> np.ndarray:
        OrdinalClassifier._check_valid_mdl_dict(mdls=mdl)
        check_valid_array(data=X, source=f'{__class__.__name__} x', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        n_features = mdl[list(mdl.keys())[0]].n_features_ if hasattr(mdl, 'n_features_') else mdl[list(mdl.keys())[0]].n_features_in_

        if n_features != X.shape[1]:
            raise InvalidInputError(msg=f'Model expects {mdl[list(mdl.keys())[0]].n_features_} features, got {X.shape[1]}.', source=f'{OrdinalClassifier.__name__} predict')
        return np.argmax(OrdinalClassifier.predict_proba(X, mdl=mdl), axis=1)

    @staticmethod
    def save(mdl: Dict[int, Union[ACCEPTED_MODELS]], save_path: Union[str, os.PathLike]):
        OrdinalClassifier._check_valid_mdl_dict(mdls=mdl)
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=f'{OrdinalClassifier.__name__} save')
        write_pickle(data=mdl, save_path=save_path)


    @staticmethod
    def load(file_path: Union[str, os.PathLike]) -> Dict[int, Union[ACCEPTED_MODELS]]:
        check_file_exist_and_readable(file_path=file_path)
        return read_pickle(data_path=file_path)


    @staticmethod
    def _check_valid_mdl_dict(mdls: Dict[int, Union[ACCEPTED_MODELS]]) -> None:
        check_valid_dict(x=mdls, valid_keys=list(range(0, 5000)), min_len_keys=2, valid_key_dtypes=Formats.INTEGER_DTYPES.value, valid_values_dtypes=ACCEPTED_MODELS)
        features_in_cnt = []
        for mdl in mdls.values():
            features_in_cnt.append(mdl.n_features_) if hasattr(mdl, 'n_features_') else features_in_cnt.append(mdl.n_features_in_)
        if len(set(features_in_cnt)) != 1:
            raise InvalidInputError(msg=f'The models has different N features [{features_in_cnt}]')




#
# for
#


# NS = [1000, 10000, 100000, 1000000, 2000000]
# CUDA = False
# for N in NS:
#     X = np.random.randint(0, 500, (N, 250)).astype(np.int32)
#     y = np.random.randint(1, 6, (N)).astype(np.int32)
#     rf_mdl = TrainModelMixin().clf_define(cuda=CUDA, verbose=False, n_estimators=250)
#     start = time.perf_counter()
#     fitted_mdl = OrdinalClassifier.fit(X, y, rf_mdl, -1, parallel=False)
#     stop = time.perf_counter()
#     elapsed = stop - start
#     print(N, '\t'* 4, elapsed)

#
#
# y_hat = OrdinalClassifier.predict_proba(X, fitted_mdl)
# y = OrdinalClassifier.predict(X, fitted_mdl)
# save_path = r"/mnt/c/Users/sroni/Downloads/Box4-20191208T1639-1652/ord_mdl/mdl.pickle"
# OrdinalClassifier.save(mdl=fitted_mdl, save_path=save_path)
# rf_mdl = OrdinalClassifier.load(file_path=save_path)
# y_hat = OrdinalClassifier.predict_proba(X, rf_mdl)
#

# X = np.random.randint(0, 500, (100, 50))
# y = np.random.randint(1, 6, (100))
# rf_mdl = TrainModelMixin().clf_define()
# fitted_mdls = OrdinalClassifier.fit(X, y, rf_mdl, -1)
# y_hat = OrdinalClassifier.predict_proba(X, fitted_mdls)
# y = OrdinalClassifier.predict(X, fitted_mdls)
# OrdinalClassifier.save(mdl=fitted_mdls, save_path=r"C:\Users\sroni\OneDrive\Desktop\mdl.pk")

#predict_proba(X)
# ordinal_clf.predict(X)
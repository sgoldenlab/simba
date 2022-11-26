from joblib import Parallel, delayed
import numpy as np
import xgboost as xgb
import itertools
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import glob, os
import pickle
import json
from collections import defaultdict

class XGBoostModellerCPU(object):
    """
    Class for fitting XGBoost models using parameter grid search and stratisfied cross-validation on CPU.

    :param list max_depths: List of max depths
    :param list n_estimators: List of estimators
    :param list scale_pos_weight: List of positive scale weights
    :param int scale_pos_weight: Number of stratisfied K-fold cross validations at each parameter.

    :Example:
    >>> grid_modeler = XGBoostModellerCPU(max_depths=[10], n_estimators=[50], scale_pos_weight=[5], n_folds=2)
    >>> grid_modeler.read_fit_data(data_path='data/_test_pickle', feature_path='data/lists/features.json', target='label')
    >>> grid_modeler.grid_search_k_fold()
    >>> best_model = grid_modeler.find_best_model()
    >>> grid_modeler.save_model(model=best_model, path='data/models/my_cpu_model.pickle')
    """

    def __init__(self,
                 max_depths: list,
                 n_estimators: list,
                 scale_pos_weight: list,
                 n_folds: int):

        self.max_depths = max_depths
        self.n_estimators = n_estimators
        self.scale_pos_weight = scale_pos_weight
        self.n_folds = n_folds
        self.search_space = list(itertools.product(*[self.max_depths, self.n_estimators, self.scale_pos_weight]))
        self.clf_xgb = xgb.XGBClassifier(objective='binary:logistic')
        self.k_fold = StratifiedKFold(n_splits=self.n_folds, shuffle=True)

    def check_lists_are_identical(self, input: list):
        if not all(input[0] == b for b in input[1:]):
            raise AssertionError('The fields of read files are not identical.')
        else:
            pass

    def check_arrays_have_same_shape(self, input: list):
        if not all(x.shape[1] == input[0].shape[1] for x in input):
            raise AssertionError('Shape 1 of all read files are not identical.')
        else:
            pass

    def check_file_exist_and_readable(self,
                                      input_type: str,
                                      file_path: str):

        if not os.path.isfile(file_path):
            raise FileNotFoundError('The path for {} ({}) is not a valid file path'.format(input_type, file_path))
        elif not os.access(file_path, os.R_OK):
            raise IOError('The path for {} ({}) is not corrupted'.format(input_type, file_path))

    def read_fit_data(self,
                      data_path: str,
                      feature_path: str,
                      target: str):

        def multiprocess_read_pickles(file_paths: list):
            data_arr_lst = []
            data_field_list = []
            for file_path in file_paths:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    data_arr_lst.append(data['data'])
                    data_field_list.append(data['fields'])

            return data_arr_lst, data_field_list

        self.check_file_exist_and_readable(input_type='Feature json', file_path=feature_path)
        data_files = glob.glob(data_path + '/*')
        data_files_split = [data_files[i:i + 2] for i in range(0, len(data_files), 2)]
        results = Parallel(n_jobs=1, verbose=1, backend="loky")(
            delayed(multiprocess_read_pickles)(x) for x in data_files_split)
        data = [item for sublist in [x[0] for x in results] for item in sublist]
        fields = [item for sublist in [x[1] for x in results] for item in sublist]
        self.check_lists_are_identical(input=fields)
        self.check_arrays_have_same_shape(input=data)
        data_arr, feature_list = np.vstack(data), fields[0]
        with open(feature_path, 'r') as fp:
            self.features = json.load(fp)['features']
        self.feature_idx = []
        for feature in self.features:
            self.feature_idx.append(feature_list.index(feature))
        self.x_data = data_arr[:, self.feature_idx]
        self.y_data = data_arr[:, [feature_list.index(target)]].flatten().astype(int)

    def grid_search_k_fold(self):
        self.results = {}
        for h_cnt, h in enumerate(self.search_space):
            self.results[h_cnt] = {}
            self.parameters = {'max_depth': h[0],
                               'n_estimators': h[1],
                               'scale_pos_weight': h[2],
                               'verbosity': 1}
            self.clf_xgb.set_params(**self.parameters)
            for fold_cnt, (train_index, test_index) in enumerate(self.k_fold.split(self.x_data, self.y_data)):
                self.x_train, self.x_test = self.x_data[train_index], self.x_data[test_index]
                self.y_train, self.y_test = self.y_data[train_index], self.y_data[test_index]
                self.model = self.clf_xgb.fit(self.x_train, self.y_train)
                self.feature_importances()
                self.evaluate()
                self.results[h_cnt][fold_cnt] = {'model': self.model,
                                                 'feature_importances': self.weight_importance,
                                                 'pr_auc': self.pr_auc,
                                                 'pr_df': self.pr,
                                                 'parameters': self.parameters}
            self.compute_agg_model_statistics(model_cnt=h_cnt)

    def feature_importances(self):
        self.weight_importance = {}
        for k, v in self.model.get_booster().get_fscore().items():
            self.weight_importance[self.features[int(k[1:])]] = v

    def evaluate(self):
        prediction_proba = self.model.predict_proba(self.x_test)
        precision, recall, thresholds = precision_recall_curve(self.y_test, prediction_proba[:, [1]])
        self.pr_auc = auc(recall, precision)
        self.pr = defaultdict(list)
        for p, r, t in zip(precision, recall, thresholds):
            self.pr['precision'].append(p)
            self.pr['recall'].append(r)
            self.pr['thresholds'].append(t)

    def compute_agg_model_statistics(self, model_cnt: int):
        pr_aucs, feature_importances = [], {}
        for fold_cnt, fold in enumerate(self.results[model_cnt].keys()):
            pr_aucs.append(self.results[model_cnt][fold]['pr_auc'])
            feature_importances[fold_cnt] = self.results[model_cnt][fold]['feature_importances']
        self.results[model_cnt]['aggregate_statistics'] = {}
        self.results[model_cnt]['aggregate_statistics']['pr_auc'] = sum(pr_aucs) / len(pr_aucs)
        feature_importances_dict = defaultdict(list)
        self.results[model_cnt]['aggregate_statistics']['feature_importances'] = {}
        for fold_name, fold_importances in feature_importances.items():
            for k, v in fold_importances.items():
                feature_importances_dict[k].append(v)
        for k, v in feature_importances_dict.items():
            self.results[model_cnt]['aggregate_statistics']['feature_importances'][k] = sum(v) / len(v)

    def find_best_model(self):
        highest_agg_pr = -np.inf
        best_model = None
        for k in self.results.keys():
            if self.results[k]['aggregate_statistics']['pr_auc'] > highest_agg_pr:
                highest_agg_pr = self.results[k]['aggregate_statistics']['pr_auc']
                best_model = self.results[k]

        return best_model

    def save_model(self, model: dict, path: str):
        with open(path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
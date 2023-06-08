import os.path, glob
import pandas as pd
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.read_write import read_config_file, read_df

@pytest.fixture(params=['tests/data/test_projects/two_c57/project_folder/project_config.ini'])
def parsed_config_args(request):
    return read_config_file(config_path=request.param)


# @pytest.mark.parametrize("file_paths", [['tests/data/test_projects/two_c57/project_folder/csv/targets_inserted/Together_1.csv']])
# def test_read_all_files_in_folder(file_paths):
#     results = TrainModelMixin().read_all_files_in_folder(file_paths=file_paths, file_type='csv', classifier_names=['Attack'])
#     assert len(results) == 1738
#     assert len(results.columns) == 50

def test_read_in_all_model_names_to_remove(parsed_config_args):
    results = TrainModelMixin().read_in_all_model_names_to_remove(config=parsed_config_args, model_cnt=2, clf_name='Attack')
    assert results == ['Sniffing']

@pytest.mark.parametrize("annotations_lst", [['Feature_1'], ['Feature_2']])
def test_delete_other_annotation_columns(annotations_lst):
    df = pd.DataFrame(columns=['Feature_1', 'Feature_2'])
    results = TrainModelMixin().delete_other_annotation_columns(df=df, annotations_lst=annotations_lst)
    assert len(results.columns) == 1
    assert annotations_lst[0] not in results.columns

@pytest.mark.parametrize("clf_name", ['X1', 'X1', 'X3'])
def test_split_df_to_x_y(clf_name):
    df = pd.DataFrame(data=[[1, 2, 0], [2, 4, 0], [4, 10, 1]], columns=['X1', 'X2', 'X3'])
    y = df.pop(clf_name)
    assert list(df.columns) == list([x for x in df.columns if x != clf_name])
    assert y.name == clf_name

@pytest.mark.parametrize("sample_ratio", [1])
def test_random_undersampler(sample_ratio):
    x_train = pd.DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y_train = pd.Series([1, 0, 0], name='Test')
    x_train_out, y_train_out = TrainModelMixin().random_undersampler(x_train=x_train, y_train=y_train, sample_ratio=sample_ratio)
    assert x_train_out.reset_index(drop=True).equals(pd.DataFrame([[1, 2, 3], [1, 2, 3]]))
    assert y_train_out.reset_index(drop=True).equals(pd.Series([1, 0], name='Test'))

@pytest.mark.parametrize("clf_path", ['tests/data/test_projects/two_c57/models/generated_models/Attack.sav'])
def test_calc_permutation_importance(clf_path):
    x_test = np.array([[1, 2], [1, 2], [1, 2]])
    y_test = np.array([[1], [1], [0]])
    clf = read_df(file_path=clf_path, file_type='pickle')
    _ = TrainModelMixin().calc_permutation_importance(x_test=x_test, y_test=y_test, clf=clf, feature_names=['Feature_1', 'Feature_2'], clf_name='Attack', save_dir=os.path.dirname(clf_path))
    assert os.path.isfile(os.path.join(os.path.dirname(clf_path), 'Attack_permutations_importances.csv'))


@pytest.mark.parametrize("clf_path", ['tests/data/test_projects/two_c57/models/generated_models/Attack.sav'])
def test_calc_learning_curve(clf_path):
    x, y = np.random.randint(1, 10, size=(10, 2)), np.random.randint(0, 2, size=(10))
    data = pd.DataFrame(x)
    data['target'] = y
    clf = read_df(file_path=clf_path, file_type='pickle')
    train_model_mixin = TrainModelMixin()
    train_model_mixin.calc_learning_curve(x_y_df=data,
                                          clf_name='target',
                                          shuffle_splits=3,
                                          dataset_splits=2,
                                          tt_size=0.2,
                                          rf_clf=clf,
                                          save_dir=os.path.dirname(clf_path))
    assert os.path.isfile(train_model_mixin.learning_curve_save_path)

@pytest.mark.parametrize("clf_path", ['tests/data/test_projects/two_c57/models/generated_models/Attack.sav'])
def test_calc_pr_curve(clf_path):
    x, y = pd.DataFrame(np.random.randint(1, 10, size=(10, 2))), pd.DataFrame(np.random.randint(0, 2, size=(10)), columns=['target'])
    clf = read_df(file_path=clf_path, file_type='pickle')
    train_model_mixin = TrainModelMixin()
    train_model_mixin.calc_pr_curve(rf_clf=clf,
                                    x_df=x,
                                    y_df=y,
                                    clf_name='target',
                                    save_dir=os.path.dirname(clf_path))
    assert os.path.isfile(train_model_mixin.pr_save_path)


@pytest.mark.parametrize("clf_path", ['tests/data/test_projects/two_c57/models/generated_models/Attack.sav'])
def test_create_clf_report(clf_path):
    x, y = pd.DataFrame(np.random.randint(1, 10, size=(500, 2))), pd.DataFrame(np.random.randint(0, 2, size=(500)), columns=['target'])
    train_model_mixin = TrainModelMixin()
    clf = read_df(file_path=clf_path, file_type='pickle')
    train_model_mixin.create_clf_report(rf_clf=clf,
                                        x_df=x,
                                        y_df=y,
                                        class_names=['not_target', 'target'],
                                        save_dir=os.path.dirname(clf_path))

@pytest.mark.parametrize("clf_path", ['tests/data/test_projects/two_c57/models/generated_models/Attack.sav'])
def test_create_x_importance_log(clf_path):
    x, y = pd.DataFrame(np.random.randint(1, 10, size=(500, 2))), pd.DataFrame(np.random.randint(0, 2, size=(500)), columns=['target'])
    clf = read_df(file_path=clf_path, file_type='pickle')
    train_model_mixin = TrainModelMixin()
    train_model_mixin.create_x_importance_log(rf_clf=clf,
                                              x_names=list(x.columns),
                                              clf_name='target',
                                              save_dir=os.path.dirname(clf_path))
    assert os.path.isfile(train_model_mixin.f_importance_save_path)


@pytest.mark.parametrize("config_path, clf_path", [['tests/data/test_projects/two_c57/project_folder/project_config.ini',
                                                   'tests/data/test_projects/two_c57/models/generated_models/Attack.sav']])
def test_create_shap_log(config_path, clf_path):
    x, y = pd.DataFrame(np.random.randint(1, 10, size=(500, 2))), pd.Series(np.random.randint(0, 2, size=(500)), name='target')
    clf = read_df(file_path=clf_path, file_type='pickle')
    train_model_mixin = TrainModelMixin()
    train_model_mixin.create_shap_log(ini_file_path=config_path,
                                      rf_clf=clf,
                                      x_df=x,
                                      y_df=y,
                                      x_names=x.columns,
                                      clf_name='target',
                                      cnt_present=1,
                                      cnt_absent=1,
                                      save_path=os.path.dirname(clf_path))
    assert os.path.isfile(train_model_mixin.out_df_shap_path)
    assert os.path.isfile(train_model_mixin.out_df_raw_path)

@pytest.mark.parametrize("config_path", ['tests/data/test_projects/two_c57/project_folder/project_config.ini'])
def test_get_all_clf_names(config_path):
    config = read_config_file(config_path=config_path)
    train_model_mixin = TrainModelMixin()
    clf_names = train_model_mixin.get_all_clf_names(config=config, target_cnt=2)
    assert clf_names == ['Attack', 'Sniffing']


def test_bout_train_test_splitter():
    x, y = pd.DataFrame(data=[[11, 23, 12], [87, 65, 76], [23, 73, 27], [10, 29, 2],
                              [12, 32, 42], [32, 73, 2], [21, 83, 98], [98, 1, 1]]), pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    x_train, x_test, y_train, y_test = TrainModelMixin().bout_train_test_splitter(x_df=x, y_df=y, test_size=0.5)
    assert (all(isinstance(var, pd.DataFrame) for var in [x_train, x_test]))
    assert (all(isinstance(var, pd.Series) for var in [y_train, y_test]))

def test_check_sampled_dataset_integrity():
    x, y = pd.DataFrame(data=[[11, 23, 12], [87, 65, 76], [23, 73, 27], [10, 29, 2],
                              [12, 32, 42], [32, 73, 2], [21, 83, 98], [98, 1, 1]]), pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    _ = TrainModelMixin().check_sampled_dataset_integrity(x_df=x, y_df=y)
    x.loc[0: 0] = np.nan
    with pytest.raises(Exception):
        _ = TrainModelMixin().check_sampled_dataset_integrity(x_df=x, y_df=y)

@pytest.mark.parametrize("clf_path", ['tests/data/test_projects/two_c57/models/generated_models/Attack.sav'])
def test_partial_dependence_calculator(clf_path):
    x = pd.DataFrame(np.random.randint(1, 10, size=(500, 2)))
    clf = read_df(file_path=clf_path, file_type='pickle')
    TrainModelMixin().partial_dependence_calculator(x_df=x,
                                                    clf=clf,
                                                    clf_name='target',
                                                    save_dir=os.path.dirname(clf_path))

@pytest.mark.parametrize("clf_path", ['tests/data/test_projects/two_c57/models/generated_models/Attack.sav'])
def test_clf_predict_proba(clf_path):
    x = pd.DataFrame(np.random.randint(1, 10, size=(10, 2)))
    clf = read_df(file_path=clf_path, file_type='pickle')
    results = TrainModelMixin().clf_predict_proba(clf=clf,x_df=x)
    assert results.shape[0] == len(x)
    assert all(i <= 1.0 for i in results)
    assert all(i >= 0.0 for i in results)

def test_clf_fit():
    clf = RandomForestClassifier()
    x, y = pd.DataFrame(data=[[11, 23, 12], [87, 65, 76], [23, 73, 27], [10, 29, 2],
                              [12, 32, 42], [32, 73, 2], [21, 83, 98], [98, 1, 1]]), pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    clf = TrainModelMixin().clf_fit(clf=clf, x_df=x, y_df=y)
    assert hasattr(clf, "classes_")


# @pytest.mark.parametrize("file_paths", [['tests/data/test_projects/two_c57/project_folder/csv/targets_inserted/Together_1.csv']])
# def test_read_all_files_in_folder_mp(file_paths):
#     data = TrainModelMixin().read_all_files_in_folder_mp(file_paths=file_paths, file_type='csv', classifier_names=['Attack', 'Sniffing'])
#     assert len(data) == 1738
#     assert len(data.columns) == 52

# @pytest.mark.parametrize("config_path, clf_path", [['tests/data/test_projects/two_c57/project_folder/project_config.ini',
#                                                    'tests/data/test_projects/two_c57/models/generated_models/Attack.sav']])
# def test_create_shap_log_mp(config_path, clf_path):
#     x, y = pd.DataFrame(np.random.randint(1, 10, size=(500, 2))), pd.Series(np.random.randint(0, 2, size=(500)), name='target')
#     clf = read_df(file_path=clf_path, file_type='pickle')
#     train_model_mixin = TrainModelMixin()
#     train_model_mixin.create_shap_log_mp(ini_file_path=config_path,
#                                          rf_clf=clf,
#                                          x_df=x,
#                                          y_df=y,
#                                          x_names=x.columns,
#                                          clf_name='target',
#                                          cnt_present=1,
#                                          cnt_absent=1,
#                                          save_path=os.path.dirname(clf_path))
#     assert os.path.isfile(train_model_mixin.out_df_shap_path)
#     assert os.path.isfile(train_model_mixin.out_df_raw_path)







    #
    #
    # def calc_learning_curve(self,
    #                         x_y_df: pd.DataFrame,
    #                         clf_name: str,
    #                         shuffle_splits: int,
    #                         dataset_splits: int,
    #                         tt_size: float,
    #                         rf_clf: RandomForestClassifier,
    #                         save_dir: str,
    #                         save_file_no: Optional[int] = None) -> None:

#test_clf_fit()

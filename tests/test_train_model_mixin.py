import os.path
import pandas as pd
import pytest
import numpy as np

from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.read_write import read_config_file, read_df

@pytest.fixture(params=['/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/two_c57/project_folder/project_config.ini'])
def parsed_config_args(request):
    return read_config_file(config_path=request.param)


@pytest.mark.parametrize("file_paths", [['/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/two_c57/project_folder/csv/targets_inserted/Together_1.csv']])
def test_read_all_files_in_folder(file_paths):
    results = TrainModelMixin().read_all_files_in_folder(file_paths=file_paths, file_type='csv', classifier_names=['Attack'])
    assert len(results) == 1738
    assert len(results.columns) == 50

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

@pytest.mark.parametrize("clf_path", ['/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/two_c57/models/generated_models/Attack.sav'])
def test_calc_permutation_importance(clf_path):
    x_test = np.array([[1, 2], [1, 2], [1, 2]])
    y_test = np.array([[1], [1], [0]])
    clf = read_df(file_path=clf_path, file_type='pickle')
    _ = TrainModelMixin().calc_permutation_importance(x_test=x_test, y_test=y_test, clf=clf, feature_names=['Feature_1', 'Feature_2'], clf_name='Attack', save_dir=os.path.dirname(clf_path))
    assert os.path.isfile(os.path.join(os.path.dirname(clf_path), 'Attack_permutations_importances.csv'))



    #
    # x_test: np.ndarray,
    # y_test: np.ndarray,
    # clf: RandomForestClassifier,
    # feature_names: List[str],
    # clf_name: str,
    # save_dir: Union[str, os.PathLike],
    # save_file_no: Optional[int] = None) -> None:
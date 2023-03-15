import pickle
import os, glob
import pandas as pd
from sklearn.preprocessing import (MinMaxScaler,
                                   StandardScaler,
                                   QuantileTransformer)
from sklearn.feature_selection import VarianceThreshold
from simba.features_scripts.unit_tests import read_video_info
from simba.enums import Options
from simba.misc_tools import detect_bouts
from joblib import Parallel, delayed
import numpy as np
from joblib.externals.loky import get_reusable_executor


def read_pickle(data_path: str) -> dict:
    if os.path.isdir(data_path):
        data = {}
        files_found = glob.glob(data_path + '/*.pickle')
        if len(files_found) == 0:
            print(f'SIMBA ERROR: Zero pickle files found in {data_path}.')
            raise ValueError
        for file_cnt, file_path in enumerate(files_found):
            with open(file_path, 'rb') as f:
                data[file_cnt] = pickle.load(f)
    if os.path.isfile(data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

    return data


def write_pickle(data: dict, save_path: str):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def check_that_directory_is_empty(directory: str) -> None:
    try:
        all_files_in_folder = [f for f in next(os.walk(directory))[2] if not f[0] == '.']
    except StopIteration:
        return 0
    else:
        if len(all_files_in_folder) > 0:
            print('''ssss''')
            print(f'SIMBA ERROR: The {directory} is not empty and contains {str(len(all_files_in_folder))} files. Use a directory that is empty.')
            raise ValueError()


def get_cluster_cnt(data: np.array,
                    clusterer_name: str,
                    minimum_clusters: int = 1) -> int:
    cnt = np.unique(data).shape[0]
    if cnt < minimum_clusters:
        print(f'SIMBA ERROR: Clustrer {clusterer_name} has {str(cnt)} clusters, but {str(minimum_clusters)} clusters is required for the operation.')
        raise ValueError()
    else:
        return cnt

def check_directory_exists(directory: str) -> None:
    if not os.path.isdir(directory):
        print(f'SIMBA ERROR: {directory} is not a valid directory.')
        raise NotADirectoryError
    else:
        pass

def define_scaler(scaler_name: str):
    if scaler_name not in Options.SCALER_NAMES.value:
        print('SIMBA ERROR: {} is not a valid scaler option (VALID OPTIONS: {}'.format(scaler_name, Options.SCALER_NAMES.value))
        raise ValueError()
    if scaler_name == 'MIN-MAX':
        return MinMaxScaler()
    elif scaler_name == 'STANDARD':
        return StandardScaler()
    elif scaler_name == 'QUANTILE':
        return QuantileTransformer()

def find_low_variance_fields(data: pd.DataFrame, variance: float):
    feature_selector = VarianceThreshold(threshold=round((variance / 100), 2))
    feature_selector.fit(data)
    return [c for c in data.columns if c not in data.columns[feature_selector.get_support()]]


def drop_low_variance_fields(data: pd.DataFrame, fields: list):
    return data.drop(columns=fields)


def scaler_transform(data: pd.DataFrame, scaler: object):
    return pd.DataFrame(scaler.transform(data), columns=data.columns)

def check_expected_fields(data_fields: list, expected_fields: list):
    remaining_fields = [x for x in data_fields if x not in expected_fields]
    if len(remaining_fields) > 0:
        print(f'The data contains {str(len(remaining_fields))} unexpected field(s): {str(remaining_fields)}')
        raise ValueError()
    else:
        pass

def find_embedding(embeddings: dict, hash: str):
    for k, v in embeddings.items():
        if v['HASH'] == hash:
            return v
    print(f'SIMBA ERROR: {hash} embedder could not be found in the embedding directory.')
    raise FileNotFoundError()


def bout_aggregator(data: pd.DataFrame,
                    clfs: list,
                    feature_names: list,
                    aggregator: str,
                    min_bout_length: int,
                    video_info: pd.DataFrame):
    print('Calculating bout aggregate statistics...')

    def bout_aggregator_mp(frms, data, clf_name):
        bout_df = data.iloc[frms[0]: frms[1]+1]
        bout_video, start_frm, end_frm = bout_df['VIDEO'].values[0], bout_df['FRAME'].values[0], bout_df['FRAME'].values[-1]
        if aggregator == 'MEAN':
            agg_df = pd.DataFrame(bout_df[feature_names].mean()).T
            agg_df['PROBABILITY'] = bout_df[f'Probability_{clf_name}'].mean()
        elif aggregator == 'MEDIAN':
            agg_df = pd.DataFrame(bout_df[feature_names].median()).T
            agg_df['PROBABILITY'] = bout_df[f'Probability_{clf_name}'].median()
        agg_df['CLASSIFIER'] = clf_name
        agg_df.insert(0, 'END_FRAME', end_frm)
        agg_df.insert(0, 'START_FRAME', start_frm)
        agg_df.insert(0, 'VIDEO', bout_video)
        return agg_df

    output = []
    for video in data['VIDEO'].unique():
        video_df = data[data['VIDEO'] == video].reset_index(drop=True)
        for clf in clfs:
            _, _, fps = read_video_info(vid_info_df=video_info, video_name=video)
            bouts = detect_bouts(data_df=video_df, target_lst=[clf], fps=fps).sort_values(by='Start_frame')
            bouts = bouts[bouts['Bout_time'] >= min_bout_length / 1000][['Start_frame', 'End_frame']].values
            if len(bouts) > 0:
                bouts = [x.tolist() for x in bouts]
                results = Parallel(n_jobs=-1, verbose=0, backend="loky")(delayed(bout_aggregator_mp)(j, video_df, clf) for j in bouts)
                results = pd.concat(results, axis=0).sort_values(by=['VIDEO', 'START_FRAME'])
                output.append(results)
    get_reusable_executor().shutdown(wait=True)

    return pd.concat(output, axis=0).reset_index(drop=True)




#
# vid_info = read_video_info_csv(file_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/logs/video_info.csv')
#
# data = pd.read_csv('/Users/simon/Desktop/bout_agg_tester.csv', index_col=0)
# data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
# feature_names = [x for x in data.columns if x not in ['Attack', 'Sniffing', 'Probability_Attack', 'Probability_Sniffing']]
# feature_names = feature_names[3:]
# bout_aggregator(data=data, clfs=['Attack', 'Sniffing'], feature_names=feature_names, min_bout_length=66, aggregator='MEAN', video_info=vid_info)
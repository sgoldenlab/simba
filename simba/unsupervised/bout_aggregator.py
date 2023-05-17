__author__ = "Simon Nilsson"

import pandas as pd
from typing import List
from joblib.externals.loky import get_reusable_executor
from joblib import Parallel, delayed
from simba.utils.read_write import read_video_info
from simba.utils.data import detect_bouts

def bout_aggregator(data: pd.DataFrame,
                    clfs: List[str],
                    feature_names: List[str],
                    aggregator: str,
                    min_bout_length: int,
                    video_info: pd.DataFrame) -> pd.DataFrame:

    """
    Helper to aggregate features to bout-level for unsupervised analyses.

    :param pd.DataFrame data: DataFrame with features.
    :param List[str] clfs: Names of classifiers
    :param feature_names: Names of features
    :param str aggregator: Aggregation type, e.g., 'MEAN', 'MEDIAN'
    :param int min_bout_length: The length of the shortest allowed bout in milliseconds.
    :param pd.DataFrame video_info: Holding video fps, resolution etc.
    :return pd.DataFrame: Featurized data at aggregate bout level.
    """

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
    for cnt, video in enumerate(data['VIDEO'].unique()):
        print(f'Processing video {video}...({str(cnt+1)}/{str(len(data["VIDEO"].unique()))})')
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
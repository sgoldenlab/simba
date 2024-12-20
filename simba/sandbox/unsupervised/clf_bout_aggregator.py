__author__ = "Simon Nilsson"

import os
from typing import List, Optional, Union, Tuple
try:
    from typing import Literal
except:
    from typing_extensions import Literal

import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from simba.utils.checks import (check_valid_boolean, check_int, check_str, check_valid_lst, check_float, check_all_file_names_are_represented_in_video_log, check_valid_dataframe, check_if_dir_exists)
from simba.utils.data import detect_bouts
from simba.utils.enums import Formats, Methods
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import find_core_cnt, read_video_info, read_df, read_video_info_csv, find_files_of_filetypes_in_directory, get_fn_ext, write_pickle

def video_clf_bout_aggregator(data: Union[str, os.PathLike],
                              feature_names: List[str],
                              clf_names: Union[str, List[str]],
                              pose_names: Union[str, List[str]],
                              sample_rate: float,
                              min_bout_length: Optional[float] = None,
                              verbose: bool = True,
                              agg_method: Literal["mean", "median"] = "mean") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Helper to aggregate features to bout-level representations.

    :param pd.DataFrame data: DataFrame with features, pose, and classifications.
    :param List[str] clfs: Names of classifier columns
    :param feature_names: Names of feature columns
    :param feature_names: Names of pose columns
    :param Optional[Literal['MEAN', 'MEDIAN']] aggregator: Aggregation type, e.g., 'MEAN', 'MEDIAN'. Default 'MEAN'.
    :param Optional[int] min_bout_length: The length of the shortest allowed bout in milliseconds. Default 0 which means all bouts.
    :param pd.DataFrame sample_rate: The sample rate at which the data was collected.
    :return: Three dataframe tuple: the aggregate feature values, metadata associated with the bout (i.e., classification probability), and the pose associated with the bout.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """

    timer = SimbaTimer(start=True)
    check_valid_lst(source='video_bout_aggregator feature_names', valid_dtypes=(str,), min_len=1, data=feature_names)
    check_valid_lst(source='video_bout_aggregator pose_names', valid_dtypes=(str,), min_len=1, data=pose_names)
    if isinstance(clf_names, list):
        check_valid_lst(source='video_bout_aggregator clf_names', valid_dtypes=(str,), min_len=1, data=clf_names)
    else:
        check_str(name='video_bout_aggregator clf_names', value=clf_names)
        clf_names = [clf_names]
    check_float(name='video_bout_aggregator sample_rate', value=sample_rate, min_value=10e-6)
    if min_bout_length is not None:
        check_float(name='video_bout_aggregator min_bout_length', value=min_bout_length, min_value=0)
    else:
        min_bout_length = 0
    check_str(name='video_bout_aggregator agg_method', value=agg_method, options=Methods.AGG_METHODS.value)
    name = get_fn_ext(filepath=data)[1]
    if verbose:
        print(f"Calculating bout aggregate statistics for video {name}...")
    data = read_df(file_path=data, file_type='csv')
    p_cols = [f'Probability_{x}' for x in clf_names]
    check_valid_dataframe(df=data, source='video_bout_aggregator data', valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=clf_names + feature_names + p_cols + pose_names)
    bouts = detect_bouts(data_df=data, target_lst=clf_names, fps=sample_rate).sort_values(by="Start_frame")
    feature_results, meta_results, pose_results = [], [], []
    for clf_cnt, clf_name in enumerate(clf_names):
        clf_bouts = bouts[bouts['Event'] == clf_name].reset_index(drop=True)
        if len(bouts) > 0:
            clf_bouts = clf_bouts[clf_bouts["Bout_time"] >= (min_bout_length / 1000)][["Start_frame", "End_frame"]].values
            for clf_bout_id, (start_idx, end_idx) in enumerate(zip(clf_bouts[:, 0], clf_bouts[:, 1])):
                clf_bout = data.iloc[start_idx: end_idx+1]
                if agg_method == "mean":
                    clf_agg_df = pd.DataFrame(clf_bout[feature_names].mean()).T
                    clf_meta_df = pd.DataFrame([clf_bout[f"Probability_{clf_name}"].mean()], columns=['PROBABILITY'])
                else:
                    clf_agg_df = pd.DataFrame(clf_bout[feature_names].median()).T
                    clf_meta_df = pd.DataFrame([clf_bout[f"Probability_{clf_name}"].median()], columns=['PROBABILITY'])
                for df in (clf_agg_df, clf_meta_df):
                    df.insert(0, "BOUT_ID", clf_bout_id)
                    df.insert(0, "CLASSIFIER", clf_name)
                    df.insert(0, "BOUT_END_FRAME", end_idx)
                    df.insert(0, "BOUT_START_FRAME", start_idx)
                    df.insert(0, "VIDEO", name)
                feature_results.append(clf_agg_df)
                meta_results.append(clf_meta_df)

    feature_results = pd.concat(feature_results, axis=0).reset_index(drop=True)
    meta_results = pd.concat(meta_results, axis=0).reset_index(drop=True)
    pose_results = data[pose_names]
    pose_results.insert(0, "VIDEO", name)
    feature_results = feature_results.reset_index(drop=True).set_index(['BOUT_ID', "CLASSIFIER", "BOUT_END_FRAME", "BOUT_START_FRAME", "VIDEO"])
    meta_results = meta_results.reset_index(drop=True).set_index(['BOUT_ID', "CLASSIFIER", "BOUT_END_FRAME", "BOUT_START_FRAME", "VIDEO"])
    timer.stop_timer()
    if verbose:
        print(f'Bout statistics for video {name} complete. Elapsed time: {timer.elapsed_time_str}s')
    return (feature_results, meta_results, pose_results)

def directory_clf_bout_aggregator(dir: Union[str, os.PathLike],
                                  save_path: Union[str, os.PathLike],
                                  feature_names: List[str],
                                  clf_names: Union[str, List[str]],
                                  pose_names: List[str],
                                  video_info: Union[str, os.PathLike, pd.DataFrame],
                                  min_bout_length: Optional[float] = None,
                                  core_cnt: int = -1,
                                  verbose: bool = True,
                                  agg_method: Literal["mean", "median"] = "mean") -> None:

    """
    Aggregate classifier bout-level data from multiple files in a directory.

    :param Union[str, os.PathLike] dir: Path to the directory containing CSV files with machine learning classifier results.
    :param Union[str, os.PathLike] save_path: Path to save the aggregated data as a pickle file.
    :param feature_names: List of feature names to include in the aggregation.
    :param clf_names: Classifier names to include in the aggregation. Can be a single string or a list of strings.
    :param List[str] pose_names: Pose-estimation body-part column names to include in the aggregation.
    :param video_info: Path to a CSV file or a DataFrame containing metadata about the videos. This file is used to match classifier results with video (i.e., FPS/sample rate).
    :param Optional[float] min_bout_length: Minimum length (in seconds) of a bout to be included in the aggregation.  If None, no filtering is applied.
    :param int core_cnt: Number of CPU cores to use for parallel processing.  Use -1 to automatically select the maximum available cores.
    :param bool verbose: If True, outputs progress and status messages during processing.
    :param Literal["mean", "median"] agg_method: ggregation method to use for summarizing bout data. Options are "mean" or "median".
    :return: None. The aggregated data is saved to the specified save_path as a pickle file.
    :rtype: None

    :example:
    >>> feature_names = list(pd.read_csv(r"C:\troubleshooting\nastacia_unsupervised\feature_names.csv", usecols=['FEATURE_NAMES']).values[:, 0])
    >>> clf_names = ['Attack', 'Escape',	'Defensive', 'anogenital_prediction',	'face',	'body']
    >>> bp_names = list(pd.read_csv(r"C:\troubleshooting\nastacia_unsupervised\bp_names.csv", usecols=['BP_NAMES']).values[:, 0])
    >>> directory_clf_bout_aggregator(dir=r'C:\troubleshooting\nastacia_unsupervised\machine_results\machine_results', feature_names=feature_names, clf_names=clf_names, pose_names=bp_names, video_info=r"C:\troubleshooting\nastacia_unsupervised\video_info.csv", save_path=r"C:\troubleshooting\nastacia_unsupervised\datasets\data.pickle")
    """

    timer = SimbaTimer(start=True)
    check_valid_lst(source='video_bout_aggregator feature_names', valid_dtypes=(str,), min_len=1, data=feature_names)
    check_valid_lst(source='video_bout_aggregator pose_names', valid_dtypes=(str,), min_len=1, data=pose_names)
    if isinstance(clf_names, list):
        check_valid_lst(source='video_bout_aggregator clf_names', valid_dtypes=(str,), min_len=1, data=clf_names)
    else:
        check_str(name='video_bout_aggregator clf_names', value=clf_names)
        clf_names = [clf_names]
    if min_bout_length is not None:
        check_float(name='video_bout_aggregator min_bout_length', value=min_bout_length, min_value=0)
    else:
        min_bout_length = 0
    check_str(name='video_bout_aggregator agg_method', value=agg_method, options=Methods.AGG_METHODS.value)
    if isinstance(video_info, str): video_info = read_video_info_csv(file_path=video_info)
    check_if_dir_exists(in_dir=os.path.dirname(save_path))
    check_int(name=f'{directory_clf_bout_aggregator.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
    check_valid_boolean(value=[verbose], source='video_bout_aggregator verbose')
    check_valid_lst(source='video_bout_aggregator pose_names', valid_dtypes=(str,), min_len=1, data=pose_names)
    core_cnt = [find_core_cnt()[0] if core_cnt -1 or core_cnt > find_core_cnt()[0] else find_core_cnt()[0]][0]
    file_paths = find_files_of_filetypes_in_directory(directory=dir, extensions=['.csv'], raise_error=True)
    check_all_file_names_are_represented_in_video_log(video_info_df=video_info, data_paths=file_paths)
    file_names = [get_fn_ext(filepath=x)[1] for x in file_paths]
    sample_rates = [read_video_info(video_name=x, video_info_df=video_info)[2] for x in file_names]
    results = Parallel(n_jobs=core_cnt, verbose=1, backend="loky")(delayed(video_clf_bout_aggregator)(data=i,
                                                                                                      feature_names=feature_names,
                                                                                                      clf_names=clf_names,
                                                                                                      pose_names=pose_names,
                                                                                                      sample_rate=j,
                                                                                                      min_bout_length=min_bout_length,
                                                                                                      verbose=verbose,
                                                                                                      agg_method=agg_method) for i, j in zip(file_paths, sample_rates))
    get_reusable_executor().shutdown(wait=True)
    feature_results = pd.concat([x[0] for x in results], axis=0)
    metadata_results = pd.concat([x[1] for x in results], axis=0)
    pose_results = pd.concat([x[2] for x in results], axis=0)
    write_pickle(data={'POSE': pose_results, 'FEATURES': feature_results, 'META': metadata_results}, save_path=save_path)
    timer.stop_timer()
    stdout_success(msg=f"Data saved at {save_path}", elapsed_time=timer.elapsed_time_str)



# name: str,
# feature_names: List[str],
# clf_names: Union[str, List[str]],
# pose_names: Union[str, List[str]],
# sample_rate: float,
# min_bout_length: Optional[float] = None,
# agg_method: Literal["mean", "median"] = "mean") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

# feature_names = list(pd.read_csv(r"C:\troubleshooting\nastacia_unsupervised\feature_names.csv", usecols=['FEATURE_NAMES']).values[:, 0])
# clf_names = ['Attack', 'Escape',	'Defensive', 'anogenital_prediction',	'face',	'body']
# bp_names = list(pd.read_csv(r"C:\troubleshooting\nastacia_unsupervised\bp_names.csv", usecols=['BP_NAMES']).values[:, 0])
# # #video_bout_aggregator(data=r"C:\troubleshooting\nastacia_unsupervised\machine_results\machine_results\Box2_IF19_7_20211109T173625_4.csv", name='Box2_IF19_7_20211109T173625_4', feature_names=feature_names, clf_names=clf_names, sample_rate=25, pose_names=bp_names)
# # #
# # #
# directory_clf_bout_aggregator(dir=r'C:\troubleshooting\nastacia_unsupervised\machine_results\machine_results',
#                               feature_names=feature_names,
#                               clf_names=clf_names,
#                               pose_names=bp_names,
#                               video_info=r"C:\troubleshooting\nastacia_unsupervised\video_info.csv",
#                               save_path=r"C:\troubleshooting\nastacia_unsupervised\datasets\data.pickle")
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
    #
    #
    #
    #
    #
    #
    #
    # check_instance(
    #     source=f"{bout_aggregator.__name__} data",
    #     instance=data,
    #     accepted_types=(pd.DataFrame,),
    # )
    # check_instance(
    #     source=f"{bout_aggregator.__name__} clfs", instance=clfs, accepted_types=(list,)
    # )
    # check_valid_lst(
    #     data=clfs,
    #     source=f"{bout_aggregator.__name__} clfs",
    #     valid_dtypes=(str,),
    #     min_len=1,
    # )
    # check_instance(
    #     source=f"{bout_aggregator.__name__} video_info",
    #     instance=video_info,
    #     accepted_types=(pd.DataFrame,),
    # )
    # check_int(
    #     name=f"{bout_aggregator.__name__} min_bout_length",
    #     value=min_bout_length,
    #     min_value=0,
    # )
    # check_str(
    #     name=f"{bout_aggregator.__name__} aggregator",
    #     value=aggregator,
    #     options=("MEAN", "MEDIAN"),
    # )
    #
    # def bout_aggregator_mp(frms, data, clf_name):
    #     bout_df = data.iloc[frms[0] : frms[1] + 1]
    #     bout_video, start_frm, end_frm = (
    #         bout_df["VIDEO"].values[0],
    #         bout_df["FRAME"].values[0],
    #         bout_df["FRAME"].values[-1],
    #     )
    #     if aggregator == "MEAN":
    #         agg_df = pd.DataFrame(bout_df[feature_names].mean()).T
    #         agg_df["PROBABILITY"] = bout_df[f"Probability_{clf_name}"].mean()
    #     elif aggregator == "MEDIAN":
    #         agg_df = pd.DataFrame(bout_df[feature_names].median()).T
    #         agg_df["PROBABILITY"] = bout_df[f"Probability_{clf_name}"].median()
    #     agg_df["CLASSIFIER"] = clf_name
    #     agg_df.insert(0, "END_FRAME", end_frm)
    #     agg_df.insert(0, "START_FRAME", start_frm)
    #     agg_df.insert(0, "VIDEO", bout_video)
    #     return agg_df
    #
    # output = []
    # for cnt, video in enumerate(data["VIDEO"].unique()):
    #     print(
    #         f'Processing video {video} ({str(cnt+1)}/{str(len(data["VIDEO"].unique()))})...'
    #     )
    #     video_df = data[data["VIDEO"] == video].reset_index(drop=True)
    #     for clf in clfs:
    #         _, _, fps = read_video_info(vid_info_df=video_info, video_name=video)
    #         bouts = detect_bouts(
    #             data_df=video_df, target_lst=[clf], fps=fps
    #         ).sort_values(by="Start_frame")
    #         bouts = bouts[bouts["Bout_time"] >= min_bout_length / 1000][
    #             ["Start_frame", "End_frame"]
    #         ].values
    #         if len(bouts) > 0:
    #             bouts = [x.tolist() for x in bouts]
    #             results = Parallel(n_jobs=core_cnt, verbose=0, backend="loky")(
    #                 delayed(bout_aggregator_mp)(j, video_df, clf) for j in bouts
    #             )
    #             results = pd.concat(results, axis=0).sort_values(
    #                 by=["VIDEO", "START_FRAME"]
    #             )
    #             output.append(results)
    # get_reusable_executor().shutdown(wait=True)
    # timer.stop_timer()
    # stdout_success(
    #     msg="Bout aggregation statistics complete!", elapsed_time=timer.elapsed_time_str
    # )
    # return pd.concat(output, axis=0).reset_index(drop=True)

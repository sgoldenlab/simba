__author__ = "Simon Nilsson"

from typing import List, Optional

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

from simba.utils.checks import (check_instance, check_int, check_str,
                                check_valid_lst)
from simba.utils.data import detect_bouts
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import find_core_cnt, read_video_info


def bout_aggregator(
    data: pd.DataFrame,
    clfs: List[str],
    feature_names: List[str],
    video_info: pd.DataFrame,
    min_bout_length: Optional[int] = 0,
    aggregator: Optional[Literal["MEAN", "MEDIAN"]] = "MEAN",
) -> pd.DataFrame:
    """
    Helper to aggregate features to bout-level representations for unsupervised analysis.

    :param pd.DataFrame data: DataFrame with features.
    :param List[str] clfs: Names of classifiers
    :param feature_names: Names of features
    :param Optional[Literal['MEAN', 'MEDIAN']] aggregator: Aggregation type, e.g., 'MEAN', 'MEDIAN'. Default 'MEAN'.
    :param Optional[int] min_bout_length: The length of the shortest allowed bout in milliseconds. Default 0 which means all bouts.
    :param pd.DataFrame video_info: Dataframe holding video names, fps, resolution etc typically located at project_folder/logs/video_info.csv of SimBA project.
    :return pd.DataFrame: Featurized data at aggregate bout level.
    """

    timer = SimbaTimer(start=True)
    core_cnt = find_core_cnt()[1]
    print("Calculating bout aggregate statistics...")
    check_instance(
        source=f"{bout_aggregator.__name__} data",
        instance=data,
        accepted_types=(pd.DataFrame,),
    )
    check_instance(
        source=f"{bout_aggregator.__name__} clfs", instance=clfs, accepted_types=(list,)
    )
    check_valid_lst(
        data=clfs,
        source=f"{bout_aggregator.__name__} clfs",
        valid_dtypes=(str,),
        min_len=1,
    )
    check_instance(
        source=f"{bout_aggregator.__name__} video_info",
        instance=video_info,
        accepted_types=(pd.DataFrame,),
    )
    check_int(
        name=f"{bout_aggregator.__name__} min_bout_length",
        value=min_bout_length,
        min_value=0,
    )
    check_str(
        name=f"{bout_aggregator.__name__} aggregator",
        value=aggregator,
        options=("MEAN", "MEDIAN"),
    )

    def bout_aggregator_mp(frms, data, clf_name):
        bout_df = data.iloc[frms[0] : frms[1] + 1]
        bout_video, start_frm, end_frm = (
            bout_df["VIDEO"].values[0],
            bout_df["FRAME"].values[0],
            bout_df["FRAME"].values[-1],
        )
        if aggregator == "MEAN":
            agg_df = pd.DataFrame(bout_df[feature_names].mean()).T
            agg_df["PROBABILITY"] = bout_df[f"Probability_{clf_name}"].mean()
        elif aggregator == "MEDIAN":
            agg_df = pd.DataFrame(bout_df[feature_names].median()).T
            agg_df["PROBABILITY"] = bout_df[f"Probability_{clf_name}"].median()
        agg_df["CLASSIFIER"] = clf_name
        agg_df.insert(0, "END_FRAME", end_frm)
        agg_df.insert(0, "START_FRAME", start_frm)
        agg_df.insert(0, "VIDEO", bout_video)
        return agg_df

    output = []
    for cnt, video in enumerate(data["VIDEO"].unique()):
        print(
            f'Processing video {video} ({str(cnt+1)}/{str(len(data["VIDEO"].unique()))})...'
        )
        video_df = data[data["VIDEO"] == video].reset_index(drop=True)
        for clf in clfs:
            _, _, fps = read_video_info(vid_info_df=video_info, video_name=video)
            bouts = detect_bouts(
                data_df=video_df, target_lst=[clf], fps=fps
            ).sort_values(by="Start_frame")
            bouts = bouts[bouts["Bout_time"] >= min_bout_length / 1000][
                ["Start_frame", "End_frame"]
            ].values
            if len(bouts) > 0:
                bouts = [x.tolist() for x in bouts]
                results = Parallel(n_jobs=core_cnt, verbose=0, backend="loky")(
                    delayed(bout_aggregator_mp)(j, video_df, clf) for j in bouts
                )
                results = pd.concat(results, axis=0).sort_values(
                    by=["VIDEO", "START_FRAME"]
                )
                output.append(results)
    get_reusable_executor().shutdown(wait=True)
    timer.stop_timer()
    stdout_success(
        msg="Bout aggregation statistics complete!", elapsed_time=timer.elapsed_time_str
    )
    return pd.concat(output, axis=0).reset_index(drop=True)

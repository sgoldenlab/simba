import os
from copy import deepcopy
from typing import List, Literal, Optional, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import pandas as pd

from simba.utils.checks import (check_instance, check_int, check_str,
                                check_valid_dataframe, check_valid_lst)
from simba.utils.data import detect_bouts, read_df
from simba.utils.enums import Formats
from simba.utils.errors import InvalidInputError
from simba.utils.printing import SimbaTimer
from simba.utils.read_write import find_core_cnt, read_video_info


def video_bout_aggregator(data: Union[str, os.PathLike, pd.DataFrame],
                          clfs: List[str],
                          feature_names: List[str],
                          sample_rate: int,
                          min_bout_length: Optional[int] = None,
                          method: Optional[Literal["MEAN", "MEDIAN"]] = "MEAN") -> pd.DataFrame:

    check_valid_lst(data=clfs, source=f"{video_bout_aggregator.__name__} clfs", valid_dtypes=(str,), min_len=1)
    check_valid_lst(data=feature_names, source=f"{video_bout_aggregator.__name__} feature_names", valid_dtypes=(str,), min_len=1)
    check_instance(source=f'{video_bout_aggregator.__name__} data', accepted_types=(str, pd.DataFrame), instance=data)
    if isinstance(data, (str, os.PathLike)):
        df = read_df(file_path=data_path, file_type='csv', usecols=feature_names + clfs)
    elif isinstance(data, (pd.DataFrame)):
        df = deepcopy(data)
    else:
        raise InvalidInputError(msg=f'data is of invalid type: {type(df)}, accepted: {str, os.PathLike, pd.DataFrame}', source=video_bout_aggregator.__name__)
    check_valid_dataframe(df=data, source=f"{video_bout_aggregator.__name__} data", valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=feature_names + clfs)
    check_int(name=f"{video_bout_aggregator.__name__} data", value=sample_rate, min_value=10e-6)
    if min_bout_length is not None:
        check_int(name=f"{video_bout_aggregator.__name__} min_bout_length", value=min_bout_length, min_value=0)
    else:
        min_bout_length = 0
    check_str(name=f"{video_bout_aggregator.__name__} method", value=method, options=("MEAN", "MEDIAN"))


# timer = SimbaTimer(start=True)
    # core_cnt = find_core_cnt()[1]
    # print("Calculating bout aggregate statistics...")

    # check_valid_dataframe(df=data, source=f"{video_bout_aggregator.__name__} data", required_fields=feature_names + clfs, valid_dtypes=Formats.NUMERIC_DTYPES.value)
    # check_valid_dataframe(df=video_info, source=f"{video_bout_aggregator.__name__} video_info", required_fields=['fps', 'video'], valid_dtypes=Formats.NUMERIC_DTYPES.value)
    # if min_bout_length is not None:
    #     check_int(name=f"{video_bout_aggregator.__name__} min_bout_length", value=min_bout_length, min_value=0)
    # check_str(name=f"{video_bout_aggregator.__name__} aggregator", value=aggregator, options=("MEAN", "MEDIAN"))
    # _, _, fps = read_video_info(vid_info_df=video_info, video_name=video)
    #
    #
    # detect_bouts(data_df=data, target_lst=clfs)
    #
    #
    # for cnt, video in enumerate(data["VIDEO"].unique()):
    #     print(f'Processing video {video} ({str(cnt+1)}/{str(len(data["VIDEO"].unique()))})...')



data_path = '/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/csv/input_csv/501_MA142_Gi_CNO_0521.csv'

video_bout_aggregator(data=data_path)

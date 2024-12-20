from typing import List

import pandas as pd

from simba.utils.checks import check_valid_lst


def check_all_dfs_in_list_has_same_cols(dfs: List[pd.DataFrame], raise_error: bool = True) -> bool:
    check_valid_lst(data=dfs, source=check_all_dfs_in_list_has_same_cols.__name__, valid_dtypes=(pd.DataFrame,), min_len=1)
    col_headers = [list(x.columns) for x in dfs]
    common_headers = set(col_headers[0]).intersection(*col_headers[1:])
    all_headers = set(item for sublist in col_headers for item in sublist)
    missing_headers = list(all_headers - common_headers)
    if len(missing_headers) > 0:
        if raise_error:
            raise MissingColumnsError(msg=f"The data in project_folder/csv/targets_inserted directory do not contain the same headers. Some files are missing the headers: {missing_headers}", source=self.__class__.__name__)
        else:
            return False
    return True
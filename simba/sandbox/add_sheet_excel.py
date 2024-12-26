import os
from itertools import combinations, product
from typing import List, Union

import numpy as np
import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict, check_str,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_lst)
from simba.utils.enums import Formats
from simba.utils.errors import CountError, DuplicationError
from simba.utils.printing import stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_pickle)

ADJ_MUTUAL_INFO = "adjusted mutual information"
FOWLKES_MALLOWS = "fowlkes mallows"
ADJ_RAND_INDEX = "adjusted rand index"
STATS_OPTIONS = [ADJ_MUTUAL_INFO, FOWLKES_MALLOWS, ADJ_RAND_INDEX]


excel_file = pd.ExcelFile('/Users/simon/Desktop/asdasd.xlsx')


def create_empty_xlsx_file(xlsx_path: Union[str, os.PathLike]):
    """
    Create an empty MS Excel file.
    :param Union[str, os.PathLike] xlsx_path: Path where to save MS Excel file on disk.
    """
    check_if_dir_exists(in_dir=os.path.dirname(xlsx_path))
    df.to_excel(xlsx_path, index=False)



def df_to_xlsx_sheet(xlsx_path: Union[str, os.PathLike],
                     df: pd.DataFrame,
                     sheet_name: str) -> None:

    """
    Append a dataframe as a new sheet in an MS Excel file.

    :param Union[str, os.PathLike] xlsx_path: Path to an existing MS Excel file on disk.
    :param pd.DataFrame df: A dataframe to save as a sheet in the MS Excel file.
    :param str sheet_name: Name of the sheet to save the dataframe under.
    """

    check_file_exist_and_readable(file_path=xlsx_path)
    check_valid_dataframe(df=df, source=df_to_xlsx_sheet.__name__)
    check_str(name=f'{df_to_xlsx_sheet} sheet_name', value=sheet_name, allow_blank=False)
    excel_file = pd.ExcelFile(xlsx_path)
    if sheet_name in excel_file.sheet_names:
        raise DuplicationError(msg=f'Sheet name {sheet_name} already exist in file {xlsx_path} with sheetnames: {excel_file.sheet_names}', source=df_to_xlsx_sheet.__name__)
    with pd.ExcelWriter(xlsx_path, mode='a') as writer:
        df.to_excel(writer, sheet_name=sheet_name)





df = pd.DataFrame(np.arange(0, 100), columns=['blah'])


df_to_xlsx_sheet(xlsx_path='/Users/simon/Desktop/asdasd.xlsx',df=df, sheet_name='asdasd')
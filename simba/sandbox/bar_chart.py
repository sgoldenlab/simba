import os
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from simba.utils.checks import (check_if_dir_exists, check_instance, check_str,
                                check_valid_lst)
from simba.utils.enums import Formats


def plot_bar_chart(df: pd.DataFrame,
                   x: str,
                   y: str,
                   error: Optional[str] = None,
                   x_label: Optional[str] = None,
                   y_label: Optional[str] = None,
                   title: Optional[str] = None,
                   fig_size: Optional[Tuple[int, int]] = (10, 8),
                   palette: Optional[str] = 'magma',
                   save_path: Optional[Union[str, os.PathLike]] = None):

    check_instance(source=f"{plot_bar_chart.__name__} df", instance=df, accepted_types=(pd.DataFrame))
    check_str(name=f"{plot_bar_chart.__name__} x", value=x, options=tuple(df.columns))
    check_str(name=f"{plot_bar_chart.__name__} y", value=y, options=tuple(df.columns))
    check_valid_lst(data=list(df[y]), source=f"{plot_bar_chart.__name__} y", valid_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_lst(data=list(df[x]), source=f"{plot_bar_chart.__name__} x", valid_dtypes=Formats.NUMERIC_DTYPES.value)
    fig, ax = plt.subplots(figsize=fig_size)
    sns.barplot(x=x, y=y, data=df, palette=palette, ax=ax)
    if error is not None:
        check_str(name=f"{plot_bar_chart.__name__} error", value=error, options=tuple(df.columns))
        check_valid_lst(data=list(df[error]), source=f"{plot_bar_chart.__name__} error",valid_dtypes=Formats.NUMERIC_DTYPES.value)
        for i, (value, error) in enumerate(zip(df['FEATURE_IMPORTANCE_MEAN'], df['FEATURE_IMPORTANCE_STDEV'])):
            plt.errorbar(i, value, yerr=(0, error), fmt='o', color='grey', capsize=1)

    if x_label is not None:
        check_str(name=f"{plot_bar_chart.__name__} x_label", value=x_label)
        plt.xlabel(x_label)
    if y_label is not None:
        check_str(name=f"{plot_bar_chart.__name__} y_label", value=y_label)
        plt.ylabel(y_label)
    if title is not None:
        check_str(name=f"{plot_bar_chart.__name__} title", value=title)
        plt.title(title, ha="center", fontsize=15)
    if save_path is not None:
        check_str(name=f"{plot_bar_chart.__name__} save_path", value=save_path)
        check_if_dir_exists(in_dir=os.path.dirname(save_path))
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
    else:
        return fig





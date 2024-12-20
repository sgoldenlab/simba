import os
from typing import Optional, Union
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from simba.utils.checks import check_instance, check_str, check_if_dir_exists


def line_plot(df: pd.DataFrame,
              x: str,
              y: str,
              x_label: Optional[str] = None,
              y_label: Optional[str] = None,
              title: Optional[str] = None,
              save_path: Optional[Union[str, os.PathLike]] = None):

    check_instance(source=f'{line_plot.__name__} df', instance=df, accepted_types=(pd.DataFrame))
    check_str(name=f'{line_plot.__name__} x', value=x, options=tuple(df.columns))
    check_str(name=f'{line_plot.__name__} y', value=y, options=tuple(df.columns))
    sns.set_style("whitegrid", {"grid.linestyle": "--"})
    plot = sns.lineplot(data=df, x=x, y=y)

    if x_label is not None:
        check_str(name=f'{line_plot.__name__} x_label', value=x_label)
        plt.xlabel(x_label)
    if y_label is not None:
        check_str(name=f'{line_plot.__name__} y_label', value=y_label)
        plt.ylabel(y_label)
    if title is not None:
        check_str(name=f'{line_plot.__name__} title', value=title)
        plt.title(title, ha="center", fontsize=15)
    if save_path is not None:
        check_str(name=f'{line_plot.__name__} save_path', value=save_path)
        check_if_dir_exists(in_dir=os.path.dirname(save_path))
        plot.figure.savefig(save_path)
        plt.close("all")
    else:
        return plot






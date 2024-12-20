from typing import Union, Optional
try:
    from typing import Literal
except:
    from typing_extensions import Literal

import seaborn as sns
import os
import matplotlib.pyplot as plt
import pandas as pd
from simba.utils.lookups import get_named_colors
from simba.utils.printing import stdout_success



def violin_plot(data: pd.DataFrame,
                x: str,
                y: str,
                save_path: Union[str, os.PathLike],
                font_rotation: Optional[int] = 45,
                font_size: Optional[int] = 10,
                img_size: Optional[tuple] = (13.7, 8.27),
                cut: Optional[int] = 0,
                scale: Optional[Literal["area", "count", "width"]] = "area"):

    named_colors = get_named_colors()
    palette = {}
    for cnt, violin in enumerate(sorted(list(data[x].unique()))):
        palette[violin] = named_colors[cnt]
    plt.figure()
    order = data.groupby(by=[x])[y].median().sort_values().iloc[::-1].index
    figure_FSCTT = sns.violinplot(x=x, y=y, data=data, cut=cut, scale=scale, order=order, palette=palette)
    figure_FSCTT.set_xticklabels(figure_FSCTT.get_xticklabels(), rotation=font_rotation, size=font_size)
    figure_FSCTT.figure.set_size_inches(img_size)
    figure_FSCTT.figure.savefig(save_path, bbox_inches="tight")
    stdout_success(msg=f"Violin plot saved at {save_path}")

pd.read_csv

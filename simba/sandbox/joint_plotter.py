import os
from typing import Union, Optional, List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from simba.utils.checks import (check_valid_array, check_instance, check_str, check_that_column_exist, check_if_dir_exists)
from simba.utils.lookups import get_categorical_palettes
from simba.utils.errors import InvalidInputError
import warnings
warnings.filterwarnings("ignore")


from simba.utils.read_write import read_pickle


def joint_plot(data: Union[np.ndarray, pd.DataFrame],
               columns: Optional[List[str]] = ('X', 'Y', 'Cluster'),
               palette: Optional[str] = 'Set1',
               kind: Optional[str] = 'scatter',
               size: Optional[int] = 10,
               title: Optional[str] = None,
               save_path: Optional[Union[str, os.PathLike]] = None):

    """
    :example:
    >>> x = np.hstack([np.random.normal(loc=10, scale=4, size=(100, 2)), np.random.randint(0, 1, size=(100, 1))])
    >>> y = np.hstack([np.random.normal(loc=25, scale=4, size=(100, 2)), np.random.randint(1, 2, size=(100, 1))])
    >>> plot = joint_plot(data=np.vstack([x, y]), columns=['X', 'Y', 'Cluster'], title='The plot')

    >>> data = read_pickle(data_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/clusters/ghostly_banzai.pickle')
    >>> embedding_data = data['DR_MODEL']['MODEL'].embedding_
    >>> labels = data['CLUSTER_MODEL']['MODEL'].labels_
    >>> data = np.hstack([embedding_data, labels.reshape(-1, 1)])
    >>> plot = joint_plot(data=data)
    """

    cmaps = get_categorical_palettes()
    if palette not in cmaps: raise InvalidInputError(msg=f'{palette} is not a valid palette. Accepted options: {cmaps}', source=joint_plot.__name__)
    check_instance(source=f'{joint_plot.__name__} data' , instance=data, accepted_types=(np.ndarray, pd.DataFrame))
    check_str(name=f'{joint_plot.__name__} kind', value=kind, options=('kde', 'reg', 'hist', 'scatter'))
    if isinstance(data, pd.DataFrame):
        check_that_column_exist(df=data, column_name=columns, file_name=joint_plot.__name__)
        data = data[list(columns)]
    else:
        check_valid_array(data=data, source=joint_plot.__name__, accepted_ndims=(2,), max_axis_1=len(columns), min_axis_1=len(columns))
        data = pd.DataFrame(data, columns=list(columns))

    sns.set_palette(palette)
    pct_x = np.percentile(data[columns[0]].values, 10)
    pct_y = np.percentile(data[columns[1]].values, 10)
    plot = sns.jointplot(data=data,
                  x=columns[0],
                  y=columns[1],
                  hue=columns[2],
                  xlim=(data[columns[0]].min() - pct_x, data[columns[0]].max() + pct_x),
                  ylim=(data[columns[1]].min() - pct_y, data[columns[1]].max() + pct_y),
                  palette=sns.color_palette(palette, len(data[columns[2]].unique())),
                  kind=kind,
                  marginal_ticks=False,
                  s=size)

    if title is not None:
        plot.fig.suptitle(title, va='baseline', ha='center', fontsize=15, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 0})
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path))
        plot.savefig(save_path)
        plt.close("all")
    else:
        return plot

def categorical_scatter(data: Union[np.ndarray, pd.DataFrame],
                        columns: Optional[List[str]] = ('X', 'Y', 'Cluster'),
                        palette: Optional[str] = 'Set1',
                        size: Optional[int] = 10,
                        title: Optional[str] = None,
                        save_path: Optional[Union[str, os.PathLike]] = None):

    cmaps = get_categorical_palettes()
    if palette not in cmaps: raise InvalidInputError(msg=f'{palette} is not a valid palette. Accepted options: {cmaps}.', source=joint_plot.__name__)
    check_instance(source=f'{joint_plot.__name__} data' , instance=data, accepted_types=(np.ndarray, pd.DataFrame))
    if isinstance(data, pd.DataFrame):
        check_that_column_exist(df=data, column_name=columns, file_name=joint_plot.__name__)
        data = data[list(columns)]
    else:
        check_valid_array(data=data, source=joint_plot.__name__, accepted_ndims=(2,), max_axis_1=len(columns), min_axis_1=len(columns))
        data = pd.DataFrame(data, columns=list(columns))

    pct_x = np.percentile(data[columns[0]].values, 25)
    pct_y = np.percentile(data[columns[1]].values, 25)
    plt.xlim(data[columns[0]].min() - pct_x, data[columns[0]].max() + pct_x)
    plt.ylim(data[columns[1]].min() - pct_y, data[columns[1]].max() + pct_y)

    plot = sns.scatterplot(data=data,
                           x=columns[0],
                           y=columns[1],
                           hue=columns[2],
                           palette=sns.color_palette(palette, len(data[columns[2]].unique())),
                           s=size)
    if title is not None:
        plt.title(title, ha="center", fontsize=15, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 0})
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path))
        plt.savefig(save_path)
        plt.close("all")
    else:
        return plot

def continuous_scatter(data: Union[np.ndarray, pd.DataFrame],
                       columns: Optional[List[str]] = ('X', 'Y', 'Cluster'),
                       palette: Optional[str] = 'Set1',
                       size: Optional[int] = 10,
                       title: Optional[str] = None,
                       save_path: Optional[Union[str, os.PathLike]] = None):

    check_instance(source=f'{joint_plot.__name__} data', instance=data, accepted_types=(np.ndarray, pd.DataFrame))
    if isinstance(data, pd.DataFrame):
        check_that_column_exist(df=data, column_name=columns, file_name=joint_plot.__name__)
        data = data[list(columns)]
    else:
        check_valid_array(data=data, source=joint_plot.__name__, accepted_ndims=(2,), max_axis_1=len(columns), min_axis_1=len(columns))
        data = pd.DataFrame(data, columns=list(columns))

    fig, ax = plt.subplots()
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plot = ax.scatter(data[columns[0]], data[columns[1]], c=data[columns[2]], s=size, cmap=palette)
    cbar = fig.colorbar(plot)
    cbar.set_label(columns[2], loc="center")
    if title is not None:
        plt.title(title, ha="center", fontsize=15, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 0})
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path))
        fig.savefig(save_path)
        plt.close("all")
    else:
        return plot

data = read_pickle(data_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/clusters/beautiful_beaver.pickle')
embedding_data = data['DR_MODEL']['MODEL'].embedding_
#labels = data['DATA']['BOUTS_TARGETS']['CLASSIFIER'].values

labels = data['CLUSTER_MODEL']['MODEL'].labels_
data = pd.DataFrame(np.hstack([embedding_data, labels.reshape(-1, 1)]), columns=['X', 'Y', 'Cluster'])
#joint_plot(data=data, save_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/cluster_vis/beautiful_beaver.png', palette='tab20', size=20, title='beautiful_beaver', kind='scatter')

#categorical_scatter(data=data, save_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/cluster_vis/beautiful_beaver.png', palette='tab20', size=20, title='beautiful_beaver')


continuous_scatter(data=data, save_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/cluster_vis/beautiful_beaver.png', palette='magma', size=20, title='beautiful_beaver')


# x = np.hstack([np.random.normal(loc=10, scale=4, size=(100, 2)), np.random.randint(0, 3, size=(100, 1))])
# y = np.hstack([np.random.normal(loc=25, scale=4, size=(100, 2)), np.random.randint(3, 9, size=(100, 1))])
# data = pd.DataFrame(np.vstack([x, y]), columns=['X', 'Y', 'Cluster'])



#plot = continuous_scatter(data=np.vstack([x, y]), columns=['X', 'Y', 'Cluster'], title='The plot')

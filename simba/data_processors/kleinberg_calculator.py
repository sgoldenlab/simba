__author__ = "Simon Nilsson"

import glob
import os
import shutil
from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from simba.data_processors.pybursts_calculator import kleinberg_burst_detection
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_float, check_if_dir_exists,
                                check_if_filepath_list_is_empty, check_int,
                                check_that_column_exist)
from simba.utils.enums import Paths, TagNames
from simba.utils.printing import log_event, stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_df
from simba.utils.warnings import KleinbergWarning


class KleinbergCalculator(ConfigReader):
    """
    Smooth classification data using the Kleinberg burst detection algorithm.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/kleinberg_filter.md>`__.

    .. image:: _static/img/kleinberg.png
       :width: 400
       :align: center

    :param str config_path: path to SimBA project config file in Configparser format
    :param List[str] classifier_names: Classifier names to apply Kleinberg smoothing to.
    :param float sigma: Burst detection sigma value. Higher sigma values and fewer, longer, behavioural bursts will be recognised. Default: 2.
    :param float gamma: Burst detection gamma value. Higher gamma values and fewer behavioural bursts will be recognised. Default: 0.3.
    :param int hierarchy: Burst detection hierarchy level. Higher hierarchy values and fewer behavioural bursts will to be recognised. Default: 1.
    :param bool hierarchical_search: See `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/kleinberg_filter.md#hierarchical-search-example>`_ Default: False.
    :param Optional[Union[str, os.PathLike]] input_dir: The directory with files to perform kleinberg smoothing on. If None, defaults to `project_folder/csv/machine_results`
    :param Optional[Union[str, os.PathLike]] output_dir: Location to save smoothened data in. If None, defaults to `project_folder/csv/machine_results`

    :example I:
    >>> kleinberg_calculator = KleinbergCalculator(config_path='MySimBAConfigPath', classifier_names=['Attack'], sigma=2, gamma=0.3, hierarchy=2, hierarchical_search=False)
    >>> kleinberg_calculator.run()

    :example 2:
    >>> output_dir = r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/kleinberg_gridsearch_test'
    >>> input_dir = r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/kleinberg_gridsearch_test'
    >>> kleinberg_calculator = KleinbergCalculator(config_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', classifier_names=['Attack', 'Sniffing', 'Rear'], sigma=2, gamma=0.3, hierarchy=3, hierarchical_search=False, input_dir=input_dir, output_dir=output_dir)

    References
    ----------

    .. [1] Kleinberg, Bursty and Hierarchical Structure in Streams, `Data Mining and Knowledge Discovery`,
           vol. 7, pp. 373–397, 2003.
    .. [2] Lee et al., Temporal microstructure of dyadic social behavior during relationship formation in mice, `PLOS One`,
           2019.
    .. [3] Bordes et al., Automatically annotated motion tracking identifies a distinct social behavioral profile
           following chronic social defeat stress, `bioRxiv`, 2022.
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        classifier_names: List[str],
        sigma: Optional[int] = 2,
        gamma: Optional[float] = 0.3,
        hierarchy: Optional[int] = 1,
        hierarchical_search: Optional[bool] = False,
        input_dir: Optional[Union[str, os.PathLike]] = None,
        output_dir: Optional[Union[str, os.PathLike]] = None,
    ):

        super().__init__(config_path=config_path)
        log_event(
            logger_name=str(self.__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )
        self.hierarchical_search, sigma, gamma, hierarchy, self.output_dir = (
            hierarchical_search,
            float(sigma),
            float(gamma),
            int(hierarchy),
            output_dir,
        )
        check_float(value=sigma, name="sigma", min_value=1.01)
        check_float(value=gamma, name="gamma", min_value=0)
        check_int(value=hierarchy, name="hierarchy", min_value=0)
        self.sigma, self.gamma, self.hierarchy, self.clfs = (
            float(sigma),
            float(gamma),
            float(hierarchy),
            classifier_names,
        )
        if input_dir is None:
            self.data_paths = self.machine_results_paths
            self.output_dir = self.machine_results_dir
            check_if_filepath_list_is_empty(
                filepaths=self.machine_results_paths,
                error_msg=f"SIMBA ERROR: No data files found in {self.machine_results_dir}. Cannot perform Kleinberg smoothing",
            )
            original_data_files_folder = os.path.join(
                self.project_path,
                Paths.MACHINE_RESULTS_DIR.value,
                f"Pre_Kleinberg_{self.datetime}",
            )
            if not os.path.exists(original_data_files_folder):
                os.makedirs(original_data_files_folder)
            for file_path in self.machine_results_paths:
                _, file_name, ext = get_fn_ext(file_path)
                shutil.copyfile(
                    file_path, os.path.join(original_data_files_folder, file_name + ext)
                )
        else:
            check_if_dir_exists(in_dir=input_dir)
            self.data_paths = glob.glob(input_dir + f"/*.{self.file_type}")
            check_if_filepath_list_is_empty(
                filepaths=self.data_paths,
                error_msg=f"SIMBA ERROR: No data files found in {input_dir}. Cannot perform Kleinberg smoothing",
            )
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
        print(
            f"Processing Kleinberg burst detection for {len(self.data_paths)} file(s)..."
        )

    def hierarchical_searcher(self):
        if (len(self.kleinberg_bouts["Hierarchy"]) == 1) and (
            int(self.kleinberg_bouts.at[0, "Hierarchy"]) == 0
        ):
            self.clf_bouts_in_hierarchy = self.kleinberg_bouts
        else:
            results = []
            kleinberg_df = deepcopy(self.kleinberg_bouts)
            kleinberg_df.loc[kleinberg_df["Hierarchy"] == 0, "Hierarchy"] = np.inf
            kleinberg_df["prior_hierarchy"] = kleinberg_df["Hierarchy"].shift(1)
            kleinberg_df["hierarchy_difference"] = (
                kleinberg_df["Hierarchy"] - kleinberg_df["prior_hierarchy"]
            )
            start_idx = list(
                kleinberg_df.index[kleinberg_df["hierarchy_difference"] <= 0]
            )
            end_idx = list([x - 1 for x in start_idx][1:])
            end_idx_2 = list(
                kleinberg_df.index[
                    (kleinberg_df["hierarchy_difference"] == 0)
                    | (kleinberg_df["hierarchy_difference"] > 1)
                ]
            )
            end_idx.extend((end_idx_2))
            for start, end in zip(start_idx, end_idx):
                hierarchies_in_bout = kleinberg_df.loc[start:end]
                target_hierarchy_in_hierarchies_bout = hierarchies_in_bout[
                    hierarchies_in_bout["Hierarchy"] == self.hierarchy
                ]
                if len(target_hierarchy_in_hierarchies_bout) == 0:
                    for lower_hierarchy in list(
                        range(int(self.hierarchy - 1.0), -1, -1)
                    ):
                        lower_hierarchy_in_hierarchies_bout = hierarchies_in_bout[
                            hierarchies_in_bout["Hierarchy"] == lower_hierarchy
                        ]
                        if len(lower_hierarchy_in_hierarchies_bout) > 0:
                            target_hierarchy_in_hierarchies_bout = (
                                lower_hierarchy_in_hierarchies_bout
                            )
                            break
                if len(target_hierarchy_in_hierarchies_bout) > 0:
                    results.append(target_hierarchy_in_hierarchies_bout)
            if len(results) > 0:
                self.clf_bouts_in_hierarchy = pd.concat(results, axis=0).drop(
                    ["prior_hierarchy", "hierarchy_difference"], axis=1
                )
            else:
                self.clf_bouts_in_hierarchy = pd.DataFrame(
                    columns=["Video", "Classifier", "Hierarchy", "Start", "Stop"]
                )

    def run(self):
        detailed_df_lst = []
        for file_cnt, file_path in enumerate(self.data_paths):
            _, video_name, _ = get_fn_ext(file_path)
            print(
                f"Kleinberg analysis video {video_name}. Video {file_cnt+1}/{len(self.data_paths)}..."
            )
            data_df = read_df(file_path, self.file_type).reset_index(drop=True)
            video_out_df = deepcopy(data_df)
            check_that_column_exist(
                df=data_df, column_name=self.clfs, file_name=video_name
            )
            save_path = os.path.join(self.output_dir, video_name + f".{self.file_type}")
            for clf in self.clfs:
                clf_offsets = data_df.index[data_df[clf] == 1].values
                if len(clf_offsets) > 0:
                    video_out_df[clf] = 0
                    self.kleinberg_bouts = pd.DataFrame(
                        kleinberg_burst_detection(
                            offsets=clf_offsets, s=self.sigma, gamma=self.gamma
                        ),
                        columns=["Hierarchy", "Start", "Stop"],
                    )
                    self.kleinberg_bouts["Stop"] += 1
                    self.kleinberg_bouts.insert(loc=0, column="Classifier", value=clf)
                    self.kleinberg_bouts.insert(loc=0, column="Video", value=video_name)
                    detailed_df_lst.append(self.kleinberg_bouts)
                    if self.hierarchical_search:
                        print("Applying hierarchical search...")
                        self.hierarchical_searcher()
                    else:
                        self.clf_bouts_in_hierarchy = self.kleinberg_bouts[
                            self.kleinberg_bouts["Hierarchy"] == self.hierarchy
                        ]
                    hierarchy_idx = list(
                        self.clf_bouts_in_hierarchy.apply(
                            lambda x: list(range(x["Start"], x["Stop"] + 1)), 1
                        )
                    )
                    hierarchy_idx = [x for xs in hierarchy_idx for x in xs]
                    hierarchy_idx = [
                        x for x in hierarchy_idx if x in list(data_df.index)
                    ]
                    video_out_df.loc[hierarchy_idx, clf] = 1

            write_df(video_out_df, self.file_type, save_path)

        self.timer.stop_timer()
        if len(detailed_df_lst) > 0:
            self.detailed_df = pd.concat(detailed_df_lst, axis=0)
            detailed_save_path = os.path.join(
                self.logs_path,
                "Kleinberg_detailed_log_{}.csv".format(str(self.datetime)),
            )
            self.detailed_df.to_csv(detailed_save_path)
            stdout_success(
                msg=f"Kleinberg analysis complete. See {detailed_save_path} for details of detected bouts of all classifiers in all hierarchies",
                elapsed_time=self.timer.elapsed_time_str,
                source=self.__class__.__name__,
            )
        else:
            print("Kleinberg analysis complete.")
            KleinbergWarning(
                msg="All behavior bouts removed following kleinberg smoothing",
                source=self.__class__.__name__,
            )


# test = KleinbergCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                            classifier_names=['Attack'],
#                            sigma=1.1,
#                            gamma=0.3,
#                            hierarchy=5,
#                            hierarchical_search=False)
#
# test.run()
#
# test.perform_kleinberg()
# #data = run_kleinberg(r'Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\project_config.ini', ['int'], sigma=2, gamma=0.3, hierarchy=1)

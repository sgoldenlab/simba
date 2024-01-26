import os
import random
from typing import Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.utils.printing import stdout_success


class GibbsSampler(ConfigReader):
    """
    Gibbs sampling for finding "motifs" in categorical sequences.

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter pd.DataFrame data: Dataframe where each row represents videos and each column represents behavior sequence.
    :parameter float pseudo_number: Small error value for fuzzy search. Default: 0.001.
    :parameter int sequence_length: The length of the motif sequence searched for.
    :parameter int iterations: Number of iterations per epoch. Default: 1000.
    :parameter int epochs: Number of epochs of iterations. Default: 5.
    :parameter int plateau_val: Terminate epoch when error rate has remained unchanged for ``plateau_val`` count of interations.

    :example:
    >>> df = pd.read_csv('/tests/data/gibbs_sample_data/gibbs_sample_ver_7.csv', index_col=0)
    >>> gibbs_sampler = GibbsSampler(data=df)
    >>> gibbs_sampler.run()


    :References:
    .. [1] Lawrence et.al, Detecting Subtle Sequence Signals: a Gibbs Sampling Strategy for Multiple Alignment,
           Science, vol. 262, pp. 208-214, 1993.
    .. [2] `Great YouTube course by Xiaole Shirley Liu <https://www.youtube.com/watch?v=NRjhfyXWHuQ>`_.
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        data: pd.DataFrame,
        pseudo_number: int = 0.001,
        sequence_length: int = 4,
        iterations: int = 1500,
        epochs: int = 2,
        stop_val: float = 0.001,
        plateau_val: int = 50,
    ):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.unique_vals = sorted(list(pd.unique(data.values.ravel("K"))))
        self.target_probability = 1 * sequence_length + (
            pseudo_number * (sequence_length + 1)
        )

        self.holdout_cols = ["H_{}".format(i) for i in range(0, sequence_length)]
        self.non_holdout_cols = ["C_{}".format(i) for i in range(sequence_length)]
        self.output_cols = ["Behavior_{}".format(i + 1) for i in range(sequence_length)]

        self.data, self.pseudo_num, self.sequence_len = (
            data,
            pseudo_number,
            sequence_length,
        )
        self.iterations, self.stop_val = [iterations] * epochs, stop_val
        self.plateau_val = plateau_val
        self.summary_df = pd.DataFrame(columns=self.output_cols)
        self.data.columns = ["F_{}".format(x) for x in range(len(self.data.columns))]
        self.save_path = os.path.join(
            self.logs_path, f"gibbs_sampling_results_{self.datetime}.csv"
        )

    def __get_start_positions(self):
        start_positions = []
        for val in range(len(self.data)):
            start_positions.append(
                random.randint(0, len(self.data.columns) - self.sequence_len)
            )
        return start_positions

    def __make_probability_table(self, df: pd.DataFrame, unique_vals: np.ndarray):
        prob_df = pd.DataFrame(columns=unique_vals)
        for column in df:
            occurance_lst = []
            for value in unique_vals:
                occurance_lst.append(
                    (len(df[df[column] == value]) + self.pseudo_num) / len(df[column])
                    + self.pseudo_num
                )
            prob_df.loc[len(prob_df)] = occurance_lst
        return prob_df

    def __iterate_through_possible_seq_in_hold_out(
        self,
        hoObs: pd.DataFrame,
        bgDf: pd.DataFrame,
        probDf: pd.DataFrame,
        start_hold_no: int,
    ):
        df_holdout_seqs = pd.DataFrame(columns=self.holdout_cols + ["Prob_weight"])
        for startv in range(0, len(hoObs.columns) - self.sequence_len + 1):
            cols = [
                "F_{}".format(ind) for ind in range(startv, startv + self.sequence_len)
            ]
            seq = hoObs.loc[:, cols]
            ptot = 1
            for ind in range(self.sequence_len):
                prob_table_cols = seq.loc[start_hold_no][ind]
                ptot *= probDf.loc[ind][prob_table_cols] / bgDf.loc[prob_table_cols]
            seq["Prob_weight"] = ptot
            seq.columns = self.holdout_cols + ["Prob_weight"]
            df_holdout_seqs = df_holdout_seqs.append(seq, ignore_index=True)
        return df_holdout_seqs

    def __sum_results(self, full_sequence_set, summary_df):
        for i in range(self.sequence_len):
            full_sequence_set = full_sequence_set.rename(
                columns={f"C_{i}": self.output_cols[i]}
            )

        summary_df = pd.concat([summary_df, full_sequence_set], axis=0).reset_index(
            drop=True
        )
        output = (
            summary_df.groupby(summary_df.columns.tolist())
            .size()
            .reset_index()
            .rename(columns={0: "records"})
            .sort_values(by=["records"], ascending=False)
        )
        output["percent"] = output["records"] / output["records"].sum()

        return summary_df, output.reset_index(drop=True)

    def run(self):
        # calculate background constants
        bg_df = self.data.stack().value_counts()
        bg_df = bg_df / bg_df.sum()
        bg_df = bg_df.sort_index()
        full_sequence_set, output = None, None
        prior_error, stable_error_cnt = np.inf, 0

        for iteration_counter, iterations in enumerate(self.iterations):
            holdout_idx = np.random.randint(0, len(self.data) - 2)
            start_positions = self.__get_start_positions()
            for iteration in range(0, iterations):
                hold_out_observation_df = self.data.iloc[[holdout_idx]]
                if iteration == 0:
                    df_it = self.data.drop(hold_out_observation_df.index)
                    idx_lst, seq_lst = [], []
                    for index, row in df_it.iterrows():
                        seq_lst.append(
                            list(
                                row[
                                    start_positions[index] : start_positions[index]
                                    + self.sequence_len
                                ]
                            )
                        )
                        idx_lst.append(index)
                    non_holdout_df = pd.DataFrame(
                        seq_lst, columns=self.non_holdout_cols, index=idx_lst
                    )
                else:
                    non_holdout_df = full_sequence_set.loc[
                        full_sequence_set.index != holdout_idx
                    ]

                probability_df = self.__make_probability_table(
                    non_holdout_df, self.unique_vals
                )
                df_holdout_seqs = self.__iterate_through_possible_seq_in_hold_out(
                    hold_out_observation_df, bg_df, probability_df, holdout_idx
                )
                new_seq = df_holdout_seqs.sample(1, weights="Prob_weight")
                new_seq = new_seq[self.holdout_cols]
                new_seq.columns = self.non_holdout_cols
                new_seq.index = [holdout_idx]
                full_sequence_set = pd.concat([new_seq, non_holdout_df])
                error = self.target_probability - probability_df.max(axis=1).sum()
                # self.summary_df, output = self.__sum_results(full_sequence_set, self.summary_df, iteration_counter)
                print(
                    f"Current error: {error}, iteration: {iteration}, epoch: {iteration_counter}"
                )
                if error <= self.stop_val:
                    print(f"Convergence reached in {iteration} iterations")
                    break
                if error == prior_error:
                    stable_error_cnt += 1
                    if stable_error_cnt >= self.plateau_val - 1:
                        print(f"Platea reached {self.plateau_val}")
                        break
                else:
                    stable_error_cnt = 0
                prior_error = error
                if holdout_idx < (len(self.data) - 1):
                    holdout_idx += 1
                else:
                    holdout_idx = 0
            self.summary_df, output = self.__sum_results(
                full_sequence_set, self.summary_df
            )

        output.to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(
            msg=f"Gibbs sampling results saved @{self.save_path}",
            elapsed_time=self.timer.elapsed_time_str,
        )


# df = pd.read_csv('/Users/simon/Desktop/envs/simba_dev/tests/data/gibbs_sample_data/gibbs_sample_7.csv', index_col=0)
# test = GibbsSampler(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', data=df)
# test.run()

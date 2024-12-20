import os
from typing import Tuple, Union

import numpy as np
import pandas as pd

from simba.utils.checks import (check_float, check_if_dir_exists, check_int,
                                check_valid_array)
from simba.utils.printing import SimbaTimer, stdout_success


class GibbSampler():
    def __init__(self,
                 data: np.ndarray,
                 save_path: Union[str, os.PathLike],
                 sequence_length: int = 4,
                 iterations: int = 1500,
                 epochs: int = 2,
                 stop_val: float = 0.001,
                 pseudo_number: float = 10e-6,
                 plateau_val: int = 50):

        """
        Gibbs sampling for finding "motifs" in categorical sequences.

        :param np.ndarray data: Two dimensional array where observations are organised by row and each sequential sample in the observation is organized by column.
        :param Union[str, os.PathLike] save_path: The path location where to save the CSV results.
        :param int sequence_length: The length of the motif sequence searched for.
        :param int iterations: Number of iterations per epoch. Default: 1500.
        :param int epochs: Number of epochs of iterations. Default: 4.
        :param float stop_val: Terminate once the error value reaches below this threshold. Default 0.001.
        :param int plateau_val: Terminate epoch when error rate has remained unchanged for ``plateau_val`` count of iterations. Default 50.
        :param float pseudo_number: Small error value for fuzzy search and avoid division by zero errors. Default: 10e-6.

        :example:
        >>> data = pd.read_csv(r"/Users/simon/Desktop/envs/simba/simba/tests/data/sample_data/gibbs_sample_cardinal.csv", index_col=0).values
        >>> sampler = GibbSampler(data=data, save_path=r'/Users/simon/Desktop/gibbs.csv', epochs=5, iterations=600)
        >>> sampler.run()

        :references:
           .. [1] Lawrence et.al, Detecting Subtle Sequence Signals: a Gibbs Sampling Strategy for Multiple Alignment, Science, vol. 262, pp. 208-214, 1993.
           .. [2] Great YouTube tutorial / explanation by Xiaole Shirley Liu - `https://www.youtube.com/watch?v=NRjhfyXWHuQ <https://www.youtube.com/watch?v=NRjhfyXWHuQ>`_.
           .. [3] Weinreb et al. Keypoint-MoSeq: parsing behavior by linking point tracking to pose dynamics, Nature Methods, 21, 1329–1339 (2024).
        """


        check_valid_array(data=data, source=f'{self.__class__.__name__} data', accepted_ndims=(2,))
        check_float(name=f'{self.__class__.__name__} stop_val', value=stop_val, min_value=0.0)
        check_float(name=f'{self.__class__.__name__} pseudo_number', value=pseudo_number, min_value=10e-16)
        check_int(name=f'{self.__class__.__name__} epochs', value=epochs, min_value=1)
        check_int(name=f'{self.__class__.__name__} iterations', value=iterations, min_value=1)
        check_int(name=f'{self.__class__.__name__} sequence_length', value=sequence_length, min_value=2)
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=f'{self.__class__.__name__} save_path')
        self.unique_vals = np.sort(np.unique(data.flatten()))
        self.target_probability = 1 * sequence_length + (pseudo_number * (sequence_length + 1))
        self.holdout_fields = [f"H_{i}" for i in range(sequence_length)]
        self.non_holdout_cols = [f"C_{i}" for i in range(sequence_length)]
        self.out_cols = [f"Behavior_{i+1}" for i in range(sequence_length)]
        self.summary_df = pd.DataFrame(columns=self.out_cols)
        self.data, self.pseudo_num, self.sequence_len, self.plateau_val = (data, pseudo_number, sequence_length, plateau_val)
        self.iterations, self.stop_val, self.epochs = [iterations] * epochs, stop_val, epochs
        self.save_path = save_path

    def __make_probability_table(self, data: np.ndarray) -> pd.DataFrame:
        prob_df = pd.DataFrame(columns=self.unique_vals)
        for field_idx in range(data.shape[1]):
            idx_data = data[:, field_idx].flatten()
            for unique_val in self.unique_vals:
                val_idx = np.argwhere(idx_data == unique_val).flatten()
                pct = (val_idx.shape[0] + self.pseudo_num) / (idx_data.shape[0] + self.pseudo_num)
                prob_df.loc[field_idx, unique_val] = pct
        return prob_df

    def __iterate_through_possible_seq_in_hold_out(self,
                                                   holdout_obs: np.ndarray,
                                                   bg_dict: dict,
                                                   probability_df: pd.DataFrame) -> pd.DataFrame:

        out_df = pd.DataFrame(columns=self.holdout_fields + ["weight"])
        for r in range(self.sequence_len, holdout_obs.shape[0]+1):
            sequence = holdout_obs[r - self.sequence_len:r]
            prob_tot = 1
            for i in range(self.sequence_len):
                val = sequence[i]
                prob_tot *= probability_df.loc[i][val] / bg_dict[val]
            out_df.loc[len(out_df)] = np.append(sequence, prob_tot)
        return out_df

    def __sum_results(self, full_sequence_set, summary_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        full_sequence_set = pd.DataFrame(full_sequence_set, columns=self.out_cols)
        summary_df = pd.concat([summary_df, full_sequence_set], axis=0).reset_index(drop=True)
        output = (summary_df.groupby(summary_df.columns.tolist()).size().reset_index().rename(columns={0: "records"}).sort_values(by=["records"], ascending=False))
        output["percent"] = output["records"] / output["records"].sum()
        return summary_df, output.reset_index(drop=True)

    def run(self):
        timer = SimbaTimer(start=True)
        unique_elements, counts = np.unique(self.data, return_counts=True)
        counts_dict = dict(zip(unique_elements, counts))
        bg_dict = {k: (v / np.sum(counts)) for k, v in counts_dict.items()}
        prior_error, stable_error_cnt, full_sequence, output = np.inf, 0, None, None

        for epoch_cnt, iterations in enumerate(self.iterations):
            holdout_idx = np.random.randint(0, self.data.shape[0]-1)
            for iteration in range(0, iterations):
                hold_out_obs = self.data[holdout_idx]
                if iteration == 0:
                    start_pos = np.random.randint(low=0, high=data.shape[1] - self.sequence_len, size=(data.shape[0]))
                    end_pos = np.array([x + self.sequence_len for x in start_pos])
                    slices = np.array([[x, y] for x, y in zip(start_pos, end_pos)])
                    epoch_sample = np.delete(self.data, [holdout_idx], axis=0)
                    sequences = np.full(shape=(epoch_sample.shape[0], self.sequence_len), fill_value='-1.0')
                    for idx in range(epoch_sample.shape[0]):
                        i = slices[idx]
                        sequences[idx] = epoch_sample[idx][i[0]: i[1]]
                else:
                    sequences = np.delete(full_sequence, [holdout_idx], axis=0)
                probability_df = self.__make_probability_table(sequences)
                df_holdout_seqs = self.__iterate_through_possible_seq_in_hold_out(hold_out_obs, bg_dict, probability_df)
                new_sequence = df_holdout_seqs.sample(1, weights="weight")[self.holdout_fields].values[0]
                full_sequence = np.insert(sequences, 0, new_sequence, axis=0)
                error = round(self.target_probability - probability_df.max(axis=1).sum(), 4)
                print(f"Current error: {error}, iteration: {iteration+1}/{iterations}, epoch: {epoch_cnt+1}/{self.epochs}")
                if error <= self.stop_val:
                    print(f"Convergence reached. Error: {error}, Stop value: {self.stop_val}")
                    break
                if error == prior_error:
                    stable_error_cnt += 1
                    if stable_error_cnt >= self.plateau_val - 1:
                        print(f"Plateau reached. Stable error count: {stable_error_cnt}. Plateau value: {self.plateau_val}")
                        break
                else:
                    stable_error_cnt = 0
                prior_error = error
                if holdout_idx < (len(self.data) - 1):
                    holdout_idx += 1
                else:
                    holdout_idx = 0
                self.summary_df, output = self.__sum_results(full_sequence, self.summary_df)

        output.to_csv(self.save_path)
        timer.stop_timer()
        stdout_success(msg=f"Gibbs sampling results saved @{self.save_path}", elapsed_time=timer.elapsed_time_str)



# data = pd.read_csv(r"C:\projects\simba\simba\tests\data\sample_data\gibbs_sample_cardinal.csv", index_col=0).values
# sampler = GibbSampler(data=data, save_path=r'C:\Users\sroni\OneDrive\Desktop\gibbs.csv', iterations=5)
# sampler.run()

# data = pd.read_csv(r"/Users/simon/Desktop/envs/simba/simba/tests/data/sample_data/gibbs_sample_cardinal.csv", index_col=0).values
# sampler = GibbSampler(data=data, save_path=r'/Users/simon/Desktop/gibbs.csv', epochs=5, iterations=600)
# sampler.run()
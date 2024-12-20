import os
from typing import Union

import numpy as np
import pandas as pd


class GibbSampler():



    def __init__(self,
                 data: np.ndarray,
                 save_path: Union[str, os.PathLike],
                 sequence_length: int = 4,
                 iterations: int = 1500,
                 epochs: int = 2,
                 stop_val: float = 0.001,
                 pseudo_number: float = 10e-6,
                 plateau_val: int = 50,):


        self.unique_vals = np.unique(data.flatten())
        self.target_p = 1 * sequence_length + (pseudo_number * (sequence_length + 1))
        self.holdout_fields = [f"H_{i}" for i in range(sequence_length)]
        self.non_holdout_cols = [f"C_{i}" for i in range(sequence_length)]
        self.out_cols = [f"Behavior_{i+1}" for i in range(sequence_length)]
        self.data, self.pseudo_num, self.sequence_len, self.plateau_val = (data, pseudo_number, sequence_length, plateau_val)
        self.epochs, self.iterations, self.stop_val = epochs, iterations, stop_val








    def run(self):
        unique_elements, counts = np.unique(self.data, return_counts=True)
        counts_dict = dict(zip(unique_elements, counts))
        counts_dict = {k: (v / np.sum(counts)) for k, v in counts_dict.items()}

        for epoch in range(self.epochs):
            holdout_idx = np.random.randint(0, self.data.shape[0]-1)
            start_pos = np.random.randint(low=0, high=data.shape[1]-self.sequence_len, size=(data.shape[0]))
            end_pos = np.array([x+self.sequence_len for x in start_pos])
            slices = np.array([[x, y] for x, y in zip(start_pos, end_pos)])
            hold_out_obs = self.data[holdout_idx]
            epoch_sample = np.delete(self.data, [holdout_idx], axis=0)
            sequences = np.full(shape=(epoch_sample.shape[0], self.sequence_len), fill_value=np.str_)
            for idx in range(epoch_sample.shape[0]):
                i = slices[idx]
                sequences[idx] = epoch_sample[idx][i[0]: i[1]]
            for iteration in range(0, self.iterations):
                probability_df = self.__make_probability_table(epoch_sample, self.unique_vals)




            #
            # df_it = self.data.drop(hold_out_observation_df.index)
            #
            #
            # for it in range(self.iterations):
            #
            #     if it == 0:
            #



data = pd.read_csv(r"C:\projects\simba\simba\tests\data\sample_data\gibbs_sample_cardinal.csv", index_col=0).values
sampler = GibbSampler(data=data, save_path=r'C:\Users\sroni\OneDrive\Desktop\gibbs.csv', iterations=5)
sampler.run()
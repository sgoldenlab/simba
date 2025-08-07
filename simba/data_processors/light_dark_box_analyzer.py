import argparse
import os
from itertools import groupby
from typing import Optional, Union

import numpy as np
import pandas as pd

from simba.utils.checks import (check_float, check_if_dir_exists, check_str,
                                check_valid_dataframe)
from simba.utils.data import detect_bouts
from simba.utils.enums import Formats
from simba.utils.printing import SimbaTimer
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    seconds_to_timestamp, write_df)

pd.options.mode.chained_assignment = None

COLUMN_NAMES = ['EVENT', 'START TIME (S)', 'END TIME (S)', 'START FRAME', 'END FRAME', 'DURATION (S)']
OUT_COL_NAMES= ['VIDEO', 'BODY-PART', 'EVENT', 'START TIME (S)', 'END TIME (S)', 'START FRAME', 'END FRAME', 'DURATION (S)']

class LightDarkBoxAnalyzer():
    """
    Perform light–dark box analysis using DeepLabCut pose estimation data.

    .. note::
       If the specified body-part is detected below the specified threshold, then the animal is the dark-box. Otherwise its in the light-box.

    .. seealso::
       For light/dark box data plotting, see :func:`simba.plotting.light_dark_box_plotter.LightDarkBoxPlotter`.

    This class analyzes animal transitions between light and dark compartments  in a light–dark box behavioral test based on the availability of pose-estimation data. It assumes that if pose-estimation for a specified body part is available,
    the animal is in the light box; otherwise, the animal is in the dark box. It detects bouts (continuous time segments) for each condition and saves the bout-level data to a CSV file.

    :param Union[str, os.PathLike] data_dir: Directory containing DeepLabCut CSV files with pose-estimation data.
    :param Union[str, os.PathLike] save_path: Full path to save the resulting CSV file with light/dark bouts.
    :param str body_part: The name of the body part used to infer the animal's presence.
    :param Union[int, float] fps: Frames per second of the video recordings.
    :param float threshold: Value between 0 and 1. If below this value, animal is in dark box. If above, animal is in light box.

    :example:
    >>> python light_dark_box_analyzer.py --data_dir 'D:\light_dark_box\project_folder\csv\input_csv' --save_path "D:\light_dark_box\project_folder\csv\results\light_dark_data.csv" --body_part nose --fps 29 --threshold 0.01
    >>> analyzer = LightDarkBoxAnalyzer(data_dir='C:\troubleshooting\two_black_animals_14bp\dlc_test', save_path="C:\troubleshooting\two_black_animals_14bp\light_dark_ex\light_dark_data.csv", body_part='Nose_1', fps=30.2)
    >>> analyzer = LightDarkBoxAnalyzer(data_dir='D:\light_dark_box\project_folder\csv\input_csv\test', save_path="C:\troubleshooting\two_black_animals_14bp\light_dark_ex\light_dark_data.csv", body_part='nose', fps=29)
    >>> analyzer.run()
    >>> analyzer.save()

    :references:
       .. [1] For discussion about the development, see - `GitHub issue 446 <https://github.com/sgoldenlab/simba/issues/446#issuecomment-2930692735>`_.
    """


    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 body_part: str,
                 fps: Union[int, float],
                 threshold: float = 0.01,
                 minimum_episode_duration: float = 10e-16,
                 save_path: Optional[Union[str, os.PathLike]] = None):

        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'], raise_error=True, as_dict=True)
        if save_path is not None:
            check_if_dir_exists(in_dir=os.path.dirname(save_path))
        check_str(name=f'{self.__class__.__name__} body_part', value=body_part)
        check_float(name=f'{self.__class__.__name__} fps', value=fps, min_value=10e-6)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        check_float(name=f'{self.__class__.__name__} minimum_duration', value=minimum_episode_duration, min_value=0.0)
        self.bp_cols = [f'{body_part}_x', f'{body_part}_y', f'{body_part}_p']
        self.save_path, self.body_part, self.fps, self.threshold, self.min_dur = save_path, body_part, fps, threshold, minimum_episode_duration
        self.file_cnt = len(list(self.data_paths.keys()))

    def _remove_outliers(self, df: pd.DataFrame):
        outlier_sequences = [g for g in (list(map(lambda x: x[1], grp)) for _, grp in groupby(enumerate(df.index[df['DURATION (S)'] < self.min_dur]), lambda x: x[0] - x[1]))]
        for outlier_sequence in outlier_sequences:
            start = df.loc[outlier_sequence[0]] if 0 in outlier_sequence else df.loc[outlier_sequence[0] - 1]
            end = df.loc[outlier_sequence[-1]]
            start['END TIME (S)'] = end['END TIME (S)']
            start['END FRAME'] = end['END FRAME']
            start['DURATION (S)'] = ((start['END FRAME'] - start['START FRAME']) + 1) / self.fps
            df = df.drop(index=outlier_sequence)
            df.loc[start.name] = start
        df['group'] = (df['EVENT'] != df['EVENT'].shift()).cumsum()
        df = (df.groupby(['group', 'EVENT', 'VIDEO', 'BODY-PART'], as_index=False).agg({'START TIME (S)': 'first', 'END TIME (S)': 'last', 'START FRAME': 'first', 'END FRAME': 'last', 'DURATION (S)': 'sum'})).drop(columns='group')
        df = df[OUT_COL_NAMES]
        df['START TIME (HH:MM:SS)'] = seconds_to_timestamp(seconds=list(df['START TIME (S)']))
        df['END TIME (HH:MM:SS)'] = seconds_to_timestamp(seconds=list(df['END TIME (S)']))
        return df

    def run(self):
        self.results = []
        for file_cnt, (file_name, file_path) in enumerate(self.data_paths.items()):
            video_timer = SimbaTimer(start=True)
            self.data_df = pd.read_csv(file_path).reset_index(drop=True).iloc[:, 1:]
            headers = self.data_df.iloc[0].unique()
            cols = np.array([np.array([f'{x}_x', f'{x}_y', f'{x}_p']) for x in headers]).flatten()
            self.data_df = self.data_df.iloc[2:].reset_index(drop=True).apply(pd.to_numeric, errors='coerce').fillna(0)
            self.data_df.columns = cols
            check_valid_dataframe(df=self.data_df, valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.bp_cols)
            self.bp_p = self.data_df[self.bp_cols[2]]
            self.data_df[self.bp_cols[2]] = (self.data_df[self.bp_cols[2]] >= self.threshold).astype(int)
            light_bouts = detect_bouts(data_df=self.data_df, target_lst=[self.bp_cols[2]], fps=self.fps)
            self.data_df[self.bp_cols[2]] = 1 - self.data_df[self.bp_cols[2]]
            dark_bouts = detect_bouts(data_df=self.data_df, target_lst=[self.bp_cols[2]], fps=self.fps)
            dark_bouts.columns, light_bouts.columns = COLUMN_NAMES, COLUMN_NAMES
            light_bouts['EVENT'], dark_bouts['EVENT'] = 'LIGHT BOX', 'DARK BOX'
            light_bouts['BODY-PART'], dark_bouts['BODY-PART'] = self.body_part, self.body_part
            light_bouts['VIDEO'], dark_bouts['VIDEO'] = file_name, file_name
            light_bouts, dark_bouts = light_bouts[OUT_COL_NAMES], dark_bouts[OUT_COL_NAMES]
            out = pd.concat([light_bouts, dark_bouts], axis=0).reset_index(drop=True).sort_values(by=['VIDEO', 'START TIME (S)']).reset_index(drop=True)
            out = self._remove_outliers(df=out)
            self.results.append(out)
            video_timer.stop_timer()
            print(f'Light-dark analysis complete video {file_name} ({file_cnt+1}/{self.file_cnt})... (elapsed time: {video_timer.elapsed_time_str}s)')
        self.results = pd.concat(self.results, axis=0)
        self.results = self.results.sort_values(by=['VIDEO', 'START TIME (S)']).reset_index(drop=True)


    def save(self):
        write_df(df=self.results, file_type='csv', save_path=self.save_path)
        print(f'COMPLETE: Light-dark box data for {self.file_cnt} video(s) saved at {self.save_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to directory containing DeepLabCut CSV files.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output CSV file with light/dark bout data.')
    parser.add_argument('--body_part', type=str, required=True, help='Name of the body part to use for determining visibility (e.g., "snout", "center").')
    parser.add_argument('--fps', type=float, required=True, help='Frames per second (int or float)')
    parser.add_argument('--threshold', type=float, required=True, help='Deeplabcut probability value. If below this value, animal is in dark box. If above, animal is in light box', default=0.1)
    parser.add_argument('--minimum_episode_duration', type=float, required=True, help='The shortest allowed visit to either the dark or the light box. Used to remove spurrious errors in the DLC tracking.', default=1)
    args = parser.parse_args()

    detector = LightDarkBoxAnalyzer(data_dir=args.data_dir,
                                    save_path=args.save_path,
                                    body_part=args.body_part,
                                    fps=args.fps,
                                    threshold=args.threshold,
                                    minimum_episode_duration=args.minimum_episode_duration)

    detector.run()
    detector.save()





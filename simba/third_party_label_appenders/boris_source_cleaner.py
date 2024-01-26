import os
from typing import Union

import pandas as pd

from simba.utils.checks import check_if_dir_exists, check_that_column_exist
from simba.utils.printing import stdout_success, stdout_warning
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, write_df)

"""
INSTRUCTIONS:
1. Change the DATA_DIR to the directory where you have your BORIS annotation files on line 8.
2. Change the SAVE_DIR to the directory where you want to save your new fixed BORIS files on line 9
3. If the names of the subjects or behavior columns changes, change these two names to the names of the columns.
4. The SETTINGS specifies to example two rules, (i) for every row where the subject is focal and the behavior is invest_no, change the behavior name to focal_invest_no, and 
   (ii) for every row where the subject is stimulus and the behavior is invest_no, change the behavior name to stimulus_invest_no. Change these rules or add more if you have more behaviors or animals.
5. Save the file.
6. Activate the SimBA conda environment and navigate to the directory where the boris_source_cleaner.py is located.
7. Run it by typing python boris_source_cleaner.py
"""

DATA_DIR = r"/Users/simon/Downloads/boris_data"  # Directory with BORIS annotations in CSV format.
SAVE_DIR = r"/Users/simon/Downloads/save_dir"  # Directory where to save the modified BORIS annotations in CSV format.
SUBJECT_COL = "Subject"  # The name of the Subject column in BORIS files. Change this if the animal names are encoded in a different column than *Subject*.
BEHAVIOR_COL = "Behavior"  # The name of the Behavior column in BORIS files. Change this if the behavior names are encoded in a different column than *Behavior*.

SETTINGS = [
    {
        SUBJECT_COL: "focal",
        BEHAVIOR_COL: "invest_no",
        "New_behavior_name": "focal_invest_no",
    },
    {
        SUBJECT_COL: "stimulus",
        BEHAVIOR_COL: "invest_no",
        "New_behavior_name": "stimulus_invest_no",
    },
]

# SUBJECT_COL: The entry in the *Subject* column. For example, animal "Simon" or "stimulus"
# BEHAVIOR_COL: The entry in the *Behavior* column. For example, "behavior1" or "invest_no"
# New_behavior_name: For rows where the subjects column and behavior column critera filled, change the behavior column this new behavior name.
# E.g., for rows where *Subject* is Simon AND *Behavior* is behavior1, the *Behavior* will be changed to Simon_behavior1.


class BorisSourceCleaner(object):
    """
    Helper to clean BORIS files where behavior with the same name are assigned to different
    animals through the **Subjects** field.

    :parameter Union[str, os.PathLike] data_dir: Directory with BORIS annotations in CSV format.
    :parameter Union[str, os.PathLike] save_dir: Directory to store the modified BORIS annotations in CSV format.
    :parameter dict settings: Rules for how to change the behavior names.

    :example:
    >>> boris_cleaner = BorisSourceCleaner(data_dir='/Users/simon/Downloads/boris_data', save_dir='/Users/simon/Downloads/save_dir', settings=SETTINGS)
    >>> boris_cleaner.run()

    """

    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        save_dir: Union[str, os.PathLike],
        settings: dict,
    ):
        check_if_dir_exists(in_dir=data_dir)
        check_if_dir_exists(in_dir=save_dir)
        self.data_paths = find_files_of_filetypes_in_directory(
            directory=data_dir,
            extensions=[".csv"],
            raise_warning=False,
            raise_error=True,
        )
        self.settings, self.save_dir = settings, save_dir

    def run(self):
        for file_cnt, file_path in enumerate(self.data_paths):
            video_name = get_fn_ext(filepath=file_path)[1]
            save_path = os.path.join(self.save_dir, f"{video_name}.csv")
            df = pd.read_csv(file_path)
            check_that_column_exist(df=df, column_name=SUBJECT_COL, file_name=file_path)
            check_that_column_exist(
                df=df, column_name=BEHAVIOR_COL, file_name=file_path
            )

            for rule in self.settings:
                rule_rows = df.loc[
                    (df[SUBJECT_COL] == rule[SUBJECT_COL])
                    & (df[BEHAVIOR_COL] == rule[BEHAVIOR_COL])
                ]
                rule_rows = rule_rows[rule_rows["Behavior type"] != "POINT"]

                if len(rule_rows) == 0:
                    stdout_warning(
                        msg=f"WARNING: No rows for subject name {rule[SUBJECT_COL]} with behavior name {rule[BEHAVIOR_COL]} found in file {file_path}"
                    )
                    continue

                elif (len(rule_rows) % 2) != 0:
                    stdout_warning(
                        msg=f"WARNING: Found an UNEVEN number of rows for subject name {rule[SUBJECT_COL]} and behavior name {rule[BEHAVIOR_COL]} found in file {file_path}"
                    )

                rule_rows[BEHAVIOR_COL] = rule["New_behavior_name"]
                df.update(rule_rows)

            write_df(df=df, file_type="csv", save_path=save_path)
            stdout_success(
                msg=f"SAVED DATA VIDEO {save_path}. Video {file_cnt+1}/{len(self.data_paths)}"
            )

        stdout_success(f"All data files saved in {self.save_dir}.")


if __name__ == "__main__":
    boris_cleaner = BorisSourceCleaner(
        data_dir=DATA_DIR, save_dir=SAVE_DIR, settings=SETTINGS
    )
    boris_cleaner.run()

__author__ = "Simon Nilsson", "JJ Choong"

from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_file_exist_and_readable,
                                          read_project_path_and_file_type)

from simba.rw_dfs import (read_df,
                          save_df)
from simba.train_model_functions import read_pickle
from simba.drop_bp_cords import (get_fn_ext,
                                 drop_bp_cords)
from copy import deepcopy
import pickle
import warnings
import time
import os
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)

class ValidateModelRunClf(object):
    """
    Class for running a single classifier on a single featurized input file. Results are saved within the
    project_folder/csv/validation directory of the SimBA project.

    Parameters
    ----------
    config_file_path: str
        path to SimBA project config file in Configparser format
    input_file_path: str
        path to file containing features
    clf_path: str
        path to pickled rf sklearn classifier.

    Notes
    -----

    Examples
    -----

    >>> ValidateModelRunClf(config_path=r"MyProjectConfigPath", input_file_path=r"FeatureFilePath", clf_path=r"ClassifierPath")

    """

    def __init__(self,
                 config_path: str,
                 input_file_path: str,
                 clf_path: str):
        self.start_time = time.time()
        self.config = read_config_file(config_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.save_path = os.path.join(self.project_path, 'csv', 'validation')
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)

        check_file_exist_and_readable(input_file_path)
        check_file_exist_and_readable(clf_path)
        _, file_name, _ = get_fn_ext(str(input_file_path))
        _, classifier_name, _ = get_fn_ext(clf_path)
        data_df = read_df(input_file_path, self.file_type)
        output_df = deepcopy(data_df)
        data_df = drop_bp_cords(data_df, config_path)
        clf = read_pickle(file_path=clf_path)
        probability_col_name = 'Probability_{}'.format(classifier_name)

        try:
            prediction = clf.predict_proba(data_df)
        except ValueError as e:
            vals = [int(s) for s in e.args[0].split() if s.isdigit()]
            if len(vals) == 2:
                print(e.args)
                print('SIMBA CLASSIFIER ERROR: Mismatch in the number of features in selected input file ({}) and what is expected from the selected model ({}). The model expects {} features, but the file contains {} features.'.format(file_name, classifier_name, str(vals[0]), str(vals[1])))
            else:
                print(e.args)
            raise ValueError

        try:
            output_df[probability_col_name] = prediction[:, 1]
        except IndexError as e:
            print(e.args)
            print('SIMBA INDEXERROR: Your classifier has not been created properly. See The SimBA GitHub FAQ page for more information and suggested fixes.')
            raise IndexError

        save_filename = os.path.join(self.save_path, file_name + '.' + self.file_type)
        save_df(output_df, self.file_type, save_filename)
        elapsed_time = str(round(time.time() - self.start_time, 2))

        print('SIMBA COMPLETE: Validation predictions generated for "{}" within the project_folder/csv/validation directory (elapsed time: {})'.format(file_name, elapsed_time))
        print('Click on "Generate plot" to inspect classifier probability thresholds. Then proceed to specify threshold and minimum bout length and click on "Validate".')
#
# ValidateModelRunClf(config_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\project_config.ini",
#                                input_file_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_2_black_060320\project_folder\csv\features_extracted\Together_1.csv",
#                                clf_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_2_black_060320\models\Approach.sav")
__author__ = "Simon Nilsson"

import os
import warnings
from copy import deepcopy
from typing import Union

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from simba.mixins.config_reader import ConfigReader
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.enums import TagNames
from simba.utils.printing import log_event, stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_df


class InferenceValidation(ConfigReader, TrainModelMixin):
    """
    Run a single classifier on a single featurized input file. Results are saved within the
    ``project_folder/csv/validation`` directory of the SimBA project.

    :param str config_file_path: path to SimBA project config file in Configparser format
    :param str input_file_path: path to file containing features
    :param str clf_path: path to pickled rf sklearn classifier.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data>`_

    :example:
    >>> InferenceValidation(config_path=r"MyProjectConfigPath", input_file_path=r"FeatureFilePath", clf_path=r"ClassifierPath")

    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        input_file_path: Union[str, os.PathLike],
        clf_path: Union[str, os.PathLike],
    ):

        ConfigReader.__init__(self, config_path=config_path)
        TrainModelMixin.__init__(self)
        log_event(
            logger_name=str(self.__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )
        self.save_path = os.path.join(self.project_path, "csv", "validation")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        check_file_exist_and_readable(input_file_path)
        check_file_exist_and_readable(clf_path)
        _, file_name, _ = get_fn_ext(str(input_file_path))
        _, classifier_name, _ = get_fn_ext(clf_path)
        data_df = read_df(input_file_path, self.file_type)
        output_df = deepcopy(data_df)
        data_df = self.drop_bp_cords(df=data_df)
        clf = self.read_pickle(file_path=clf_path)
        probability_col_name = f"Probability_{classifier_name}"
        output_df[probability_col_name] = self.clf_predict_proba(
            clf=clf, x_df=data_df, model_name=classifier_name, data_path=input_file_path
        )
        save_filename = os.path.join(self.save_path, file_name + "." + self.file_type)
        write_df(output_df, self.file_type, save_filename)
        self.timer.stop_timer()
        stdout_success(
            msg=f'Validation predictions generated for "{file_name}" within the project_folder/csv/validation directory',
            elapsed_time=self.timer.elapsed_time_str,
            source=self.__class__.__name__,
        )
        print(
            'Click on "Interactive probability plot" to inspect classifier probability thresholds. If satisfactory proceed to specify threshold and minimum bout length and click on "Validate" to create video.'
        )


#
# ValidateModelRunClf(config_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\project_config.ini",
#                                input_file_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_2_black_060320\project_folder\csv\features_extracted\Together_1.csv",
#                                clf_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_2_black_060320\models\Approach.sav")

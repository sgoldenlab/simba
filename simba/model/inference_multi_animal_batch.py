__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from typing import Dict, List, Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.model.inference_batch import InferenceBatch
from simba.utils.checks import check_file_exist_and_readable, check_str
from simba.utils.errors import (InvalidInputError, NoDataError,
                                NoFilesFoundError)
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import read_df


class InferenceMultiAnimalBatch(TrainModelMixin, ConfigReader):
    """
    Run a single trained behavior classifier across every animal in a SimBA project, producing per-animal predictions in the output CSVs.

    .. seealso::
       Training counterpart: :class:`simba.model.grid_search_rf.GridSearchRandomForestClassifier` with ``feature_subset_suffix='_animal_<N>'``.

    :param Union[str, os.PathLike] config_path: Path to the SimBA project_config.ini.
    :param str clf_name: Name of the configured classifier to run multi-animal inference for.

    :example:

    >>> InferenceMultiAnimalBatch(config_path=r'/path/project_folder/project_config.ini', clf_name='wing_wave').run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 clf_name: str):

        check_file_exist_and_readable(file_path=config_path)
        check_str(name=f'{self.__class__.__name__} clf_name', value=clf_name)
        ConfigReader.__init__(self, config_path=os.fspath(config_path), create_logger=False)
        TrainModelMixin.__init__(self)

        if self.animal_cnt < 2:
            raise InvalidInputError(
                msg=f'{self.__class__.__name__} requires a project with 2 or more animals (project animal_cnt={self.animal_cnt}). '
                    f'Use simba.model.inference_batch.InferenceBatch for single-animal projects.',
                source=self.__class__.__name__)
        if clf_name not in self.clf_names:
            raise InvalidInputError(
                msg=f'Classifier "{clf_name}" is not configured in the project. Available classifiers: {self.clf_names}.',
                source=self.__class__.__name__)
        if len(self.feature_file_paths) == 0:
            raise NoFilesFoundError(
                msg=f'No feature files found in {self.features_dir}. Run feature extraction before inference.',
                source=self.__class__.__name__)

        model_dict = self.get_model_info(config=self.config, model_cnt=self.clf_cnt)
        clfs_with_valid_models = [v['model_name'] for v in model_dict.values()]
        if clf_name not in clfs_with_valid_models:
            raise NoFilesFoundError(
                msg=f'No valid trained model file is registered for classifier "{clf_name}". '
                    f'Check that "model_path_N" for "{clf_name}" in the [SML settings] section of the project config points to an existing .sav file. '
                    f'Classifiers with valid model files: {clfs_with_valid_models}.',
                source=self.__class__.__name__)

        features_df = read_df(file_path=self.feature_file_paths[0], file_type=self.file_type)
        feature_subset_dict: Dict[str, List[str]] = {}
        for animal_id in range(1, self.animal_cnt + 1):
            suffix = f'_animal_{animal_id}'
            cols = [c for c in features_df.columns if c.endswith(suffix)]
            if len(cols) == 0:
                raise NoDataError(
                    msg=f'No feature columns ending in "{suffix}" found in {self.feature_file_paths[0]}. '
                        f'Multi-animal inference requires features named with "_animal_<N>" suffixes for every animal.',
                    source=self.__class__.__name__)
            feature_subset_dict[suffix[1:]] = cols

        self.clf_name = clf_name
        self.feature_subsets_by_clf = {clf_name: feature_subset_dict}

    def run(self) -> None:
        timer = SimbaTimer(start=True)
        stdout_information(msg=f'Running multi-animal inference for classifier "{self.clf_name}" across {self.animal_cnt} animals...')
        InferenceBatch(config_path=self.config_path, feature_subsets_by_clf=self.feature_subsets_by_clf).run()
        timer.stop_timer()
        stdout_success(msg=f'Multi-animal inference complete for classifier "{self.clf_name}" (elapsed: {timer.elapsed_time_str}s).', source=self.__class__.__name__)


# if __name__ == "__main__":
#     InferenceMultiAnimalBatch(config_path=r"F:\troubleshooting\sophiaa\project_folder\project_config.ini", clf_name="wing_wave").run()

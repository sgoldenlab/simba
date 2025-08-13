import os
from typing import Dict, List, Optional, Union

import numpy as np

from simba.data_processors.interpolate import Interpolate
from simba.data_processors.smoothing import Smoothing
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pose_importer_mixin import PoseImporterMixin
from simba.utils.checks import (check_if_dir_exists,
                                check_if_keys_exist_in_dict, check_int,
                                check_str, check_valid_lst)
from simba.utils.enums import ConfigKey, Formats, TagNames
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (clean_superanimal_topview_filename,
                                    find_all_videos_in_project,
                                    find_files_of_filetypes_in_directory,
                                    get_video_meta_data,
                                    read_dlc_superanimal_h5, write_df)


class SuperAnimalTopViewImporter(PoseImporterMixin, ConfigReader):

    """
    Import SuperAnimal-TopView mouse data to SimBA

    .. note::
       For more information see the DeepLabCutModelZoo-SuperAnimal-TopViewMouse on huggingface https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse

       Trackes 27 body-parts on one or more mice recorded from zenith.

    .. image:: _static/img/superanimal_topview.png
       :width: 150
       :align: center

    :param config_path: path to SimBA project config file in Configparser format
    :param str data_folder: Path to folder containing SuperAnimal data in ``.h5`` format.
    :param List[str] id_lst: Names of animals.
    :param Optional[Dict[str, str]] interpolation_setting: Dict defining the type and method to use to perform interpolation {'type': 'animals', 'method': 'linear'}.
    :param Optional[Dict[str, Union[str, int]]] smoothing_settings: Dictionary defining the pose estimation smoothing method {'time_window': 500, 'method': 'gaussian'}.

    :references:
        .. [1] Ye, Shaokai, Anastasiia Filippova, Jessy Lauer, et al. “SuperAnimal Pretrained Pose Estimation Models for Behavioral Analysis.” Nature Communications 15, no. 1 (2024): 5165. https://doi.org/10.1038/s41467-024-48792-2.
        .. [2] mwmathis lab on huggingface - `https://huggingface.co/mwmathis/ <https://huggingface.co/mwmathis/>`_.

    :example:
    >>> importer = SuperAnimalTopViewImporter(config_path=r"C:\troubleshooting\super_animal_import\project_folder\project_config.ini", data_folder=r'C:\troubleshooting\super_animal_import\data_files', id_lst=['Animal_1'])
    >>> importer.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_folder: Union[str, os.PathLike],
                 id_lst: List[str],
                 interpolation_settings: Optional[Dict[str, str]] = None,
                 smoothing_settings: Optional[Dict[str, Union[int, str]]] = None):


        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        PoseImporterMixin.__init__(self)
        check_if_dir_exists(in_dir=data_folder)
        check_valid_lst(data=id_lst, source=f'{self.__class__.__name__} id_lst', valid_dtypes=(str,), min_len=1)
        if interpolation_settings is not None:
            check_if_keys_exist_in_dict(data=interpolation_settings, key=['method', 'type'], name=f'{self.__class__.__name__} interpolation_settings')
            check_str(name=f'{self.__class__.__name__} interpolation_settings type', value=interpolation_settings['type'], options=('body-parts', 'animals'))
            check_str(name=f'{self.__class__.__name__} interpolation_settings method', value=interpolation_settings['method'], options=('linear', 'quadratic', 'nearest'))
        if smoothing_settings is not None:
            check_if_keys_exist_in_dict(data=smoothing_settings, key=['method', 'time_window'], name=f'{self.__class__.__name__} smoothing_settings')
            check_str(name=f'{self.__class__.__name__} smoothing_settings method', value=smoothing_settings['method'], options=('savitzky-golay', 'gaussian'))
            check_int(name=f'{self.__class__.__name__} smoothing_settings time_window', value=smoothing_settings['time_window'], min_value=1)

        log_event(logger_name=str(__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        self.interpolation_settings, self.smoothing_settings = (interpolation_settings, smoothing_settings)
        self.data_folder, self.id_lst = data_folder, id_lst
        self.import_log_path = os.path.join(self.logs_path, f"data_import_log_{self.datetime}.csv")
        self.video_paths = find_all_videos_in_project(videos_dir=self.video_dir, raise_error=True if len(id_lst) > 1 else False)
        self.input_data_paths = find_files_of_filetypes_in_directory(directory=self.data_folder, extensions=['.h5'], raise_error=True)
        self.data_and_videos_lk = self.link_video_paths_to_data_paths(data_paths=self.input_data_paths, video_paths=self.video_paths, raise_error=True if len(id_lst) > 1 else False, filename_cleaning_func=clean_superanimal_topview_filename)
        self.check_multi_animal_status()
        self.config.set(ConfigKey.GENERAL_SETTINGS.value, ConfigKey.ANIMAL_CNT.value, str(len(self.id_lst)))
        with open(self.config_path, "w") as f: self.config.write(f)
        f.close()
        print(f"Importing {len(list(self.data_and_videos_lk.keys()))} SuperAnimal-TopView H5 file(s)...")

    def _get_expected_column_names(self):
        self.field_names, self.bp_names = [], []
        for animal_id in self.id_lst:
            if len(self.id_lst) == 1:
                animal_field_names = [f"{animal_id}_{s}" for s in Formats.SUPERANIMAL_TOPVIEW_BP_NAMES.value]
            else:
                animal_field_names = Formats.SUPERANIMAL_TOPVIEW_BP_NAMES.value
            self.bp_names.extend((animal_field_names))
            animal_field_names = [f"{s}{suffix}" for s in animal_field_names for suffix in ("_x", "_y", "_p")]
            self.field_names.extend((animal_field_names))
        with open(self.body_parts_path, "w") as f:
            for name in self.bp_names:
                f.write(name + "\n")
        f.close()
        ConfigReader.__init__(self, config_path=self.config_path, read_video_info=False, create_logger=False)
        self.get_body_part_names()
        self.animal_bp_dict = self.create_body_part_dictionary(self.multi_animal_status, self.id_lst, len(self.id_lst), self.x_cols, self.y_cols, self.p_cols, self.clr_lst)

    def run(self):
        self._get_expected_column_names()
        for cnt, (video_name, video_data) in enumerate(self.data_and_videos_lk.items()):
            video_timer = SimbaTimer(start=True)
            self.add_spacer, self.frame_no, self.video_data, self.video_name = (2, 1, video_data, video_name)
            print(f"Processing {video_name} ({cnt+1}/{len(self.input_data_paths)})...")
            self.data_df = read_dlc_superanimal_h5(path=video_data['DATA'], col_names=self.field_names)
            if self.animal_cnt > 1:
                self.initialize_multi_animal_ui(animal_bp_dict=self.animal_bp_dict, video_info=get_video_meta_data(video_data["VIDEO"]), data_df=self.data_df, video_path=video_data["VIDEO"])
                self.multianimal_identification()
            else:
                self.out_df = self.insert_multi_idx_columns(df=self.data_df.fillna(0))
            self.save_path = os.path.join(os.path.join(self.input_csv_dir, f"{self.video_name}.{self.file_type}"))
            self.out_df = self.out_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            self.out_df[self.out_df < 0] = 0
            write_df(df=self.out_df, file_type=self.file_type, save_path=self.save_path, multi_idx_header=True)
            if self.interpolation_settings is not None:
                interpolator = Interpolate(config_path=self.config_path, data_path=self.save_path, type=self.interpolation_settings['type'], method=self.interpolation_settings['method'], multi_index_df_headers=True, copy_originals=False)
                interpolator.run()
            if self.smoothing_settings is not None:
                smoother = Smoothing(config_path=self.config_path, data_path=self.save_path, time_window=self.smoothing_settings['time_window'], method=self.smoothing_settings['method'], multi_index_df_headers=True, copy_originals=False)
                smoother.run()
            video_timer.stop_timer()
            stdout_success(msg=f"Video {video_name} data imported...", elapsed_time=video_timer.elapsed_time_str)
        self.timer.stop_timer()
        stdout_success(msg=f"All SuperAnimal-TopView H5 data files imported to {self.input_csv_dir} directory", elapsed_time=self.timer.elapsed_time_str)


# importer = SuperAnimalTopViewImporter(config_path=r"C:\troubleshooting\super_animal_import\project_folder\project_config.ini",
#                            data_folder=r'C:\troubleshooting\super_animal_import\data_files',
#                            id_lst=['Animal_1'])
# importer.run()
#### MODIFIED FROM @Toshea111 - https://github.com/Toshea111/sleap/blob/develop/docs/notebooks/Convert_HDF5_to_CSV_updated.ipynb
import io
import os
from typing import Any, Dict, List, Union

import h5py
import pandas as pd

from simba.data_processors.interpolate import Interpolate
from simba.data_processors.smoothing import Smoothing
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pose_importer_mixin import PoseImporterMixin
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict, check_int,
                                check_str, check_valid_lst)
from simba.utils.enums import Methods, TagNames
from simba.utils.errors import BodypartColumnNotFoundError
from simba.utils.printing import (SimbaTimer, log_event, stdout_success,
                                  stdout_warning)
from simba.utils.read_write import (clean_sleap_file_name,
                                    find_all_videos_in_project, get_fn_ext,
                                    get_video_meta_data, write_df)


class SLEAPImporterH5(ConfigReader, PoseImporterMixin):
    """
    Importing SLEAP pose-estimation data into SimBA project in ``H5`` format

    :param str config_path: path to SimBA project config file in Configparser format
    :param str data_folder: Path to folder containing SLEAP data in `H5` format.
    :param List[str] id_lst: Animal names. This will be ignored in one animal projects and default to ``Animal_1``.
    :param Optional[Dict[str, str]] interpolation_setting: Dict defining the type and method to use to perform interpolation {'type': 'animals', 'method': 'linear'}.
    :param Optional[Dict[str, Union[str, int]]] smoothing_settings: Dictionary defining the pose estimation smoothing method {'time_window': 500, 'method': 'gaussian'}.


    .. note::
       `Multi-animal import tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Multi_animal_pose.md>`__.

    References
    ----------
    .. [1] Pereira et al., SLEAP: A deep learning system for multi-animal pose tracking, `Nature Methods`,
           2022.

    >>> sleap_h5_importer = SLEAPImporterH5(config_path=r'project_folder/project_config.ini', data_folder=r'data_folder', actor_IDs=['Termite_1', 'Termite_2', 'Termite_3', 'Termite_4', 'Termite_5'], interpolation_settings="Body-parts: Nearest", smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
    >>> sleap_h5_importer.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_folder: Union[str, os.PathLike],
                 id_lst: List[str],
                 interpolation_settings: Union[str, None],
                 smoothing_settings: Union[Dict[str, Any], None]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        PoseImporterMixin.__init__(self)
        log_event(logger_name=str(__class__.__name__),log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        check_file_exist_and_readable(file_path=config_path)
        check_if_dir_exists(in_dir=data_folder, source=self.__class__.__name__)
        check_valid_lst(data=id_lst, source=f'{self.__class__.__name__} id_lst', valid_dtypes=(str,), min_len=1)
        if interpolation_settings is not None:
            check_if_keys_exist_in_dict(data=interpolation_settings, key=['method', 'type'], name=f'{self.__class__.__name__} interpolation_settings')
            check_str(name=f'{self.__class__.__name__} interpolation_settings type', value=interpolation_settings['type'], options=('body-parts', 'animals'))
            check_str(name=f'{self.__class__.__name__} interpolation_settings method', value=interpolation_settings['method'], options=('linear', 'quadratic', 'nearest'))
        if smoothing_settings is not None:
            check_if_keys_exist_in_dict(data=smoothing_settings, key=['method', 'time_window'], name=f'{self.__class__.__name__} smoothing_settings')
            check_str(name=f'{self.__class__.__name__} smoothing_settings method', value=smoothing_settings['method'], options=('savitzky-golay', 'gaussian'))
            check_int(name=f'{self.__class__.__name__} smoothing_settings time_window', value=smoothing_settings['time_window'], min_value=1)

        self.interpolation_settings, self.smoothing_settings = interpolation_settings, smoothing_settings
        self.data_folder, self.id_lst = data_folder, id_lst
        self.import_log_path = os.path.join(self.logs_path, f"data_import_log_{self.datetime}.csv")
        self.video_paths = find_all_videos_in_project(videos_dir=self.video_dir)
        self.input_data_paths = self.find_data_files(dir=self.data_folder, extensions=[".h5"])
        if self.pose_setting is Methods.USER_DEFINED.value:
            self.__update_config_animal_cnt()
        if self.animal_cnt > 1:
            self.data_and_videos_lk = self.link_video_paths_to_data_paths(data_paths=self.input_data_paths, video_paths=self.video_paths, filename_cleaning_func=clean_sleap_file_name)
            self.check_multi_animal_status()
            self.animal_bp_dict = self.create_body_part_dictionary(self.multi_animal_status,self.id_lst,self.animal_cnt,self.x_cols,self.y_cols,self.p_cols,self.clr_lst)
            if self.pose_setting is Methods.USER_DEFINED.value:
                self.update_bp_headers_file(update_bp_headers=True)
        else:
            self.data_and_videos_lk = dict([(get_fn_ext(file_path)[1], {"DATA": file_path, "VIDEO": None}) for file_path in self.input_data_paths])
        log_event(logger_name=str(self.__class__.__name__), log_type="info", msg=f"Importing data files: {self.input_data_paths}. Interpolation: {interpolation_settings}, Smoothing: {smoothing_settings}")
        print(f"Importing {len(list(self.data_and_videos_lk.keys()))} file(s)...")

    def run(self):
        for file_cnt, (video_name, video_data) in enumerate(self.data_and_videos_lk.items()):
            self.output_filename = clean_sleap_file_name(filename=video_name)
            print(f"Importing {self.output_filename}...")
            video_timer = SimbaTimer(start=True)
            self.video_name = self.output_filename
            with h5py.File(video_data["DATA"], "r") as f:
                missing_keys = [x for x in ["tracks", "point_scores", "node_names", "track_names"] if not x in list(f.keys())]
                if missing_keys:
                    stdout_warning(msg=f'{video_data["DATA"]} is not a valid SLEAP H5 file. Missing keys {missing_keys} Skipping {self.output_filename}...')
                    continue
                tracks = f["tracks"][:].T
                point_scores = f["point_scores"][:].T

            csv_rows = []
            n_frames, n_nodes, _, n_tracks = tracks.shape
            print(tracks)
            for frame_ind in range(n_frames):
                csv_row = []
                for track_ind in range(n_tracks):
                    for node_ind in range(n_nodes):
                        for xyp in range(3):
                            #print(tracks[frame_ind])
                            if xyp == 0 or xyp == 1:
                                data = tracks[frame_ind, node_ind, xyp, track_ind]
                            else:
                                data = point_scores[frame_ind, node_ind, track_ind]

                            csv_row.append(f"{data:.3f}")
                csv_rows.append(" ".join(csv_row))
            csv_rows = "\n".join(csv_rows)
            self.data_df = pd.read_csv(io.StringIO(csv_rows), delim_whitespace=True, header=None).fillna(0)
            if len(self.data_df.columns) != len(self.bp_headers):
                raise BodypartColumnNotFoundError(
                    msg=f'The number of body-parts in data file {video_data["DATA"]} do not match the number of body-parts in your SimBA project. '
                    f"The number of of body-parts expected by your SimBA project is {int(len(self.bp_headers) / 3)}. "
                    f'The number of of body-parts contained in data file {video_data["DATA"]} is {int(len(self.data_df.columns) / 3)}. '
                    f"Make sure you have specified the correct number of animals and body-parts in your project.",
                    source=self.__class__.__name__)
            self.data_df.columns = self.bp_headers
            if self.animal_cnt > 1:
                self.initialize_multi_animal_ui(animal_bp_dict=self.animal_bp_dict, video_info=get_video_meta_data(video_data["VIDEO"]), data_df=self.data_df, video_path=video_data["VIDEO"])
                self.multianimal_identification()
            else:
                self.out_df = self.insert_multi_idx_columns(df=self.data_df.fillna(0))

            self.save_path = os.path.join(os.path.join(self.input_csv_dir, f"{self.output_filename}.{self.file_type}"))
            write_df(df=self.out_df, file_type=self.file_type, save_path=self.save_path, multi_idx_header=True)
            if self.interpolation_settings is not None:
                interpolator = Interpolate(config_path=self.config_path, data_path=self.save_path, type=self.interpolation_settings['type'], method=self.interpolation_settings['method'], multi_index_df_headers=True, copy_originals=False)
                interpolator.run()
            if self.smoothing_settings is not None:
                smoother = Smoothing(config_path=self.config_path, data_path=self.save_path, time_window=self.smoothing_settings['time_window'], method=self.smoothing_settings['method'], multi_index_df_headers=True, copy_originals=False)
                smoother.run()
            video_timer.stop_timer()
            stdout_success(msg=f"Video {self.output_filename} data imported...", elapsed_time=video_timer.elapsed_time_str, source=self.__class__.__name__)
        self.timer.stop_timer()
        stdout_success(msg="All SLEAP H5 data files imported", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)



#
#
# test = SLEAPImporterH5(config_path=r"C:\troubleshooting\slp_single_rat\project_folder\project_config.ini",
#                         data_folder=r"C:\troubleshooting\slp_single_rat\h5",
#                         id_lst=['Jarryd'],
#                         interpolation_settings=None,
#                         smoothing_settings = None)
# test.run()



# test = SLEAPImporterH5(config_path="/Users/simon/Desktop/envs/simba/troubleshooting/sleap_two_animals/project_folder/project_config.ini",
#                         data_folder=r'/Users/simon/Desktop/envs/simba/troubleshooting/sleap_two_animals/h5_import',
#                         id_lst=['White', 'Black'],
#                         interpolation_settings={'type': 'body-parts', 'method': 'nearest'},
#                         smoothing_settings = {'time_window': 500, 'method': 'gaussian'})
# test.run()


# test = SLEAPImporterH5(config_path="/Users/simon/Desktop/envs/troubleshooting/sleap_import/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/sleap_import/data_h5',
#                    id_lst=['White', 'Black'],
#                    interpolation_settings= "Body-parts: Nearest", #'"Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'None', 'Parameters': {'Time_window': '200'}})
# test.run()


# test = SLEAPImporterH5(config_path="/Users/simon/Desktop/envs/troubleshooting/sleap_dropped_frms_2/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/sleap_dropped_frms_2/data',
#                    id_lst=['Simon'],
#                    interpolation_settings= 'None', #'"Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'None', 'Parameters': {'Time_window': '200'}})
# test.run()

# test = SLEAPImporterH5(config_path="/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/Termites_5/import_h5',
#                    id_lst=['Simon', 'Nastacia', 'Ben', 'John', 'JJ'],
#                    interpolation_settings="Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
# test.run()
# print('All SLEAP imports complete.')


#
# test = SLEAPImporterH5(config_path="/Users/simon/Desktop/envs/troubleshooting/sleap_dropped_frms/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/sleap_dropped_frms/data',
#                    id_lst=['Simon'],
#                    interpolation_settings= 'None', #'"Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'None', 'Parameters': {'Time_window': '200'}})
# test.run()


# test = SLEAPImporterH5(config_path="/Users/simon/Desktop/envs/troubleshooting/SLEAP_2_Animals_16_body_parts/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/SLEAP_2_Animals_16_body_parts/data/h5',
#                    id_lst=['Simon', 'Nastacia'],
#                    interpolation_settings="Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
# test.run()
# print('All SLEAP imports complete.')


# test = SLEAPImporterH5(config_path="/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/Termites_5/import_h5',
#                    id_lst=['Simon', 'Nastacia', 'Ben', 'John', 'JJ'],
#                    interpolation_settings='None',
#                    smoothing_settings = {'Method': 'None'})
# test.run()
# print('All SLEAP imports complete.')
#
# test = SLEAPImporterH5(config_path="/Users/simon/Desktop/envs/troubleshooting/sleap_cody/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/sleap_cody/import_h5',
#                    actor_IDs=['Nastacia', 'Sam'],
#                    interpolation_settings="Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
# test.import_sleap()
# print('All SLEAP imports complete.')


# test = SLEAPImporterH5(config_path="/Users/simon/Desktop/envs/troubleshooting/sleap_cody/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/sleap_cody/import_h5',
#                    actor_IDs=['Nastacia'],
#                    interpolation_settings="Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
# test.import_sleap()
# # print('All SLEAP imports complete.')

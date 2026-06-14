__author__ = "Simon Nilsson; sronilsson@gmail.com"

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
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
from simba.utils.errors import CountError
from simba.utils.printing import (SimbaTimer, log_event, stdout_information,
                                  stdout_success)
from simba.utils.read_write import (clean_sleap_file_name,
                                    find_all_videos_in_project, get_fn_ext,
                                    get_video_meta_data, write_df)

FRAME_IDX_COL, INST_START_COL, INST_END_COL = 2, 3, 4
INSTANCE_TRACK_COL = 4
POINT_X_COL, POINT_Y_COL, POINT_SCORE_COL = 0, 1, 4


def _field(arr: np.ndarray, name: str, pos: int) -> np.ndarray:
    """Return a column of a (possibly structured) numpy array by field name, falling back to positional index."""
    names = arr.dtype.names
    if names is not None and name in names:
        return np.asarray(arr[name])
    if names is not None:
        return np.asarray(arr[names[pos]])
    return np.asarray(arr[:, pos])


class SLEAPImporterSLP(ConfigReader, PoseImporterMixin):
    """
    Class for importing SLEAP pose-estimation data (``.slp`` files) into a SimBA project.

    :param str project_path: path to SimBA project config file in Configparser format.
    :param str data_folder: Path to folder containing SLEAP data in ``.slp`` format.
    :param List[str] id_lst: Animal names. This will be ignored in one animal projects and default to ``Animal_1``.
    :param Optional[Dict[str, str]] interpolation_settings: Dict defining the type and method to use to perform interpolation e.g. {'type': 'animals', 'method': 'linear'}.
    :param Optional[Dict[str, Union[str, int]]] smoothing_settings: Dictionary defining the pose estimation smoothing method e.g. {'time_window': 500, 'method': 'gaussian'}.

    :example:
    >>> slp_importer = SLEAPImporterSLP(project_path="MyConfigPath", data_folder=r'MySLPDataFolder', id_lst=['Mouse_1', 'Mouse_2'], interpolation_settings={'type': 'animals', 'method': 'linear'}, smoothing_settings={'time_window': 200, 'method': 'savitzky-golay'})
    >>> slp_importer.run()

    References
    ----------
    .. [1] Pereira, T. D., et al. (2022). SLEAP: A deep learning system for multi-animal pose tracking.
           `Nature Methods, 19, 486–495 <https://doi.org/10.1038/s41592-022-01426-1>`_.
    """

    def __init__(self,
                 project_path: Union[str, os.PathLike],
                 data_folder:  Union[str, os.PathLike],
                 id_lst: List[str],
                 interpolation_settings: Optional[Dict[str, str]] = None,
                 smoothing_settings: Optional[Dict[str, Any]] = None):

        ConfigReader.__init__(self, config_path=project_path, read_video_info=False)
        PoseImporterMixin.__init__(self)
        log_event(logger_name=str(__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        check_file_exist_and_readable(file_path=project_path)
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
        self.video_paths = find_all_videos_in_project(videos_dir=self.video_dir, raise_error=True if len(id_lst) > 1 else False)
        self.input_data_paths = self.find_data_files(dir=self.data_folder, extensions=[".slp"])
        if self.pose_setting is Methods.USER_DEFINED.value:
            self.__update_config_animal_cnt()
        if self.animal_cnt > 1:
            self.data_and_videos_lk = self.link_video_paths_to_data_paths(data_paths=self.input_data_paths, video_paths=self.video_paths, filename_cleaning_func=clean_sleap_file_name)
            self.check_multi_animal_status()
            self.animal_bp_dict = self.create_body_part_dictionary(self.multi_animal_status, self.id_lst, self.animal_cnt, self.x_cols, self.y_cols, self.p_cols, self.clr_lst)
            if self.pose_setting is Methods.USER_DEFINED.value:
                self.update_bp_headers_file(update_bp_headers=True)
        else:
            self.data_and_videos_lk = dict([(get_fn_ext(file_path)[1], {"DATA": file_path, "VIDEO": None}) for file_path in self.input_data_paths])
        stdout_information(msg=f"Importing {len(list(self.data_and_videos_lk.keys()))} file(s)...", source=self.__class__.__name__)

    @staticmethod
    def _read_metadata(file_path: Union[str, os.PathLike]) -> dict:
        """Pull the JSON ``metadata`` blob (skeleton / node order) out of an .slp file."""
        result = {}

        def _visitor(name, obj):
            if name == "metadata":
                attrs = list(obj.attrs.items())
                result["meta"] = dict(json.loads(attrs[1][1].decode("utf-8")))

        with h5py.File(file_path, "r") as f:
            f.visititems(_visitor)
        return result["meta"]

    def _slp_to_arrays(self, file_path: Union[str, os.PathLike]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Read an .slp file and return:
            * ``data`` : (n_instances, n_nodes * 3) float64 array of interleaved [x, y, p].
            * ``idx``  : (n_instances, 2) array of [track_id, frame_idx].
            * ``ordered_bps`` : body-part names in skeleton node order.
        """
        meta = self._read_metadata(file_path)
        bp_names = [bp["name"] for bp in meta["nodes"]]
        ordered_ids = [n["id"] for n in meta["skeletons"][0]["nodes"]]
        ordered_bps = [bp_names[i] for i in ordered_ids]
        n_nodes = len(ordered_bps)

        with h5py.File(file_path, "r") as f:
            frames = f["frames"][:]
            instances = f["instances"][:]
            pred_points = f["pred_points"][:]

        n_instances = instances.shape[0]
        expected_points = n_instances * n_nodes
        if pred_points.shape[0] != expected_points:
            raise CountError(msg=f"SLP file {file_path}: expected {expected_points} predicted points "
                                 f"({n_instances} instances x {n_nodes} nodes) but found {pred_points.shape[0]}.",
                             source=self.__class__.__name__)

        frame_idx_vals = _field(frames, "frame_idx", FRAME_IDX_COL).astype(np.int64)
        inst_start = _field(frames, "instance_id_start", INST_START_COL).astype(np.int64)
        inst_end = _field(frames, "instance_id_end", INST_END_COL).astype(np.int64)
        instance_frame_idx = np.repeat(frame_idx_vals, inst_end - inst_start)

        track_ids = _field(instances, "track", INSTANCE_TRACK_COL).astype(np.int64)

        px = _field(pred_points, "x", POINT_X_COL).astype(np.float64).reshape(n_instances, n_nodes)
        py = _field(pred_points, "y", POINT_Y_COL).astype(np.float64).reshape(n_instances, n_nodes)
        pp = _field(pred_points, "score", POINT_SCORE_COL).astype(np.float64).reshape(n_instances, n_nodes)

        data = np.empty((n_instances, n_nodes * 3), dtype=np.float64)
        data[:, 0::3] = px
        data[:, 1::3] = py
        data[:, 2::3] = pp

        idx = np.column_stack([track_ids, instance_frame_idx])
        return data, idx, ordered_bps

    def run(self):
        for file_cnt, (video_name, video_data) in enumerate(self.data_and_videos_lk.items()):
            output_filename = clean_sleap_file_name(filename=video_name)
            stdout_information(msg=f"Importing {output_filename}...", source=self.__class__.__name__)
            video_timer = SimbaTimer(start=True)
            self.video_name = video_name
            self.save_path = os.path.join(self.input_csv_dir, f"{output_filename}.{self.file_type}")

            data, idx, _ = self._slp_to_arrays(file_path=video_data["DATA"])

            if self.animal_cnt > 1:
                self.data_df = pd.DataFrame(self.transpose_multi_animal_table(data=data, idx=idx, animal_cnt=self.animal_cnt))
            else:
                self.data_df = pd.DataFrame(data).set_index(idx[:, 1]).sort_index()
                self.data_df = self.data_df.reindex(range(0, int(self.data_df.index[-1]) + 1), fill_value=0)

            if len(self.bp_headers) != len(self.data_df.columns):
                raise CountError(msg=f"The SimBA project expects {len(self.bp_headers)} data columns, but the reshaped "
                                     f"SLEAP data at {video_data['DATA']} produced {len(self.data_df.columns)} columns. "
                                     f"Make sure you have specified the correct number of animals and body-parts in your project.",
                                 source=self.__class__.__name__)
            self.data_df.columns = self.bp_headers

            if self.animal_cnt > 1:
                self.initialize_multi_animal_ui(animal_bp_dict=self.animal_bp_dict, video_info=get_video_meta_data(video_data["VIDEO"]), data_df=self.data_df, video_path=video_data["VIDEO"])
                self.multianimal_identification()
            else:
                self.out_df = self.insert_multi_idx_columns(df=self.data_df.fillna(0))

            write_df(df=self.out_df, file_type=self.file_type, save_path=self.save_path, multi_idx_header=True)
            if self.interpolation_settings is not None:
                Interpolate(config_path=self.config_path, data_path=self.save_path, type=self.interpolation_settings['type'], method=self.interpolation_settings['method'], multi_index_df_headers=True, copy_originals=False).run()
            if self.smoothing_settings is not None:
                Smoothing(config_path=self.config_path, data_path=self.save_path, time_window=self.smoothing_settings['time_window'], method=self.smoothing_settings['method'], multi_index_df_headers=True, copy_originals=False).run()
            video_timer.stop_timer()
            stdout_success(msg=f"Video {output_filename} data imported...", elapsed_time=video_timer.elapsed_time_str, source=self.__class__.__name__)
        self.timer.stop_timer()
        stdout_success(msg=f"All SLEAP SLP data files imported to {self.input_csv_dir} directory", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)

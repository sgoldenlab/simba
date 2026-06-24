import os
from copy import copy
from typing import Optional, Union

from simba.data_processors.interpolate import Interpolate
from simba.data_processors.smoothing import Smoothing
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_if_dir_exists,
                                check_if_keys_exist_in_dict, check_int,
                                check_str, check_valid_boolean)
from simba.utils.enums import ConfigKey, Dtypes, Methods
from simba.utils.errors import InvalidInputError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_facemap_h5, write_df)


class FaceMapImporter(ConfigReader):
    """
    Import FaceMap orofacial keypoint tracking data into a SimBA project.

    FaceMap tracks 15 keypoints on the mouse face (4 eye, 5 nose, 3 whisker, mouth + lowerlip, and paw),
    storing each keypoint's ``x``, ``y`` and ``likelihood`` per frame in an ``.h5`` file. This class reads each
    ``.h5`` file, converts it to a SimBA pose-estimation CSV (one ``<keypoint>_x``, ``<keypoint>_y``, ``<keypoint>_p``
    triplet per keypoint), and saves it into the project's ``outlier_corrected_movement_location`` directory.
    Interpolation and smoothing are applied afterwards if requested.

    .. image:: _static/img/simba.pose_importers.facemap_h5_importer.FaceMapImporter.webp
       :alt: FaceMap tracks 15 mouse-face keypoints; FaceMapImporter converts each .h5 to a SimBA CSV with optional interpolation and smoothing
       :width: 800
       :align: center

    .. seealso::
       The ``.h5`` parsing is performed by :func:`simba.utils.read_write.read_facemap_h5`

    .. note::
       The SimBA project must be created as a FaceMap project (``pose_estimation_body_parts: Facemap``); importing into a project of any other pose-configuration raises an error.

       `FaceMap GitHub repository <https://github.com/mouseland/facemap>`__.

    References
    ----------
    .. [1] Syeda, A., Zhong, L., Tung, R., et al. Facemap: a framework for modeling neural activity based on orofacial tracking. `Nature Neuroscience, 26, 1775–1783 (2023) <https://www.nature.com/articles/s41593-023-01490-6>`_.
    .. [2] FaceMap, MouseLand, `https://github.com/mouseland/facemap <https://github.com/mouseland/facemap>`__.

    :param Union[str, os.PathLike] config_path: Path to SimBA project ``project_config.ini`` (must be a FaceMap project).
    :param Union[str, os.PathLike] data_path: Path to a FaceMap ``.h5`` file, or a directory containing one or more ``.h5`` files.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory to save the imported CSVs. If None, defaults to the project's ``outlier_corrected_movement_location`` directory.
    :param Optional[dict] smoothing_settings: Optional smoothing, e.g. ``{'method': 'savitzky-golay', 'time_window': 100}``. ``method`` is one of ``'savitzky-golay'`` / ``'gaussian'``; ``time_window`` is in milliseconds. Default None (no smoothing).
    :param Optional[dict] interpolation_settings: Optional interpolation, e.g. ``{'type': 'body-parts', 'method': 'linear'}``. ``type`` is ``'body-parts'`` / ``'animals'``; ``method`` is ``'linear'`` / ``'quadratic'`` / ``'nearest'``. Default None (no interpolation).
    :param Optional[bool] verbose: If True, print per-file progress. Default True.

    :example:
    >>> r = FaceMapImporter(config_path=r"C:\troubleshooting\facemap_project\project_folder\project_config.ini", data_path=r'C:\troubleshooting\facemap_project\data', smoothing_settings={'method': 'savitzky-golay', 'time_window': 100})
    >>> r.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_path: Union[str, os.PathLike],
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 smoothing_settings: Optional[dict] = None,
                 interpolation_settings: Optional[dict] = None,
                 verbose: Optional[bool] = True):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        pose_config_name = self.read_config_entry(config=self.config, section=ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, option=ConfigKey.POSE_SETTING.value, default_value=None, data_type=Dtypes.STR.value).strip()
        if pose_config_name != Methods.FACEMAP.value:
            raise InvalidInputError(msg=f'The project {config_path} is not a FaceMap project. Cannot import FaceMap data to a non SimBA FaceMap project ({ConfigKey.POSE_SETTING.value}: {pose_config_name}, expected: {Methods.FACEMAP.value})', source=self.__class__.__name__)
        if os.path.isdir(data_path):
            self.data_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=['.h5'], raise_error=True)
        elif os.path.isfile(data_path):
            self.data_paths = [data_path]
        else:
            raise NoFilesFoundError(msg=f'{data_path} is not a valid file path or valid directory path', source=self.__class__.__name__)
        if interpolation_settings is not None:
            check_if_keys_exist_in_dict(data=interpolation_settings, key=['method', 'type'], name=f'{self.__class__.__name__} interpolation_settings')
            check_str(name=f'{self.__class__.__name__} interpolation_settings type', value=interpolation_settings['type'], options=('body-parts', 'animals'))
            check_str(name=f'{self.__class__.__name__} interpolation_settings method', value=interpolation_settings['method'], options=('linear', 'quadratic', 'nearest'))
            self.interpolation_type, self.interpolation_method = interpolation_settings['type'], interpolation_settings['method']
        else:
            self.interpolation_type, self.interpolation_method = None, None
        if smoothing_settings is not None:
            check_if_keys_exist_in_dict(data=smoothing_settings, key=['method', 'time_window'], name=f'{self.__class__.__name__} smoothing_settings')
            check_str(name=f'{self.__class__.__name__} smoothing_settings method', value=smoothing_settings['method'], options=('savitzky-golay', 'gaussian'))
            check_int(name=f'{self.__class__.__name__} smoothing_settings time_window', value=smoothing_settings['time_window'], min_value=1)
            self.smoothing_time, self.smoothing_method = smoothing_settings['time_window'], smoothing_settings['method']
        else:
            self.smoothing_time, self.smoothing_method = None, None
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir', raise_error=True)
        else:
            save_dir = copy(self.outlier_corrected_dir)
        self.interpolation_settings, self.smoothing_settings, = (interpolation_settings, smoothing_settings)
        self.save_dir, self.verbose = save_dir, verbose

    def run(self):
        for file_cnt, file_path in enumerate(self.data_paths):
            video_name = get_fn_ext(filepath=file_path)[1]
            if self.verbose:
                print(f'Importing FaceMap data for video {video_name}...')
            file_timer = SimbaTimer(start=True)
            data_df = read_facemap_h5(file_path=file_path)
            save_path = os.path.join(self.save_dir, f'{video_name}.csv')
            write_df(df=data_df, file_type=self.file_type, save_path=save_path, multi_idx_header=False)
            if self.interpolation_settings is not None:
                interpolator = Interpolate(config_path=self.config_path, data_path=save_path, type=self.interpolation_type, method=self.interpolation_method, multi_index_df_headers=False, copy_originals=False)
                interpolator.run()
            if self.smoothing_settings is not None:
                smoother = Smoothing(config_path=self.config_path, data_path=save_path, time_window=self.smoothing_time, method=self.smoothing_method, multi_index_df_headers=False, copy_originals=False)
                smoother.run()
            file_timer.stop_timer()
            print(f'Imported data for video {video_name} (elapsed time: {file_timer.elapsed_time}s)')
        self.timer.stop_timer()
        stdout_success(msg=f"{len(self.data_paths)} SimBA FaceMap tracking files file(s) imported to the SimBA project {self.save_dir}", source=self.__class__.__name__, elapsed_time=self.timer.elapsed_time)



# r = FaceMapImporter(config_path=r"C:\troubleshooting\facemap_project\project_folder\project_config.ini", data_path=r'C:\troubleshooting\facemap_project\data', smoothing_settings={'method': 'savitzky-golay', 'time_window': 100})
# r.run()
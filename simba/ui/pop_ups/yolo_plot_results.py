from tkinter import *

import numpy as np

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.yolo_pose_visualizer import YOLOPoseVisualizer
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FileSelect,
                                        FolderSelect, SimbaButton,
                                        SimBADropDown)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.enums import Options, PackageNames
from simba.utils.errors import (NoDataError, SimBAGPUError,
                                SimBAPAckageVersionError)
from simba.utils.printing import stdout_warning
from simba.utils.read_write import (find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_pkg_version, get_video_meta_data,
                                    str_2_bool)
from simba.utils.warnings import MissingFileWarning

MAX_TRACKS_OPTIONS = ['None', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
CORE_CNT_OPTIONS = list(range(1, find_core_cnt()[0]))
THRESHOLD_OPTIONS = np.arange(0.1, 1.1, 0.1).astype(np.float32)
SIZE_OPTIONS = list(range(1, 21, 1))
SIZE_OPTIONS.insert(0, 'AUTO')

class YoloPoseVisualizerPopUp(PopUpMixin):

    def __init__(self):
        gpu_available, gpus = _is_cuda_available()
        if not gpu_available:
            raise SimBAGPUError(msg=f'Cannot train YOLO pose-estimation model. No NVIDA GPUs detected on machine', source=self.__class__.__name__)
        ultralytics_version = get_pkg_version(pkg=PackageNames.ULTRALYTICS.value)
        if ultralytics_version is None:
            raise SimBAPAckageVersionError(msg=f'Cannot train YOLO pose-estimation model: Could not find ultralytics package in python environment',  source=self.__class__.__name__)

        PopUpMixin.__init__(self, title="PLOT YOLO POSE ESTIMATION RESULTS", icon='ultralytics_2')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        self.save_dir = FolderSelect(settings_frm, folderDescription="SAVE DIRECTORY:", lblwidth=35, entry_width=45)
        self.core_cnt_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=CORE_CNT_OPTIONS, label="CPU CORE COUNT:", label_width=35, dropdown_width=40, value=int(max(CORE_CNT_OPTIONS) / 2))
        self.bbox_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="SHOW BOUNDING BOXES:",  label_width=35, dropdown_width=40, value='FALSE')
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="VERBOSE:",  label_width=35, dropdown_width=40, value='TRUE')
        self.threshold_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=THRESHOLD_OPTIONS, label="THRESHOLD:",  label_width=35, dropdown_width=40, value=0.5)
        self.thickness_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=SIZE_OPTIONS, label="LINE THICKNESS:",  label_width=35, dropdown_width=40, value='AUTO')
        self.circle_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=SIZE_OPTIONS, label="CIRCLE SIZE:", label_width=35, dropdown_width=40, value='AUTO')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=0, column=0, sticky=NW)
        self.core_cnt_dropdown.grid(row=1, column=0, sticky=NW)
        self.bbox_dropdown.grid(row=2, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=3, column=0, sticky=NW)
        self.threshold_dropdown.grid(row=4, column=0, sticky=NW)
        self.thickness_dropdown.grid(row=5, column=0, sticky=NW)
        self.circle_dropdown.grid(row=6, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="PLOT SINGLE VIDEO", icon_name='video')
        self.data_path = FileSelect(parent=single_video_frm, fileDescription='DATA PATH (CSV):', lblwidth=35,  entry_width=45, file_types=[("YOLO CSV RESULT", ".csv")])
        self.video_path = FileSelect(parent=single_video_frm, fileDescription='VIDEO PATH:', lblwidth=35,  entry_width=45, file_types=[("YOLO CSV RESULT", Options.ALL_VIDEO_FORMAT_OPTIONS.value)])
        single_video_btn = SimbaButton(parent=single_video_frm, txt='CREATE SINGLE VIDEO', img='rocket', cmd=self.run, cmd_kwargs={'multiple': False})

        single_video_frm.grid(row=1, column=0, sticky=NW)
        self.data_path.grid(row=0, column=0, sticky=NW)
        self.video_path.grid(row=1, column=0, sticky=NW)
        single_video_btn.grid(row=2, column=0, sticky=NW)

        multiple_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="PLOT MULTIPLE VIDEOS", icon_name='stack')
        self.data_dir_path = FolderSelect(parent=multiple_video_frm, folderDescription='DATA DIRECTORY:', lblwidth=35,  entry_width=45)
        self.video_dir_path = FolderSelect(parent=multiple_video_frm, folderDescription='VIDEO DIRECTORY:', lblwidth=35,  entry_width=45)
        multiple_video_btn = SimbaButton(parent=multiple_video_frm, txt='CREATE MULTIPLE VIDEOS', img='rocket', cmd=self.run, cmd_kwargs={'multiple': True})

        multiple_video_frm.grid(row=2, column=0, sticky=NW)
        self.data_dir_path.grid(row=0, column=0, sticky=NW)
        self.video_dir_path.grid(row=1, column=0, sticky=NW)
        multiple_video_btn.grid(row=2, column=0, sticky=NW)

        self.main_frm.mainloop()

    def run(self, multiple: bool):

        save_dir = self.save_dir.folder_path
        core_cnt = int(self.core_cnt_dropdown.get_value())
        bbox = str_2_bool(self.bbox_dropdown.get_value())
        verbose = str_2_bool(self.verbose_dropdown.get_value())
        threshold = float(self.threshold_dropdown.get_value())
        thickness = None if self.threshold_dropdown.get_value() == 'AUTO' else float(self.threshold_dropdown.get_value())
        circle_size = None if self.circle_dropdown.get_value() == 'AUTO' else float(self.circle_dropdown.get_value())
        if not multiple:
            data_path = self.data_path.file_path
            video_path = self.video_path.file_path
            check_file_exist_and_readable(file_path=data_path, raise_error=True)
            _ = get_video_meta_data(video_path=video_path)
            plotter = YOLOPoseVisualizer(data_path=data_path, video_path=video_path, save_dir=save_dir, core_cnt=core_cnt, threshold=threshold, thickness=thickness, circle_size=circle_size, verbose=verbose, bbox=bbox)
            plotter.run()
        else:
            data_dir = self.data_dir_path.folder_path
            video_dir = self.video_dir_path.folder_path
            check_if_dir_exists(in_dir=data_dir)
            check_if_dir_exists(in_dir=video_dir)
            data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'], raise_error=True, as_dict=True)
            video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True, as_dict=True)

            missing_videos = list([x for x in video_paths.keys() if x not in data_paths.keys()])
            missing_data_paths = list([x for x in data_paths.keys() if x not in video_paths.keys()])
            if len(missing_videos) > 0:
                MissingFileWarning(msg=f'Data files are missing video files in the {video_dir} directory: {missing_videos}', source=self.__class__.__name__)
            if len(missing_data_paths) > 0:
                MissingFileWarning(msg=f'Video files are missing data files in the {data_dir} directory: {missing_data_paths}', source=self.__class__.__name__)

            data_paths = {k:v for k, v in data_paths.items() if k in video_paths.keys()}
            if len(list(data_paths.keys())) == 0:
                raise NoDataError(msg=f'No data file in the {data_dir} directory has a representative video file in the {video_dir} directory', source=self.__class__.__name__)
            video_cnt = len(list(data_paths.keys()))
            for cnt, (name, data_path) in enumerate(data_paths.items()):
                if name in video_paths.keys():
                    video_path = video_paths[name]
                    print(f'Plotting YOLO results for video {name} (video {cnt+1}/{video_cnt})')
                    plotter = YOLOPoseVisualizer(data_path=data_path, video_path=video_path, save_dir=save_dir, core_cnt=core_cnt, threshold=threshold, thickness=thickness, circle_size=circle_size, verbose=verbose, bbox=bbox)
                    plotter.run()
                else:
                    stdout_warning(msg=f'Skipping video {name} (no video exist in {video_dir})...')



#YoloPlotSingleVideoPopUp()





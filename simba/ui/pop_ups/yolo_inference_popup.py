import os
from tkinter import *

import numpy as np

import simba
from simba.data_processors.cuda.utils import _is_cuda_available
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.model.yolo_pose_inference import YOLOPoseInference
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FileSelect,
                                        FolderSelect, SimbaButton,
                                        SimBADropDown)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.enums import Options, PackageNames, Paths
from simba.utils.errors import SimBAGPUError, SimBAPAckageVersionError
from simba.utils.read_write import (find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_pkg_version, get_video_meta_data,
                                    str_2_bool)

MAX_TRACKS_OPTIONS = ['None', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
BATCH_SIZE_OPTIONS =  list(range(50, 1050, 50))
CORE_CNT_OPTIONS = list(range(1, find_core_cnt()[0]))
IMG_SIZE_OPTIONS = [256, 320, 416, 480, 512, 640, 720, 768, 960, 1280]
SMOOTHING_OPTIONS = ['None', 50, 100, 200, 300, 400, 500]

devices = ['CPU']
THRESHOLD_OPTIONS = np.arange(0.1, 1.1, 0.1).astype(np.float32)

class YOLOPoseInferencePopUP(PopUpMixin):

    def __init__(self):
        gpu_available, gpus = _is_cuda_available()
        if not gpu_available:
            raise SimBAGPUError(msg=f'Cannot train YOLO pose-estimation model. No NVIDA GPUs detected on machine', source=self.__class__.__name__)
        ultralytics_version = get_pkg_version(pkg=PackageNames.ULTRALYTICS.value)
        if ultralytics_version is None:
            raise SimBAPAckageVersionError(msg=f'Cannot train YOLO pose-estimation model: Could not find ultralytics package in python environment',  source=self.__class__.__name__)

        simba_dir = os.path.dirname(simba.__file__)
        yolo_schematics_dir = os.path.join(simba_dir, Paths.YOLO_SCHEMATICS_DIR.value)
        seven_bp_dir = os.path.join(yolo_schematics_dir, 'yolo_7bps.csv')

        PopUpMixin.__init__(self, title="PREDICT USING YOLO POSE ESTIMATION MODEL", icon='ultralytics_2')
        devices.extend([f'{x} : {y["model"]}' for x, y in gpus.items()])

        paths_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MODEL & DATA PATHS", icon_name='browse')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')

        self.weights_path = FileSelect(parent=paths_frm, fileDescription='MODEL PATH (E.G., .PT):', lblwidth=35,  entry_width=45, file_types=[("YOLO MODEL FILE", Options.ALL_YOLO_MODEL_FORMAT_STR_OPTIONS.value)])
        self.save_dir = FolderSelect(paths_frm, folderDescription="SAVE DIRECTORY:", lblwidth=35, entry_width=45)
        self.bp_config_csv_path = FileSelect(parent=paths_frm, fileDescription='BODY-PART NAMES (.CSV):', lblwidth=35,  entry_width=45, file_types=[("CSV FILE", ".csv")], initialdir=yolo_schematics_dir, initial_path=seven_bp_dir)

        self.batch_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=BATCH_SIZE_OPTIONS, label="BATCH SIZE: ", label_width=35, dropdown_width=40, value=250)
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="VERBOSE:", label_width=35, dropdown_width=40, value='TRUE')
        self.workers_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=CORE_CNT_OPTIONS, label="CPU WORKERS:", label_width=35, dropdown_width=40, value=int(max(CORE_CNT_OPTIONS)/2))
        self.format_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=Options.VALID_YOLO_FORMATS.value, label="FORMAT:", label_width=35, dropdown_width=40, value='pb')
        self.img_size_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=IMG_SIZE_OPTIONS, label="IMAGE SIZE:", label_width=35, dropdown_width=40, value=640)
        self.devices_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=devices, label="DEVICE:", label_width=35, dropdown_width=40, value=devices[1])
        self.interpolate_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="INTERPOLATE:",  label_width=35, dropdown_width=40, value='TRUE')
        self.stream_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="STREAM:", label_width=35, dropdown_width=40, value='TRUE')
        self.threshold_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=THRESHOLD_OPTIONS, label="THRESHOLD:", label_width=35, dropdown_width=40, value=0.5)
        self.iou_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=THRESHOLD_OPTIONS,  label="IOU:", label_width=35, dropdown_width=40, value=0.7)
        self.max_tracks_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=MAX_TRACKS_OPTIONS, label="MAX TRACKS:", label_width=35, dropdown_width=40, value=MAX_TRACKS_OPTIONS[0])
        self.max_track_per_id_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=MAX_TRACKS_OPTIONS, label="MAX TRACKS PER ID:", label_width=35, dropdown_width=40, value=1)
        self.smoothing_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=SMOOTHING_OPTIONS, label="SMOOTHING (MS):", label_width=35, dropdown_width=40, value=SMOOTHING_OPTIONS[2])

        paths_frm.grid(row=0, column=0, sticky=NW)
        settings_frm.grid(row=1, column=0, sticky=NW)

        self.weights_path.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)
        self.bp_config_csv_path.grid(row=2, column=0, sticky=NW)

        self.format_dropdown.grid(row=0, column=0, sticky=NW)
        self.img_size_dropdown.grid(row=1, column=0, sticky=NW)
        self.threshold_dropdown.grid(row=2, column=0, sticky=NW)
        self.interpolate_dropdown.grid(row=3, column=0, sticky=NW)
        self.iou_dropdown.grid(row=4, column=0, sticky=NW)
        self.stream_dropdown.grid(row=5, column=0, sticky=NW)
        self.workers_dropdown.grid(row=6, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=7, column=0, sticky=NW)
        self.devices_dropdown.grid(row=8, column=0, sticky=NW)
        self.max_tracks_dropdown.grid(row=9, column=0, sticky=NW)
        self.max_track_per_id_dropdown.grid(row=10, column=0, sticky=NW)
        self.smoothing_dropdown.grid(row=11, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="ANALYZE SINGLE VIDEO", icon_name='video')
        self.video_path = FileSelect(parent=single_video_frm, fileDescription='VIDEO PATH:', lblwidth=35, entry_width=45, file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        run_single_video_btn = SimbaButton(parent=single_video_frm, txt='ANALYZE SINGLE VIDEO', txt_clr='blue', img='rocket', cmd=self.run, cmd_kwargs={'multiple': False})

        single_video_frm.grid(row=2, column=0, sticky=NW)
        self.video_path.grid(row=0, column=0, sticky=NW)
        run_single_video_btn.grid(row=1, column=0, sticky=NW)


        video_dir_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="ANALYZE VIDEO DIRECTORY", icon_name='stack')
        self.video_dir = FolderSelect(parent=video_dir_frm, folderDescription='VIDEO DIRECTORY PATH:', entry_width=45, lblwidth=35)
        run_multiple_video_btn = SimbaButton(parent=video_dir_frm, txt='ANALYZE VIDEO DIRECTORY', txt_clr='blue', img='rocket', cmd=self.run, cmd_kwargs={'multiple': True})

        video_dir_frm.grid(row=3, column=0, sticky=NW)
        self.video_dir.grid(row=0, column=0, sticky=NW)
        run_multiple_video_btn.grid(row=1, column=0, sticky=NW)

        self.main_frm.mainloop()

    def run(self, multiple: bool):
        mdl_path = self.weights_path.file_path
        save_dir = self.save_dir.folder_path
        format = self.format_dropdown.get_value()
        img_size = int(self.img_size_dropdown.get_value())
        batch_size = int(self.batch_dropdown.get_value())
        threshold = float(self.threshold_dropdown.get_value())
        interpolate = str_2_bool(self.interpolate_dropdown.get_value())
        iou = float(self.iou_dropdown.get_value())
        stream = str_2_bool(self.stream_dropdown.get_value())
        max_tracks = self.max_tracks_dropdown.get_value()
        max_tracks_per_id = self.max_track_per_id_dropdown.get_value()
        smoothing = self.smoothing_dropdown.get_value()
        verbose = str_2_bool(self.verbose_dropdown.get_value())
        workers = int(self.workers_dropdown.get_value())
        device = self.devices_dropdown.get_value()
        device = 'cpu' if device == 'CPU' else int(device.split(':', 1)[0])
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} SAVE DIRECTORY')
        check_file_exist_and_readable(file_path=mdl_path, raise_error=True)

        max_tracks = None if max_tracks == 'None' else int(max_tracks)
        max_tracks_per_id = None if max_tracks_per_id == 'None' else int(max_tracks_per_id)
        smoothing = None if smoothing == 'None' else int(smoothing)

        if multiple:
            check_if_dir_exists(in_dir=self.video_dir.folder_path, source=self.__class__.__name__, raise_error=True)
            video_paths = find_files_of_filetypes_in_directory(directory=self.video_dir.folder_path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        else:
            check_file_exist_and_readable(file_path=self.video_path.file_path, raise_error=True)
            video_paths = [self.video_path.file_path]
        for video_path in video_paths:
            _ = get_video_meta_data(video_path=video_path)

        runner = YOLOPoseInference(weights=mdl_path,
                                   video_path=video_paths,
                                   verbose=verbose,
                                   save_dir=save_dir,
                                   device=device, format=format,
                                   batch_size=batch_size,
                                   torch_threads=workers,
                                   box_threshold=threshold,
                                   max_tracks=max_tracks,
                                   max_per_class=max_tracks_per_id,
                                   interpolate=interpolate,
                                   imgsz=img_size,
                                   iou=iou,
                                   stream=stream,
                                   smoothing=smoothing)
        runner.run()



#YOLOPoseInferencePopUP()

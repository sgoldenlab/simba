
from tkinter import *

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.model.yolo_fit import FitYolo
from simba.third_party_label_appenders.transform.utils import \
    check_valid_yolo_map
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FileSelect,
                                        FolderSelect, SimBADropDown)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.enums import Options, PackageNames
from simba.utils.errors import SimBAGPUError, SimBAPAckageVersionError
from simba.utils.read_write import find_core_cnt, get_pkg_version, str_2_bool

EPOCH_OPTIONS = list(range(100, 5750, 250))
PATIENCE_OPTIONS = list(range(50, 1050, 50))
IMG_SIZE_OPTIONS = [256, 320, 416, 480, 512, 640, 720, 768, 960, 1280]
CORE_CNT_OPTIONS = list(range(1, find_core_cnt()[0]))
BATCH_SIZE_OPTIONS =  [2, 4, 8, 16, 32, 64, 128]
devices = ['CPU']

class YOLOPoseTrainPopUP(PopUpMixin):

    def __init__(self):
        gpu_available, gpus = _is_cuda_available()
        if not gpu_available:
            raise SimBAGPUError(msg=f'Cannot train YOLO pose-estimation model. No NVIDA GPUs detected on machine', source=self.__class__.__name__)
        ultralytics_version = get_pkg_version(pkg=PackageNames.ULTRALYTICS.value)
        if ultralytics_version is None:
            raise SimBAPAckageVersionError(msg=f'Cannot train YOLO pose-estimation model: Could not find ultralytics package in python environment',  source=self.__class__.__name__)

        PopUpMixin.__init__(self, title="TRAIN YOLO POSE ESTIMATION MODEL", icon='ultralytics_2')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        devices.extend([f'{x} : {y["model"]}' for x, y in gpus.items()])
        self.yolo_map_path = FileSelect(parent=settings_frm, fileDescription='YOLO MAP FILE (YAML):', lblwidth=35, entry_width=45, file_types=[("YOLO MODEL FILE", Options.ALL_YOLO_MODEL_FORMAT_STR_OPTIONS.value)])
        self.save_dir = FolderSelect(settings_frm, folderDescription="SAVE DIRECTORY:", lblwidth=35, entry_width=45)
        self.weights_path = FileSelect(parent=settings_frm, fileDescription='INITIAL WEIGHT FILE (E.G., .PT):', lblwidth=35, entry_width=45)

        self.epochs_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=EPOCH_OPTIONS, label="EPOCHS: ", label_width=35, dropdown_width=40, value=500)
        self.batch_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=BATCH_SIZE_OPTIONS, label="BATCH SIZE: ", label_width=35, dropdown_width=40, value=16)
        self.plots_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="PLOTS:", label_width=35, dropdown_width=40, value='TRUE')
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="VERBOSE:", label_width=35, dropdown_width=40, value='TRUE')
        self.workers_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=CORE_CNT_OPTIONS, label="CPU WORKERS:", label_width=35, dropdown_width=40, value=int(max(CORE_CNT_OPTIONS)/2))
        self.format_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=Options.VALID_YOLO_FORMATS.value, label="FORMAT:", label_width=35, dropdown_width=40, value='engine')
        self.img_size_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=IMG_SIZE_OPTIONS, label="IMAGE SIZE:", label_width=35, dropdown_width=40, value=640)
        self.patience_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=PATIENCE_OPTIONS, label="PATIENCE:", label_width=35, dropdown_width=40, value=100)
        self.devices_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=devices, label="DEVICE:", label_width=35, dropdown_width=40, value=devices[1])

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.yolo_map_path.grid(row=0, column=0, sticky=NW)
        self.weights_path.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=2, column=0, sticky=NW)
        self.epochs_dropdown.grid(row=3, column=0, sticky=NW)
        self.img_size_dropdown.grid(row=4, column=0, sticky=NW)
        self.batch_dropdown.grid(row=5, column=0, sticky=NW)
        self.plots_dropdown.grid(row=6, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=7, column=0, sticky=NW)
        self.workers_dropdown.grid(row=8, column=0, sticky=NW)
        self.format_dropdown.grid(row=9, column=0, sticky=NW)
        self.patience_dropdown.grid(row=10, column=0, sticky=NW)
        self.devices_dropdown.grid(row=11, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()


    def run(self):
        yolo_map_path = self.yolo_map_path.file_path
        weights_path = self.weights_path.file_path
        save_dir = self.save_dir.folder_path
        plots = str_2_bool(self.plots_dropdown.get_value())
        verbose = str_2_bool(self.verbose_dropdown.get_value())
        epochs = int(self.epochs_dropdown.get_value())
        workers = int(self.workers_dropdown.get_value())
        batch_size = int(self.batch_dropdown.get_value())
        device = self.devices_dropdown.get_value()
        device = 'cpu' if device == 'CPU' else int(device.split(':', 1)[0])
        format = self.format_dropdown.get_value()
        imgsz = int(self.img_size_dropdown.get_value())
        patience = int(self.patience_dropdown.get_value())

        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} SAVE DIRECTORY')
        check_file_exist_and_readable(file_path=weights_path, raise_error=True)
        check_file_exist_and_readable(file_path=yolo_map_path, raise_error=True)
        check_valid_yolo_map(yolo_map=yolo_map_path)
        runner = FitYolo(weights_path=weights_path, model_yaml=yolo_map_path, save_path=save_dir, epochs=epochs, batch=batch_size, plots=plots, format=format, device=device, verbose=verbose, workers=workers, imgsz=imgsz, patience=patience)
        runner.run()

#YOLOPoseTrainPopUP()
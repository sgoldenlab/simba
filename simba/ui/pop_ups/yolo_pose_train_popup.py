from tkinter import *

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FileSelect,
                                        FolderSelect, SimBADropDown)
from simba.utils.enums import Options
from simba.utils.errors import SimBAGPUError, SimBAPAckageVersionError
from simba.utils.read_write import find_core_cnt, get_pkg_version

ULTRALYTICS = 'ultralytics'
EPOCH_OPTIONS = list(range(100, 1750, 250))
IMG_SIZE_OPTIONS = [256, 320, 416, 480, 512, 640, 720, 768, 960, 1280]
CORE_CNT_OPTIONS = list(range(1, find_core_cnt()[0]))

devices = ['CPU']

class YOLOPoseTrainPopUP(PopUpMixin):

    def __init__(self):
        gpu_available, gpus = _is_cuda_available()
        if not gpu_available:
            raise SimBAGPUError(msg=f'Cannot train YOLO pose-estimation model. No NVIDA GPUs detected on machine', source=self.__class__.__name__)
        ultralytics_version = get_pkg_version(pkg=ULTRALYTICS)
        if ultralytics_version is None:
            raise SimBAPAckageVersionError(msg=f'Cannot train YOLO pose-estimation model. Could not find ultralytics package',  source=self.__class__.__name__)

        PopUpMixin.__init__(self, title="TRAIN YOLO POSE ESTIMATION MODEL", icon='ultralytics_2')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        devices.extend([f'{x} : {y["model"]}' for x, y in gpus.items()])
        self.yolo_map_path = FileSelect(parent=settings_frm, fileDescription='YOLO MAP FILE (YAML):', lblwidth=35, entry_width=45)
        self.save_dir = FolderSelect(settings_frm, folderDescription="SAVE DIRECTORY:", lblwidth=35, entry_width=45)
        self.weights_path = FileSelect(parent=settings_frm, fileDescription='INITIAL WEIGHT FILE (.PT):', lblwidth=35, entry_width=45)

        self.epochs_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=EPOCH_OPTIONS, label="EPOCHS: ", label_width=35, dropdown_width=40, value=100)
        self.plots_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="PLOTS:", label_width=35, dropdown_width=40, value='TRUE')
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="VERBOSE:", label_width=35, dropdown_width=40, value='TRUE')
        self.workers_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=CORE_CNT_OPTIONS, label="CPU WORKERS:", label_width=35, dropdown_width=40, value=int(max(CORE_CNT_OPTIONS)/2))
        self.format_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=Options.VALID_YOLO_FORMATS.value, label="FORMAT:", label_width=35, dropdown_width=40, value='onnx')
        self.img_size = SimBADropDown(parent=settings_frm, dropdown_options=IMG_SIZE_OPTIONS, label="IMAGE SIZE:", label_width=35, dropdown_width=40, value=480)
        self.devices_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=devices, label="DEVICE:", label_width=35, dropdown_width=40, value=devices[1])

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.yolo_map_path.grid(row=0, column=0, sticky=NW)
        self.weights_path.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=2, column=0, sticky=NW)
        self.epochs_dropdown.grid(row=3, column=0, sticky=NW)
        self.plots_dropdown.grid(row=4, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=5, column=0, sticky=NW)
        self.workers_dropdown.grid(row=6, column=0, sticky=NW)
        self.format_dropdown.grid(row=7, column=0, sticky=NW)
        self.devices_dropdown.grid(row=8, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()


    def run(self):
        yolo_map_path = self.yolo_map_path.file_path
        weights_path = self.weights_path.file_path

        coco_file_path = self.save_dir.folder_path




        pass


#YOLOPoseTrainPopUP()
import os
import sys
import subprocess
import tempfile
from tkinter import *
from tkinter import messagebox

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.mixins.pop_up_mixin import PopUpMixin
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
# On Windows, PyTorch DataLoader often deadlocks with workers > 8; cap options to avoid hang
_max_workers = min(find_core_cnt()[0], 8) if sys.platform == 'win32' else find_core_cnt()[0]
CORE_CNT_OPTIONS = list(range(1, _max_workers + 1))
BATCH_SIZE_OPTIONS =  [2, 4, 8, 16, 32, 64, 128]
devices = ['CPU']
FORMAT_OPTIONS =  Options.VALID_YOLO_FORMATS.value
FORMAT_OPTIONS.insert(0, 'None')
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
        self.yolo_map_path = FileSelect(parent=settings_frm, fileDescription='YOLO MAP FILE (YAML):', lblwidth=35, entry_width=45, file_types=[("YOLO MODEL FILE", ".yaml")], lbl_icon='file', tooltip_key='yolo_map_path')
        self.save_dir = FolderSelect(settings_frm, folderDescription="SAVE DIRECTORY:", lblwidth=35, entry_width=45, lbl_icon='save', tooltip_key='SAVE_DIR')
        self.weights_path = FileSelect(parent=settings_frm, fileDescription='INITIAL WEIGHT FILE (E.G., .PT):', lblwidth=35, entry_width=45, lbl_icon='file', tooltip_key='yolo_initial_weights_path')

        self.epochs_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=EPOCH_OPTIONS, label="EPOCHS: ", label_width=35, dropdown_width=40, value=500, img='rotate', tooltip_key='epochs_dropdown')
        self.batch_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=BATCH_SIZE_OPTIONS, label="BATCH SIZE: ", label_width=35, dropdown_width=40, value=16, img='weight', tooltip_key='batch_dropdown')
        self.plots_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="PLOTS:", label_width=35, dropdown_width=40, value='TRUE', img='plot', tooltip_key='plots_dropdown')
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="VERBOSE:", label_width=35, dropdown_width=40, value='TRUE', img='verbose', tooltip_key='verbose_dropdown')
        self.workers_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=CORE_CNT_OPTIONS, label="CPU WORKERS:", label_width=35, dropdown_width=40, value=int(max(CORE_CNT_OPTIONS)/2), img='cpu_small', tooltip_key='workers_dropdown')
        self.format_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=FORMAT_OPTIONS, label="FORMAT:", label_width=35, dropdown_width=40, value='None', img='file_type', tooltip_key='format_dropdown')
        self.img_size_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=IMG_SIZE_OPTIONS, label="IMAGE SIZE:", label_width=35, dropdown_width=40, value=640, img='resize', tooltip_key='img_size_dropdown')
        self.patience_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=PATIENCE_OPTIONS, label="PATIENCE:", label_width=35, dropdown_width=40, value=100, img='timer', tooltip_key='patience_dropdown')
        self.devices_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=devices, label="DEVICE:", label_width=35, dropdown_width=40, value=devices[1], img='gpu_3', tooltip_key='devices_dropdown')

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
        device_str = 'cpu' if device == 'CPU' else device.split(':', 1)[0]
        format_val = None if self.format_dropdown.get_value() == 'None' else self.format_dropdown.get_value()
        imgsz = int(self.img_size_dropdown.get_value())
        patience = int(self.patience_dropdown.get_value())
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} SAVE DIRECTORY')
        if not check_file_exist_and_readable(file_path=weights_path, raise_error=False):
            weights_path = None
        check_file_exist_and_readable(file_path=yolo_map_path, raise_error=True)
        check_valid_yolo_map(yolo_map=yolo_map_path)

        # On Windows, PyTorch DataLoader with workers > 8 often deadlocks after train cache scan
        workers_for_subprocess = min(workers, 8) if sys.platform == 'win32' else workers

        # Run training in a separate process to avoid GUI + YOLO sharing memory (prevents OOM)
        cmd = [
            sys.executable, '-m', 'simba.model.yolo_fit',
            '--model_yaml', yolo_map_path,
            '--save_path', save_dir,
            '--epochs', str(epochs),
            '--batch', str(batch_size),
            '--plots', 'True' if plots else 'False',
            '--imgsz', str(imgsz),
            '--device', str(device_str),
            '--verbose', 'True' if verbose else 'False',
            '--workers', str(workers_for_subprocess),
            '--patience', str(patience),
        ]
        if weights_path is not None:
            cmd.extend(['--weights_path', weights_path])
        if format_val is not None:
            cmd.extend(['--format', format_val])

        creationflags = subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        # Use non-interactive matplotlib backend so label plots save to file without opening GUI (avoids hang)
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        try:
            if sys.platform == 'win32':
                # Run via a temp .bat so a visible console opens and stays open (pause) after exit
                cmd_line = subprocess.list2cmdline(cmd)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.bat', delete=False, newline='') as f:
                    f.write('@echo off\n' + cmd_line + '\npause\n')
                    bat_path = f.name
                subprocess.Popen([bat_path], creationflags=creationflags, env=env)
            else:
                subprocess.Popen(cmd, creationflags=creationflags, env=env)
        except Exception as e:
            messagebox.showerror('YOLO training', f'Failed to start training process: {e}')
            return
        msg = (
            'YOLO training has been started in a separate process to avoid memory issues.\n\n'
            'On Windows a new console window will show training progress. '
            'On other platforms, check the terminal from which SimBA was launched.\n\n'
            f'Results will be saved to:\n{save_dir}'
        )
        messagebox.showinfo('YOLO training started', msg)


#@YOLOPoseTrainPopUP()
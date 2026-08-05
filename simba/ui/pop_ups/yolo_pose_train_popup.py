import os
import subprocess
import sys
import tempfile
from tkinter import *
from tkinter import messagebox
from typing import Optional

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.transform.utils import \
    check_valid_yolo_map
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FileSelect,
                                        FolderSelect, SimBADropDown)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, is_torch_cuda_available)
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
# How long the training subprocess is watched for immediate death before it is assumed to be running
STARTUP_POLL_INTERVAL_MS = 1000
STARTUP_POLL_ATTEMPTS = 60
CONSOLE_TITLE = 'SimBA - YOLO POSE MODEL TRAINING'
FORMAT_OPTIONS =  Options.VALID_YOLO_FORMATS.value
FORMAT_OPTIONS.insert(0, 'None')


def _bat_echo_txt(txt: str) -> str:
    """Escape text so it survives ``echo`` inside a Windows batch file, e.g. paths holding ``&`` or ``%``."""
    for char in ('^', '&', '|', '<', '>', '(', ')'):
        txt = txt.replace(char, f'^{char}')
    return txt.replace('%', '%%')


class YOLOPoseTrainPopUP(PopUpMixin):
    def __init__(self):
        gpu_available, gpus = is_torch_cuda_available()
        if not gpu_available:
            raise SimBAGPUError(msg=f'Cannot train YOLO pose-estimation model. No NVIDA GPUs detected on machine', source=self.__class__.__name__)
        ultralytics_version = get_pkg_version(pkg=PackageNames.ULTRALYTICS.value)
        if ultralytics_version is None:
            raise SimBAPAckageVersionError(msg=f'Cannot train YOLO pose-estimation model: Could not find ultralytics package in python environment',  source=self.__class__.__name__)

        PopUpMixin.__init__(self, title="TRAIN YOLO POSE ESTIMATION MODEL", icon='ultralytics_2')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        device_options = devices + [f'{x} : {y["model"]}' for x, y in (gpus or {}).items()]
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
        self.devices_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=device_options, label="DEVICE:", label_width=35, dropdown_width=40, value=device_options[1] if len(device_options) > 1 else device_options[0], img='gpu_3', tooltip_key='devices_dropdown')

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
        device_str = 'cpu' if device == 'CPU' else device.split(':', 1)[0].strip()
        format_val = None if self.format_dropdown.get_value() == 'None' else self.format_dropdown.get_value()
        imgsz = int(self.img_size_dropdown.get_value())
        patience = int(self.patience_dropdown.get_value())
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} SAVE DIRECTORY')
        if not check_file_exist_and_readable(file_path=weights_path, raise_error=False):
            weights_path = None
        check_file_exist_and_readable(file_path=yolo_map_path, raise_error=True)
        check_valid_yolo_map(yolo_map=yolo_map_path)
        workers_for_subprocess = min(workers, 8) if sys.platform == 'win32' else workers
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
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        status_path = None
        try:
            if sys.platform == 'win32':
                tmp_dir = tempfile.mkdtemp(prefix='simba_yolo_train_')
                bat_path = os.path.join(tmp_dir, 'yolo_train.bat')
                status_path = os.path.join(tmp_dir, 'exit_code.txt')
                # A banner is printed before the python call: importing torch and ultralytics takes ~15s during
                # which the console is otherwise entirely silent and looks like nothing is happening.
                # The console window is kept open with `pause` so training output stays readable, which means the
                # shell outlives the python process - the exit code is written to file so it can be polled below.
                with open(bat_path, 'w', newline='') as f:
                    f.write('@echo off\n'
                            f'title {CONSOLE_TITLE}\n'
                            f'echo {"=" * 76}\n'
                            f'echo   {CONSOLE_TITLE}\n'
                            f'echo {"=" * 76}\n'
                            'echo.\n'
                            'echo   Loading python, pytorch and ultralytics ...\n'
                            'echo   This takes ~15-30s, no output is printed until it completes.\n'
                            'echo   Do NOT close this window - training runs inside it.\n'
                            'echo.\n'
                            f'echo   YOLO MAP:  {_bat_echo_txt(yolo_map_path)}\n'
                            f'echo   SAVE DIR:  {_bat_echo_txt(save_dir)}\n'
                            f'echo   DEVICE:    {_bat_echo_txt(device_str)}  ^|  EPOCHS: {epochs}  ^|  BATCH: {batch_size}  ^|  IMG SIZE: {imgsz}\n'
                            f'echo {"=" * 76}\n'
                            'echo.\n'
                            f'{subprocess.list2cmdline(cmd)}\n'
                            'set EC=%ERRORLEVEL%\n'
                            f'>"{status_path}" echo %EC%\n'
                            'echo.\n'
                            'if not "%EC%"=="0" echo   *** TRAINING EXITED WITH ERROR CODE %EC% - see the messages above. ***\n'
                            'if "%EC%"=="0" echo   *** TRAINING PROCESS ENDED. ***\n'
                            'echo.\n'
                            'pause\n')
                proc = subprocess.Popen([bat_path], creationflags=creationflags, env=env)
            else:
                proc = subprocess.Popen(cmd, creationflags=creationflags, env=env)
        except Exception as e:
            messagebox.showerror('YOLO training', f'Failed to start training process: {e}')
            return
        self.main_frm.after(STARTUP_POLL_INTERVAL_MS, lambda: self._check_startup(proc=proc, status_path=status_path, attempts_left=STARTUP_POLL_ATTEMPTS))
        msg = (
            'YOLO training has been started in a separate process to avoid memory issues.\n\n'
            'On Windows a new console window will show training progress. '
            'On other platforms, check the terminal from which SimBA was launched.\n\n'
            'If the training process terminates within the first '
            f'{int(STARTUP_POLL_INTERVAL_MS * STARTUP_POLL_ATTEMPTS / 1000)}s, an error will be reported here.\n\n'
            f'Results will be saved to:\n{save_dir}'
        )
        messagebox.showinfo('YOLO training started', msg)

    def _check_startup(self, proc: subprocess.Popen, status_path: Optional[str], attempts_left: int) -> None:
        """Poll the training subprocess and report an error if it terminates during the start-up window."""
        exit_code = None
        if status_path is None:
            exit_code = proc.poll()
        elif os.path.isfile(status_path):
            try:
                with open(status_path, 'r') as f:
                    exit_code_str = f.read().strip()
                exit_code = int(exit_code_str) if exit_code_str.lstrip('-').isdigit() else 'unknown'
            except OSError:
                exit_code = None
        if exit_code is not None:
            messagebox.showerror('YOLO training failed', f'The YOLO training process terminated immediately (exit code {exit_code}).\n\n'
                                                         'No model has been trained. See the training console window for the error message.')
        elif attempts_left > 1:
            self.main_frm.after(STARTUP_POLL_INTERVAL_MS, lambda: self._check_startup(proc=proc, status_path=status_path, attempts_left=attempts_left - 1))


#@YOLOPoseTrainPopUP()
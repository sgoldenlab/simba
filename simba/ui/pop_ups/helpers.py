from simba.ui.pop_ups.roi_video_table_pop_up import ROIVideoTable
from typing import Union
import os
from tkinter import Toplevel
from simba.utils.checks import check_instance, check_file_exist_and_readable

def restart_roi_video_table(roi_video_table_popup: Toplevel, config_path: Union[str, os.PathLike]):
    valid_frm = check_instance(source=f'{restart_roi_video_table.__name__} roi_table_popup', instance=roi_video_table_popup, accepted_types=(Toplevel,), raise_error=False)
    check_file_exist_and_readable(file_path=config_path)
    if valid_frm:
        roi_video_table_popup.destroy()
        ROIVideoTable(config_path=config_path)
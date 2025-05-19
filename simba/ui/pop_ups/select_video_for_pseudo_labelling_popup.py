import os
from tkinter import *
from typing import Union

from simba.labelling.standard_labeller import LabellingInterface
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        FileSelect, SimBALabel)
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_valid_dataframe)
from simba.utils.enums import Formats, Links, Options
from simba.utils.errors import NoDataError, NoFilesFoundError
from simba.utils.read_write import get_video_meta_data, read_df


class SelectPseudoLabellingVideoPupUp(ConfigReader, PopUpMixin):

    """
    Launch PopUp to select video for labelling

    :example:
    >>> SelectPseudoLabellingVideoPupUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
    """
    def __init__(self,
                 config_path: Union[str, os.PathLike]):


        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        PopUpMixin.__init__(self, title="SETTINGS - PSEUDO LABELLING")
        if len(self.clf_names) == 0:
            raise NoDataError(msg='To pseudo-label behaviors, your SimBA project needs at least one defined classifier. Found 0 classifiers defined in SimBA project', source=self.__class__.__name__)

        instructions_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='INSTRUCTIONS', icon_name='documentation', relief='solid', padx=5, pady=5)
        pseudo_intructions_lbl_1 = SimBALabel(parent=instructions_frm, txt="Note: SimBA pseudo-labelling require initial machine predictions", font=Formats.FONT_REGULAR.value)
        pseudo_intructions_lbl_2 = SimBALabel(parent=instructions_frm, txt="Click here more information on how to use the SimBA pseudo-labelling interface.", txt_clr='blue', cursor="hand2", font=Formats.FONT_REGULAR.value, link=Links.LABEL_BEHAVIOR.value)

        instructions_frm.grid(row=0, column=0, sticky=NSEW, padx=10, pady=10)
        pseudo_intructions_lbl_1.grid(row=0, column=0, sticky=NW)
        pseudo_intructions_lbl_2.grid(row=1, column=0, sticky=NW)

        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='PSEUDO-LABELLING SETTINGS', icon_name='settings', relief='solid', padx=5, pady=5)
        self.video_select = FileSelect(parent=settings_frm, fileDescription='SELECT VIDEO: ', lblwidth=45, file_types=[('VIDEO FILE', Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        settings_frm.grid(row=1, column=0, sticky=NSEW, padx=10, pady=10)
        self.video_select.grid(row=0, column=0, sticky=NW)
        self.clf_entry_boxes = {}
        for clf_cnt, clf_name in enumerate(self.clf_names):
            self.clf_entry_boxes[clf_name] = Entry_Box(parent=settings_frm, fileDescription=f'{clf_name} THRESHOLD:', labelwidth=45, entry_box_width=10, value=0.5, justify='center')
            self.clf_entry_boxes[clf_name].grid(row=clf_cnt+1, column=0, sticky=NW)
        self.create_run_frm(run_function=self._run)


    def _run(self):
        video_path = self.video_select.file_path
        check_file_exist_and_readable(file_path=video_path)
        video_meta_data = get_video_meta_data(video_path=video_path)
        thresholds = {}
        for clf, threshold_ebs in self.clf_entry_boxes.items():
            check_float(name=clf, value=threshold_ebs.entry_get, min_value=0.0, max_value=1.0, raise_error=True)
            thresholds[clf] = float(threshold_ebs.entry_get)
        self.machine_results_file_path = os.path.join(self.machine_results_dir, f"{video_meta_data['video_name']}.{self.file_type}")
        if not os.path.isfile(self.machine_results_file_path):
            raise NoFilesFoundError(msg=f'When doing pseudo-annotations, SimBA expects a file at {self.machine_results_file_path} representing video {video_meta_data["video_name"]}. SimBA could not find this file.', source=self.__class__.__name__)
        self.data_df = read_df(self.machine_results_file_path, self.file_type)
        check_valid_dataframe(df=self.data_df, source=self.__class__.__name__, required_fields=self.clf_names + self.p_cols)
        _ = LabellingInterface(config_path=self.config_path,
                               file_path=video_path,
                               thresholds=thresholds,
                               continuing=False)

        self.main_frm.destroy()

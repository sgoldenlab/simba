__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.pose_processors.pose_reset import PoseResetter
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        FileSelect, SimbaButton, SimBADropDown,
                                        TwoOptionQuestionPopUp)
from simba.utils.checks import check_str
from simba.utils.enums import ConfigKey, Keys, Links
from simba.utils.errors import DuplicationError, NoDataError
from simba.utils.printing import stdout_success, stdout_trash
from simba.utils.read_write import tabulate_clf_info


class AddClfPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, config_path=config_path, title="ADD CLASSIFIER", icon='plus')
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.clf_eb = Entry_Box(parent=self.main_frm, fileDescription="CLASSIFIER NAME:", labelwidth=25, entry_box_width=30, justify='center', img='decision_tree_small')
        add_btn = SimbaButton(parent=self.main_frm, txt="ADD CLASSIFIER", cmd=self.run, img='rocket')
        self.clf_eb.grid(row=0, column=0, sticky=NW)
        add_btn.grid(row=1, column=0, sticky=NW)

    def run(self):
        clf_name = self.clf_eb.entry_get.strip()
        check_str(name="CLASSIFIER NAME", value=clf_name)
        if clf_name in self.clf_names:
            raise DuplicationError(msg=f'The classifier name {clf_name} already exist in the SimBA project.', source=self.__class__.__name__)
        self.config.set( ConfigKey.SML_SETTINGS.value, ConfigKey.TARGET_CNT.value, str(self.clf_cnt + 1))
        self.config.set(ConfigKey.SML_SETTINGS.value, f"model_path_{str(self.clf_cnt + 1)}", "")
        self.config.set(ConfigKey.SML_SETTINGS.value, f"target_name_{str(self.clf_cnt + 1)}", clf_name)
        self.config.set(ConfigKey.THRESHOLD_SETTINGS.value, f"threshold_{str(self.clf_cnt + 1)}", "None")
        self.config.set(ConfigKey.MIN_BOUT_LENGTH.value, f"min_bout_{str(self.clf_cnt + 1)}", "None")
        with open(self.config_path, "w") as f:
            self.config.write(f)
        stdout_success(msg=f"{clf_name} classifier added to SimBA project", source=self.__class__.__name__)


class RemoveAClassifierPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if not isinstance(self.clf_names, (list, tuple)) or len(self.clf_names) < 1:
            raise NoDataError(msg='The SimBA project has no classifiers: Cannot remove a classifier.', source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="WARNING: REMOVE CLASSIFIER", icon='trash_red')
        self.remove_clf_frm = CreateLabelFrameWithIcon( parent=self.main_frm, header="SELECT A CLASSIFIER TO REMOVE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.REMOVE_CLF.value)
        self.clf_dropdown = SimBADropDown(parent=self.remove_clf_frm, dropdown_options=self.clf_names, label_width=20, dropdown_width=40, label='CLASSIFIER:', value=self.clf_names[0])
        run_btn = SimbaButton(parent=self.main_frm, txt="REMOVE CLASSIFIER", cmd=self.run, img='trash')
        self.remove_clf_frm.grid(row=0, column=0, sticky=W)
        self.clf_dropdown.grid(row=0, column=0, sticky=W)
        run_btn.grid(row=1, pady=10)

    def run(self):
        clf_to_remove = self.clf_dropdown.get_value()
        question = TwoOptionQuestionPopUp(title="WARNING!", question=f"Do you want to remove the {clf_to_remove} \nclassifier from the SimBA project?", option_one="YES", option_two="NO")
        if question.selected_option == "YES":
            for i in range(len(self.clf_names)):
                self.config.remove_option("SML settings", f"model_path_{i+1}")
                self.config.remove_option("SML settings", f"target_name_{i+1}")
                self.config.remove_option("threshold_settings", f"threshold_{i+1}")
                self.config.remove_option("Minimum_bout_lengths", f"min_bout_{i+1}")
            self.clf_names.remove(self.clf_dropdown.getChoices())
            self.config.set("SML settings", "no_targets", str(len(self.clf_names)))

            for clf_cnt, clf_name in enumerate(self.clf_names):
                self.config.set("SML settings", f"model_path_{clf_cnt + 1}", "")
                self.config.set("SML settings", f"target_name_{clf_cnt + 1}", clf_name)
                self.config.set("threshold_settings", f"threshold_{clf_cnt + 1}", "None")
                self.config.set("Minimum_bout_lengths", f"min_bout_{clf_cnt + 1}", "None")

            with open(self.config_path, "w") as f:
                self.config.write(f)

            stdout_trash(msg=f"{self.clf_dropdown.getChoices()} classifier removed from SimBA project.", source=self.__class__.__name__)
        else:
            pass


# _ = RemoveAClassifierPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')


class PrintModelInfoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="PRINT MACHINE MODEL INFO", size=(250, 250))
        model_info_frame = LabelFrame(
            self.main_frm, text="PRINT MODEL INFORMATION", padx=5, pady=5, font="bold"
        )
        model_path_selector = FileSelect(
            model_info_frame, "Model path", title="Select a video file"
        )
        btn_print_info = Button(
            model_info_frame,
            text="PRINT MODEL INFO",
            command=lambda: tabulate_clf_info(clf_path=model_path_selector.file_path),
        )
        model_info_frame.grid(row=0, sticky=W)
        model_path_selector.grid(row=0, sticky=W, pady=5)
        btn_print_info.grid(row=1, sticky=W)


class PoseResetterPopUp:
    def __init__(self):
        # PopUpMixin.__init__(self, title="WARNING!", size=(300, 100))
        # popupframe = LabelFrame(self.main_frm)
        # label = Label(popupframe, text="Do you want to remove user-defined pose-configurations?")
        # label.grid(row=0, columnspan=2)
        # B1 = Button(popupframe,text="YES",fg="blue",command=lambda: PoseResetter(master=self.main_frm))
        # B2 = Button(popupframe, text="NO", fg="red", command=self.main_frm.destroy)
        question = TwoOptionQuestionPopUp(
            title="WARNING!",
            question="Do you want to remove user-defined pose-configurations?",
            option_one="YES",
            option_two="NO",
        )
        if question.selected_option == "YES":
            _ = PoseResetter(master=None)
        else:
            pass

        # popupframe.grid(row=0, columnspan=2)
        # B1.grid(row=1, column=0, sticky=W)
        # B2.grid(row=1, column=1, sticky=W)
        # self.main_frm.mainloop()

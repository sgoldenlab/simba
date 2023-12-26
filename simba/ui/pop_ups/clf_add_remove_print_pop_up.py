__author__ = "Simon Nilsson"

from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.enums import ConfigKey, Keys, Links
from simba.ui.tkinter_functions import DropDownMenu, FileSelect, Entry_Box, CreateLabelFrameWithIcon
from simba.utils.printing import stdout_success, stdout_trash
from simba.utils.checks import check_str
from simba.utils.read_write import tabulate_clf_info
from simba.pose_processors.pose_reset import PoseResetter


class AddClfPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):
        PopUpMixin.__init__(self, config_path=config_path, title='ADD CLASSIFIER')
        ConfigReader.__init__(self, config_path=config_path)
        self.clf_eb = Entry_Box(self.main_frm,'CLASSIFIER NAME', '15')
        add_btn = Button(self.main_frm, text='ADD CLASSIFIER', command=lambda: self.run())
        self.clf_eb.grid(row=0, column=0, sticky=NW)
        add_btn.grid(row=1, column=0, sticky=NW)

    def run(self):
        clf_name = self.clf_eb.entry_get.strip()
        check_str(name='CLASSIFIER NAME', value=clf_name)
        self.config.set(ConfigKey.SML_SETTINGS.value, ConfigKey.TARGET_CNT.value, str(self.clf_cnt + 1))
        self.config.set(ConfigKey.SML_SETTINGS.value, f'model_path_{str(self.clf_cnt + 1)}', '')
        self.config.set(ConfigKey.SML_SETTINGS.value, f'target_name_{str(self.clf_cnt + 1)}', clf_name)
        self.config.set(ConfigKey.THRESHOLD_SETTINGS.value, f'threshold_{str(self.clf_cnt + 1)}', 'None')
        self.config.set(ConfigKey.MIN_BOUT_LENGTH.value, f'min_bout_{str(self.clf_cnt + 1)}', 'None')
        with open(self.config_path, 'w') as f:
            self.config.write(f)
        stdout_success(msg=f'{clf_name} classifier added to SimBA project', source=self.__class__.__name__)


class RemoveAClassifierPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, title='Warning: Remove classifier(s) settings')
        ConfigReader.__init__(self, config_path=config_path)
        self.remove_clf_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SELECT A CLASSIFIER TO REMOVE', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.REMOVE_CLF.value)
        self.clf_dropdown = DropDownMenu(self.remove_clf_frm, 'Classifier', self.clf_names, '12')
        self.clf_dropdown.setChoices(self.clf_names[0])

        run_btn = Button(self.main_frm,text='REMOVE CLASSIFIER',command=lambda:self.run())
        self.remove_clf_frm.grid(row=0,sticky=W)
        self.clf_dropdown.grid(row=0,sticky=W)
        run_btn.grid(row=1,pady=10)

    def run(self):
        for i in range(len(self.clf_names)):
            self.config.remove_option('SML settings', 'model_path_{}'.format(str(i+1)))
            self.config.remove_option('SML settings', 'target_name_{}'.format(str(i+1)))
            self.config.remove_option('threshold_settings', 'threshold_{}'.format(str(i+1)))
            self.config.remove_option('Minimum_bout_lengths', 'min_bout_{}'.format(str(i+1)))
        self.clf_names.remove(self.clf_dropdown.getChoices())
        self.config.set('SML settings', 'no_targets', str(len(self.clf_names)))

        for clf_cnt, clf_name in enumerate(self.clf_names):
            self.config.set('SML settings', 'model_path_{}'.format(str(clf_cnt+1)), '')
            self.config.set('SML settings', 'target_name_{}'.format(str(clf_cnt+1)), clf_name)
            self.config.set('threshold_settings', 'threshold_{}'.format(str(clf_cnt+1)), 'None')
            self.config.set('Minimum_bout_lengths', 'min_bout_{}'.format(str(clf_cnt+1)), 'None')

        with open(self.config_path, 'w') as f:
            self.config.write(f)

        stdout_trash(msg=f'{self.clf_dropdown.getChoices()} classifier removed from SimBA project.', source=self.__class__.__name__)

#_ = RemoveAClassifierPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')

class PrintModelInfoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="PRINT MACHINE MODEL INFO", size=(250, 250))
        model_info_frame = LabelFrame(self.main_frm, text='PRINT MODEL INFORMATION', padx=5, pady=5, font='bold')
        model_path_selector = FileSelect(model_info_frame, 'Model path', title='Select a video file')
        btn_print_info = Button(model_info_frame,text='PRINT MODEL INFO',command=lambda: tabulate_clf_info(clf_path=model_path_selector.file_path))
        model_info_frame.grid(row=0, sticky=W)
        model_path_selector.grid(row=0, sticky=W, pady=5)
        btn_print_info.grid(row=1, sticky=W)

class PoseResetterPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="WARNING!", size=(300, 100))
        popupframe = LabelFrame(self.main_frm)
        label = Label(popupframe, text='Do you want to remove user-defined pose-configurations?')
        label.grid(row=0,columnspan=2)
        B1 = Button(popupframe, text='YES', fg='blue', command=lambda: PoseResetter(master=self.main_frm))
        B2 = Button(popupframe, text="NO", fg='red', command=self.main_frm.destroy)
        popupframe.grid(row=0,columnspan=2)
        B1.grid(row=1,column=0,sticky=W)
        B2.grid(row=1,column=1,sticky=W)
        self.main_frm.mainloop()

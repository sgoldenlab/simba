import os.path
from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FileSelect,
                                        FolderSelect, SimbaButton,
                                        SimBADropDown)
from simba.utils.checks import check_if_dir_exists, check_nvidea_gpu_available
from simba.utils.enums import Options
from simba.utils.read_write import (find_all_videos_in_directory,
                                    find_core_cnt, get_video_meta_data,
                                    str_2_bool)
from simba.video_processors.videos_to_frames import video_to_frames

ALL_IMAGES = 'ALL IMAGES'
batch_size_options = list(range(100, 5100, 100))
batch_size_options.insert(0, ALL_IMAGES)

class MultipleVideos2FramesPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="EXTRACT FRAMES FROM SINGLE VIDEO", icon='frames')
        gpu_available = check_nvidea_gpu_available()
        gpu_state = DISABLED if not gpu_available else NORMAL
        core_cnt = find_core_cnt()[0]
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name='settings', padx=5, pady=5, relief='solid')

        self.video_dir = FolderSelect(parent=settings_frm, folderDescription="VIDEO DIRECTORY:", lblwidth=40)
        self.save_dir = FolderSelect(parent=settings_frm, folderDescription="SAVE DIRECTORY:", lblwidth=40)
        self.gpu_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label='USE GPU: ', label_width=40, state=gpu_state, value='FALSE')
        self.core_cnt_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=list(range(1, core_cnt+1)), label='CORE COUNT: ', label_width=40, value=int(core_cnt/2))
        self.quality_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=list(range(1, 101)), label='IMAGE QUALITY: ', label_width=40, value=90)
        self.img_format = SimBADropDown(parent=settings_frm, dropdown_options=['jpeg', 'png', 'webp'], label='IMAGE FORMAT: ', label_width=40, value='jpeg', command=self._inactivate_quality)
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label='VERBOSE: ', label_width=40, value='TRUE')
        self.batch_size_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=batch_size_options, label='BATCH SIZE: ', label_width=40, value=batch_size_options[0])

        settings_frm.grid(row=0, column=0, sticky=NW, padx=10, pady=10)
        self.video_dir.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)

        self.gpu_dropdown.grid(row=2, column=0, sticky=NW)
        self.core_cnt_dropdown.grid(row=3, column=0, sticky=NW)
        self.quality_dropdown.grid(row=4, column=0, sticky=NW)
        self.img_format.grid(row=5, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=6, column=0, sticky=NW)
        self.batch_size_dropdown.grid(row=7, column=0, sticky=NW)

        run_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='RUN', icon_name='run', padx=5, pady=5, relief='solid')
        run_btn = SimbaButton(parent=run_frm, txt='RUN', img='rocket', cmd=self._run)

        run_frm.grid(row=1, column=0, sticky=NW, padx=10, pady=10)
        run_btn.grid(row=0, column=0, sticky=NW)
        self.main_frm.mainloop()

    def _inactivate_quality(self, img_format):
        if img_format != 'jpeg': self.quality_dropdown.disable()
        else: self.quality_dropdown.enable()

    def _run(self):
        video_dir = self.video_dir.folder_path
        video_dict = find_all_videos_in_directory(directory=video_dir, as_dict=True, raise_error=True)
        save_dir = self.save_dir.folder_path
        gpu = str_2_bool(self.gpu_dropdown.get_value())
        core_cnt = int(self.core_cnt_dropdown.get_value())
        quality = int(self.quality_dropdown.get_value())
        verbose = str_2_bool(self.verbose_dropdown.get_value())
        img_format = self.img_format.get_value()
        check_if_dir_exists(in_dir=save_dir)
        batch_size = None if self.batch_size_dropdown.get_value() == ALL_IMAGES else int(self.batch_size_dropdown.get_value())

        for video_cnt, (video_name, video_path) in enumerate(video_dict.items()):
            video_meta_data = get_video_meta_data(video_path=video_path)
            video_save_dir = os.path.join(save_dir, video_meta_data["video_name"])
            print(f'Extracting frames from {video_meta_data["video_name"]} and saving at {video_save_dir} (video {video_cnt+1})....')
            video_to_frames(video_path=video_path,
                            save_dir=save_dir,
                            gpu=gpu,
                            batch_size=batch_size,
                            core_cnt=core_cnt,
                            quality=quality,
                            img_format=img_format,
                            verbose=verbose)

#MovementAnalysisPopUp()




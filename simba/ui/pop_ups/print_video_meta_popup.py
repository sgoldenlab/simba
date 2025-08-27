from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.get_tree_view import GetTreeView
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FileSelect,
                                        FolderSelect, SimbaButton)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.enums import Options
from simba.utils.lookups import print_video_meta_data
from simba.utils.read_write import (find_all_videos_in_directory,
                                    get_video_meta_data)


class PrintVideoMetaDataPopUp(PopUpMixin):

    def __init__(self):

        PopUpMixin.__init__(self, title="PRINT VIDEO META DATA", icon='video')
        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO", icon_name='video')
        self.video_path = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        file_btn = SimbaButton(parent=single_video_frm, txt='RUN', txt_clr='navy', img='rocket', cmd=self.run, cmd_kwargs={'directory': False})

        multiple_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS", icon_name='stack')
        self.directory_path = FolderSelect(multiple_video_frm, "VIDEO DIRECTORY PATH:",  title="Select folder with videos: ", lblwidth=25)
        dir_btn = SimbaButton(parent=multiple_video_frm, txt='RUN', txt_clr='navy', img='rocket', cmd=self.run, cmd_kwargs={'directory': True})

        single_video_frm.grid(row=0, column=0, sticky=NW)
        self.video_path.grid(row=0, column=0, sticky=NW)
        file_btn.grid(row=1, column=0, sticky=NW)

        multiple_video_frm.grid(row=1, column=0, sticky=NW)
        self.directory_path.grid(row=0, column=0, sticky=NW)
        dir_btn.grid(row=1, column=0, sticky=NW)
        self.main_frm.mainloop()

    def run(self, directory: bool):
        results = {}
        if directory:
            check_if_dir_exists(in_dir=self.directory_path.folder_path)
            videos = find_all_videos_in_directory(directory=self.directory_path.folder_path, as_dict=True)
            for video_name, video_path in videos.items():
                v = get_video_meta_data(video_path=video_path, fps_as_int=False)
                results[v['video_name']] = v
        else:
            check_file_exist_and_readable(file_path=self.video_path.file_path)
            v = get_video_meta_data(video_path=self.video_path.file_path, fps_as_int=False)
            results[v['video_name']] = v

        _ = GetTreeView(data=results, index_col_name='VIDEO', headers=('VALUE',), title='VIDEO META DATA')

#PrintVideoMetaDataPopUp()
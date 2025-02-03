import os
import warnings
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.image_mixin import ImageMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.roi_ui_mixin import ROI_mixin
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.enums import ROI_SETTINGS
from simba.utils.read_write import (find_all_videos_in_directory, get_fn_ext,
                                    get_video_meta_data)

warnings.filterwarnings("ignore")


WINDOW_SIZE = (775, 900)

class ROI_ui(ROI_mixin, ConfigReader):

    """

    :example:
    >>> ROI_ui(config_path=r"C:\troubleshooting\mouse_open_field\project_folder\project_config.ini", video_path=r"C:\troubleshooting\mouse_open_field\project_folder\videos\Video1.mp4")
    >>> ROI_ui(config_path=r"/mnt/c/troubleshooting/mouse_open_field/project_folder/project_config.ini", video_path=r"/mnt/c/troubleshooting/mouse_open_field/project_folder/videos/Video1.mp4")

    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike]):

        check_file_exist_and_readable(file_path=config_path)
        check_file_exist_and_readable(file_path=video_path)
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        self.video_meta =  get_video_meta_data(video_path=video_path, fps_as_int=False)
        self.video_ext = get_fn_ext(filepath=video_path)[2][1:]
        self.img, self.img_idx = ImageMixin.find_first_non_uniform_clr_frm(video_path=video_path)
        self.define_ui = PopUpMixin(title="REGION OF INTEREST (ROI) SETTINGS", size=WINDOW_SIZE)
        ROI_mixin.__init__(self, video_path=video_path, config_path=config_path, img_idx=self.img_idx)
        self.other_project_video_paths = find_all_videos_in_directory(directory=self.video_dir, as_dict=True).values()
        self.other_project_video_paths = [x for x in self.other_project_video_paths if x != video_path]
        self.other_project_video_names = [get_fn_ext(x)[1] for x in self.other_project_video_paths]
        self.settings = {item.name: item.value for item in ROI_SETTINGS}
        self.text_size, _, _ = PlottingMixin().get_optimal_font_scales(text='TEN DIGITS', accepted_px_width=int(self.video_meta['width']/10), accepted_px_height=int(self.video_meta['height']/10), text_thickness=self.settings['TEXT_THICKNESS'], font=self.settings['FONT'])
        self.get_video_info_panel(parent_frame=self.define_ui.main_frm, row_idx=0)
        self.get_select_img_panel(parent_frame=self.define_ui.main_frm, row_idx=1)
        self.get_select_shape_type_panel(parent_frame=self.define_ui.main_frm, row_idx=2)
        self.get_shape_attr_panel(parent_frame=self.define_ui.main_frm, row_idx=3)
        self.get_shape_name_panel(parent_frame=self.define_ui.main_frm, row_idx=4)
        self.get_interact_panel(parent_frame=self.define_ui.main_frm, row_idx=5)
        self.get_draw_panel(parent_frame=self.define_ui.main_frm, row_idx=6)
        self.get_shapes_from_other_video_panel(parent_frame=self.define_ui.main_frm, row_idx=7)
        self.get_save_roi_panel(parent_frame=self.define_ui.main_frm, row_idx=8)
        self.get_status_bar_panel(parent_frame=self.define_ui.main_frm, row_idx=9)
        self.set_img(img=self.img)
        self.get_file_menu(root=self.define_ui.root)
        self.define_ui.root.protocol("WM_DELETE_WINDOW", self._close)
        self.define_ui.main_frm.mainloop()

    def _close(self):
        self.close_img()
        self.define_ui.root.destroy()

#ROI_ui(config_path=r"/mnt/c/troubleshooting/mouse_open_field/project_folder/project_config.ini", video_path=r"/mnt/c/troubleshooting/mouse_open_field/project_folder/videos/Video1.mp4")




#ROI_ui(config_path=r"C:\troubleshooting\mouse_open_field\project_folder\project_config.ini", video_path=r"C:\troubleshooting\mouse_open_field\project_folder\videos\Video1.mp4")




import threading
from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.ez_path_plot import EzPathPlot
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, FileSelect)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_valid_rgb_tuple, check_int, check_str)
from simba.utils.enums import Keys, Links, Options
from simba.utils.lookups import get_color_dict


class MakePathPlotPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CREATE SIMPLE PATH PLOT", size=(500, 300))
        settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SETTINGS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        self.video_path = FileSelect(
            settings_frm,
            "VIDEO PATH: ",
            lblwidth="30",
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        self.body_part = Entry_Box(settings_frm, "BODY PART: ", "30")
        self.data_path = FileSelect(
            settings_frm, "DATA PATH (e.g., H5 or CSV file): ", lblwidth="30"
        )
        color_lst = list(get_color_dict().keys())
        self.background_color = DropDownMenu(
            settings_frm, "BACKGROUND COLOR: ", color_lst, "30"
        )
        self.background_color.setChoices(choice="White")
        self.line_color = DropDownMenu(settings_frm, "LINE COLOR: ", color_lst, "30")
        self.line_color.setChoices(choice="Red")
        self.line_thickness = DropDownMenu(
            settings_frm, "LINE THICKNESS: ", list(range(1, 11)), "30"
        )
        self.line_thickness.setChoices(choice=1)
        self.circle_size = DropDownMenu(
            settings_frm, "CIRCLE SIZE: ", list(range(1, 11)), "30"
        )
        self.last_frm_only_dropdown = DropDownMenu(
            settings_frm, "LAST FRAME ONLY: ", ["TRUE", "FALSE"], "30"
        )
        self.last_frm_only_dropdown.setChoices("FALSE")
        self.circle_size.setChoices(choice=5)
        settings_frm.grid(row=0, sticky=W)
        self.video_path.grid(row=0, sticky=W)
        self.data_path.grid(row=1, sticky=W)
        self.body_part.grid(row=2, sticky=W)
        self.background_color.grid(row=3, sticky=W)
        self.line_color.grid(row=4, sticky=W)
        self.line_thickness.grid(row=5, sticky=W)
        self.circle_size.grid(row=6, sticky=W)
        self.last_frm_only_dropdown.grid(row=7, sticky=W)
        Label(
            settings_frm,
            fg="green",
            text=" NOTE: For more complex path plots, faster, \n see 'CREATE PATH PLOTS' under the [VISUALIZATIONS] tab after loading your SimBA project",
        ).grid(row=8, sticky=W)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        data_path = self.data_path.file_path
        video_path = self.video_path.file_path
        background_color = get_color_dict()[self.background_color.getChoices()]
        line_color = get_color_dict()[self.line_color.getChoices()]
        line_thickness = self.line_thickness.getChoices()
        circle_size = self.circle_size.getChoices()
        bp = self.body_part.entry_get
        check_file_exist_and_readable(file_path=data_path)
        check_int(
            name=f"{self.__class__.__name__} line_thickness",
            value=line_thickness,
            min_value=1,
        )
        check_int(
            name=f"{self.__class__.__name__} circle_size",
            value=circle_size,
            min_value=1,
        )
        check_if_valid_rgb_tuple(data=background_color)
        check_if_valid_rgb_tuple(data=line_color)
        check_str(name=f"{self.__class__.__name__} body-part", value=bp)
        last_frm = self.last_frm_only_dropdown.getChoices()
        if last_frm == "TRUE":
            last_frm = True
        else:
            last_frm = False
        plotter = EzPathPlot(
            data_path=data_path,
            video_path=video_path,
            body_part=bp,
            bg_color=background_color,
            line_color=line_color,
            line_thickness=int(line_thickness),
            circle_size=int(circle_size),
            last_frm_only=last_frm,
        )
        threading.Thread(target=plotter.run).start()


# MakePathPlotPopUp()

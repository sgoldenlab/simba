import threading
from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.ez_path_plot import EzPathPlot
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        FileSelect, SimBADropDown, SimBALabel)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_valid_rgb_tuple, check_int, check_str)
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import str_2_bool


class MakePathPlotPopUp(PopUpMixin):
    """
    Tkinter pop-up window for creating simple path plots from pose estimation data.

    This pop-up provides a simplified interface to visualize animal movement paths
    from pose estimation data files (H5 or CSV format). It creates a path plot
    showing where a selected body-part traveled over time, displayed as a line
    connecting sequential positions.

    The path plot can be created as either:
    - A video showing the path growing frame-by-frame
    - A single image showing the complete cumulative path

    This tool works independently of SimBA projects and can be used with any
    pose estimation data file. For more advanced path plots with multiple animals,
    ROI overlays, classification markers, and other features, use the full
    "CREATE PATH PLOTS" tool available in the SimBA project interface.

    :example:
    >>> popup = MakePathPlotPopUp()
    >>> # User selects video, data file, body-part, and styling options
    >>> # Then clicks RUN to generate the path plot
    """
    def __init__(self):
        PopUpMixin.__init__(self, title="CREATE SIMPLE PATH PLOT", size=(600, 300), icon='path_2')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.video_path = FileSelect(settings_frm, "VIDEO PATH: ", lblwidth=30, entry_width=30, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)], lbl_icon='video_2', tooltip_key='SIMPLE_PATH_PLOT_VIDEO_PATH')
        self.body_part = Entry_Box(settings_frm, fileDescription= "BODY PART: ", labelwidth=30, entry_box_width=30, img='nose', justify='center', tooltip_key='SIMPLE_PATH_PLOT_BODY_PART')
        self.data_path = FileSelect(settings_frm, fileDescription="DATA PATH (e.g., H5 or CSV file): ", lblwidth=30, entry_width=30, lbl_icon='file', tooltip_key='SIMPLE_PATH_PLOT_DATA_PATH')
        color_lst = list(get_color_dict().keys())

        self.background_color = SimBADropDown(parent=settings_frm, label="BACKGROUND COLOR: ", dropdown_options=color_lst, label_width=30, dropdown_width=30, img='fill', value="White", tooltip_key='SIMPLE_PATH_PLOT_BACKGROUND_COLOR')
        self.line_color = SimBADropDown(parent=settings_frm, label="LINE COLOR: ", dropdown_options=color_lst, label_width=30, dropdown_width=30, img='line', value="Red", tooltip_key='SIMPLE_PATH_PLOT_LINE_COLOR')
        self.line_thickness = SimBADropDown(parent=settings_frm, label="LINE THICKNESS: ", dropdown_options=list(range(1, 11)), label_width=30, dropdown_width=30, img='bold', value=1, tooltip_key='SIMPLE_PATH_PLOT_LINE_THICKNESS')
        self.circle_size = SimBADropDown(parent=settings_frm, label="CIRCLE SIZE: ", dropdown_options=list(range(1, 11)), label_width=30, dropdown_width=30, img='circle_small', value=5, tooltip_key='SIMPLE_PATH_PLOT_CIRCLE_SIZE')
        self.last_frm_only_dropdown = SimBADropDown(parent=settings_frm, label="LAST FRAME ONLY: ", dropdown_options=["TRUE", "FALSE"], label_width=30, dropdown_width=30, img='finish', value='TRUE', tooltip_key='SIMPLE_PATH_PLOT_LAST_FRAME_ONLY')

        settings_frm.grid(row=0, sticky=W)
        self.video_path.grid(row=0, sticky=W)
        self.data_path.grid(row=1, sticky=W)
        self.body_part.grid(row=2, sticky=W)
        self.background_color.grid(row=3, sticky=W)
        self.line_color.grid(row=4, sticky=W)
        self.line_thickness.grid(row=5, sticky=W)
        self.circle_size.grid(row=6, sticky=W)
        self.last_frm_only_dropdown.grid(row=7, sticky=W)
        SimBALabel(parent=settings_frm, font=Formats.FONT_REGULAR_ITALICS.value, txt=" NOTE: For more complex path plots, faster, \n see 'CREATE PATH PLOTS' under the [VISUALIZATIONS] tab after loading your SimBA project", txt_clr="green").grid(row=8, sticky=W)
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
        last_frm = str_2_bool(self.last_frm_only_dropdown.getChoices())
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


#MakePathPlotPopUp()

import os
import platform
from configparser import ConfigParser
from tkinter import *

from simba.roi_tools import ROI_image
from simba.roi_tools.ROI_define import ROI_definitions
from simba.roi_tools.ROI_image import ROI_image_class
from simba.roi_tools.ROI_multiply import multiply_ROIs
from simba.roi_tools.ROI_reset import reset_video_ROIs
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon
from simba.utils.enums import ConfigKey, Keys, Links
from simba.utils.errors import NoFilesFoundError
from simba.utils.lookups import get_color_dict


class ROI_menu:
    def __init__(self, config_path, new_roi=True):
        self.config_path = config_path
        config = ConfigParser()
        config.read(config_path)
        self.project_path = config.get(
            ConfigKey.GENERAL_SETTINGS.value, ConfigKey.PROJECT_PATH.value
        )
        self.measures_dir = os.path.join(self.project_path, "logs", "measures")
        self.video_dir = os.path.join(self.project_path, "videos")
        self.duplicate_jump_size = 20
        self.click_sens = 10
        self.text_size = 5
        self.text_thickness = 3
        self.line_type = -1
        self.named_shape_colors = get_color_dict()
        self.roi_table_menu()

    def roi_table_menu(self):
        self.filesFound = []
        self.row = []
        for i in os.listdir(self.video_dir):
            if i.endswith((".avi", ".mp4", ".mov", "flv")):
                self.filesFound.append(i)
        if len(self.filesFound) == 0:
            raise NoFilesFoundError(
                "No videos found the SimBA project (no avi, mp4, mov, flv files in the project_folder/videos directory)"
            )

        maxname = max(self.filesFound, key=len)
        self.roimenu = Toplevel()
        self.roimenu.minsize(720, 960)
        self.roimenu.wm_title("ROI Table")

        self.scroll_window = hxtScrollbar(self.roimenu)
        tableframe = CreateLabelFrameWithIcon(
            parent=self.scroll_window,
            header="Video Name",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.ROI.value,
        )
        # tableframe = LabelFrame(self.scroll_window, text='Video Name', labelanchor=NW)

        self.image_data = ROI_image_class(
            self.config_path,
            self.named_shape_colors,
            self.duplicate_jump_size,
            self.line_type,
            self.click_sens,
            self.text_size,
            self.text_thickness,
        )
        for i in range(len(self.filesFound)):
            self.row.append(
                roitableRow(
                    self.image_data,
                    tableframe,
                    self.video_dir,
                    str(self.filesFound[i]),
                    str(len(maxname) + 20),
                    str(i + 1) + ".",
                    projectini=self.config_path,
                )
            )
            self.row[i].grid(row=i + 1, sticky=W)
        tableframe.grid(row=0)


class roitableRow(Frame):
    def __init__(
        self,
        image_data: ROI_image,
        parent=None,
        dirname="",
        filename="",
        widths="",
        indexs="",
        projectini="",
    ):
        self.image_data = image_data
        self.projectini = projectini
        self.filename = os.path.join(dirname, filename)
        Frame.__init__(self, master=parent)
        if not os.path.exists(self.filename):
            if os.path.islink(self.filename):
                filename += " symlink"
            filename += " path not exists"
        var = StringVar()
        self.index = Entry(self, textvariable=var, width=4)
        var.set(indexs)
        self.index.grid(row=0, column=0)
        self.lblName = Label(self, text=filename, width=widths, anchor=W)
        self.lblName.grid(row=0, column=1, sticky=W)
        if os.path.exists(self.filename):
            self.btnset = Button(self, text="Draw", command=self.draw)
            self.btnset.grid(row=0, column=2)
            self.btnreset = Button(self, text="Reset", command=self.reset)
            self.btnreset.grid(row=0, column=3)
            self.btnapplyall = Button(self, text="Apply to all", command=self.applyall)
            self.btnapplyall.grid(row=0, column=4)

    def draw(self):
        ROI_definitions(self.projectini, self.filename, self.image_data)

    def reset(self):
        reset_video_ROIs(self.projectini, self.filename)

    def applyall(self):
        multiply_ROIs(self.projectini, self.filename)


def onMousewheel(event, canvas):
    try:
        scrollSpeed = event.delta
        if platform.system() == "Darwin":
            scrollSpeed = event.delta
        elif platform.system() == "Windows":
            scrollSpeed = int(event.delta / 120)
        canvas.yview_scroll(-1 * (scrollSpeed), "units")
    except:
        pass


def bindToMousewheel(event, canvas):
    canvas.bind_all("<MouseWheel>", lambda event: onMousewheel(event, canvas))


def unbindToMousewheel(event, canvas):
    canvas.unbind_all("<MouseWheel>")


def onFrameConfigure(canvas):
    """Reset the scroll region to encompass the inner frame"""
    canvas.configure(scrollregion=canvas.bbox("all"))


def hxtScrollbar(master):
    """
    Create canvas.
    Create a frame and put it in the canvas.
    Create two scrollbar and insert command of canvas x and y view
    Use canvas to create a window, where window = frame
    Bind the frame to the canvas
    """

    bg = master.cget("background")
    acanvas = Canvas(master, borderwidth=0, background=bg)
    vsb = Scrollbar(master, orient="vertical", command=acanvas.yview)
    vsb2 = Scrollbar(master, orient="horizontal", command=acanvas.xview)
    acanvas.configure(yscrollcommand=vsb.set)
    acanvas.configure(xscrollcommand=vsb2.set)
    vsb.pack(side=RIGHT, fill="y")
    vsb2.pack(side=BOTTOM, fill="x")
    acanvas.pack(side="left", fill="both", expand=True)

    # bind the frame to the canvas
    acanvas.bind("<Configure>", lambda event, canvas=acanvas: onFrameConfigure(acanvas))
    acanvas.bind("<Enter>", lambda event: bindToMousewheel(event, acanvas))
    acanvas.bind("<Leave>", lambda event: unbindToMousewheel(event, acanvas))
    frame = Frame(acanvas, background=bg)
    acanvas.create_window((10, 10), window=frame, anchor="nw")

    return frame

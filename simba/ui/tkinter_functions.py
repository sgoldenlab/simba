__author__ = "Simon Nilsson"

import os.path
import platform
import webbrowser
from tkinter import *
from tkinter.filedialog import askdirectory, askopenfilename
from typing import Optional, Union

import PIL.Image
from PIL import ImageTk

from simba.utils.enums import Defaults, Formats
from simba.utils.lookups import get_icons_paths

MENU_ICONS = get_icons_paths()


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
    acanvas = Canvas(
        master,
        borderwidth=0,
        background=bg,
        width=master.winfo_width(),
        height=master.winfo_reqheight(),
    )
    frame = Frame(acanvas, background=bg)
    vsb = Scrollbar(master, orient="vertical", command=acanvas.yview)
    vsb2 = Scrollbar(master, orient="horizontal", command=acanvas.xview)
    acanvas.configure(yscrollcommand=vsb.set)
    acanvas.configure(xscrollcommand=vsb2.set)
    vsb.pack(side=RIGHT, fill="y")
    vsb2.pack(side=BOTTOM, fill="x")
    acanvas.pack(side=LEFT, fill=BOTH, expand=True)

    acanvas.create_window((10, 10), window=frame, anchor="nw")

    # bind the frame to the canvas
    acanvas.bind("<Configure>", lambda event, canvas=acanvas: onFrameConfigure(acanvas))
    acanvas.bind("<Enter>", lambda event: bindToMousewheel(event, acanvas))
    acanvas.bind("<Leave>", lambda event: unbindToMousewheel(event, acanvas))
    acanvas.update()
    return frame


def form_validator_is_numeric(inStr, acttyp):
    if acttyp == "1":  # insert
        if not inStr.isdigit():
            return False
    return True


class DropDownMenu(Frame):
    def __init__(
        self,
        parent=None,
        dropdownLabel="",
        choice_dict=None,
        labelwidth="",
        com=None,
        **kw
    ):
        Frame.__init__(self, master=parent, **kw)
        self.dropdownvar = StringVar()
        self.lblName = Label(self, text=dropdownLabel, width=labelwidth, anchor=W)
        self.lblName.grid(row=0, column=0)
        self.choices = choice_dict
        self.popupMenu = OptionMenu(self, self.dropdownvar, *self.choices, command=com)
        self.popupMenu.grid(row=0, column=1)

    def getChoices(self):
        return self.dropdownvar.get()

    def setChoices(self, choice):
        self.dropdownvar.set(choice)

    def enable(self):
        self.popupMenu.configure(state="normal")

    def disable(self):
        self.popupMenu.configure(state="disable")


class FileSelect(Frame):
    def __init__(
        self,
        parent=None,
        fileDescription="",
        color=None,
        title=None,
        lblwidth=None,
        file_types=None,
        dropdown: DropDownMenu = None,
        initialdir: Optional[Union[str, os.PathLike]] = None,
        **kw
    ):

        self.title, self.dropdown, self.initialdir = title, dropdown, initialdir
        self.file_type = file_types
        self.color = color if color is not None else "black"
        self.lblwidth = lblwidth if lblwidth is not None else 0
        self.parent = parent
        Frame.__init__(self, master=parent, **kw)
        browse_icon = ImageTk.PhotoImage(
            image=PIL.Image.open(MENU_ICONS["browse"]["icon_path"])
        )
        self.filePath = StringVar()
        self.lblName = Label(
            self,
            text=fileDescription,
            fg=str(self.color),
            width=str(self.lblwidth),
            anchor=W,
        )
        self.lblName.grid(row=0, column=0, sticky=W)
        self.entPath = Label(self, textvariable=self.filePath, relief=SUNKEN)
        self.entPath.grid(row=0, column=1)
        self.btnFind = Button(
            self,
            text=Defaults.BROWSE_FILE_BTN_TEXT.value,
            compound="left",
            image=browse_icon,
            relief=RAISED,
            command=self.setFilePath,
        )
        self.btnFind.image = browse_icon
        self.btnFind.grid(row=0, column=2)
        self.filePath.set(Defaults.NO_FILE_SELECTED_TEXT.value)

    def setFilePath(self):
        if self.initialdir is not None:
            if not os.path.isdir(self.initialdir):
                self.initialdir = None
            else:
                pass

        if self.file_type:
            file_selected = askopenfilename(
                title=self.title,
                parent=self.parent,
                filetypes=self.file_type,
                initialdir=self.initialdir,
            )
        else:
            file_selected = askopenfilename(
                title=self.title, parent=self.parent, initialdir=self.initialdir
            )
        if file_selected:
            if self.dropdown is not None:
                self.dropdown.setChoices(os.path.basename(file_selected))
                self.filePath.set(os.path.basename(file_selected))
            else:
                self.filePath.set(file_selected)

        else:
            self.filePath.set(Defaults.NO_FILE_SELECTED_TEXT.value)

    @property
    def file_path(self):
        return self.filePath.get()

    def set_state(self, setstatus):
        self.entPath.config(state=setstatus)
        self.btnFind["state"] = setstatus


class Entry_Box(Frame):
    def __init__(
        self,
        parent=None,
        fileDescription="",
        labelwidth="",
        status=None,
        validation=None,
        entry_box_width=None,
        **kw
    ):
        super(Entry_Box, self).__init__(master=parent)
        self.validation_methods = {
            "numeric": (self.register(form_validator_is_numeric), "%P", "%d"),
        }
        self.status = status if status is not None else NORMAL
        self.labelname = fileDescription
        Frame.__init__(self, master=parent, **kw)
        self.filePath = StringVar()
        self.lblName = Label(self, text=fileDescription, width=labelwidth, anchor=W)
        self.lblName.grid(row=0, column=0)
        if not entry_box_width:
            self.entPath = Entry(
                self,
                textvariable=self.filePath,
                state=self.status,
                validate="key",
                validatecommand=self.validation_methods.get(validation, None),
            )
        else:
            self.entPath = Entry(
                self,
                textvariable=self.filePath,
                state=self.status,
                width=entry_box_width,
                validate="key",
                validatecommand=self.validation_methods.get(validation, None),
            )

        self.entPath.grid(row=0, column=1)

    @property
    def entry_get(self):
        self.entPath.get()
        return self.entPath.get()

    def entry_set(self, val):
        self.filePath.set(val)

    def set_state(self, setstatus):
        self.entPath.config(state=setstatus)

    def destroy(self):
        try:
            self.lblName.destroy()
            self.entPath.destroy()
        except:
            pass


class FolderSelect(Frame):
    def __init__(
        self,
        parent: Frame,
        folderDescription: Optional[str] = "",
        color: Optional[str] = None,
        title: Optional[str] = "",
        lblwidth: Optional[int] = 0,
        initialdir: Optional[Union[str, os.PathLike]] = None,
        **kw
    ):

        self.title, self.initialdir = title, initialdir
        self.color = color if color is not None else "black"
        self.lblwidth = lblwidth if lblwidth is not None else 0
        self.parent = parent
        Frame.__init__(self, master=parent, **kw)
        browse_icon = ImageTk.PhotoImage(
            image=PIL.Image.open(MENU_ICONS["browse"]["icon_path"])
        )
        self.folderPath = StringVar()
        self.lblName = Label(
            self,
            text=folderDescription,
            fg=str(self.color),
            width=str(self.lblwidth),
            anchor=W,
        )
        self.lblName.grid(row=0, column=0, sticky=W)
        self.entPath = Label(self, textvariable=self.folderPath, relief=SUNKEN)
        self.entPath.grid(row=0, column=1)
        self.btnFind = Button(
            self,
            text=Defaults.BROWSE_FOLDER_BTN_TEXT.value,
            compound="left",
            image=browse_icon,
            relief=RAISED,
            command=self.setFolderPath,
        )
        self.btnFind.image = browse_icon
        self.btnFind.grid(row=0, column=2)
        self.folderPath.set("No folder selected")

    def setFolderPath(self):
        if self.initialdir is not None:
            if not os.path.isdir(self.initialdir):
                self.initialdir = None
            else:
                pass
        folder_selected = askdirectory(
            title=str(self.title), parent=self.parent, initialdir=self.initialdir
        )
        if folder_selected:
            self.folderPath.set(folder_selected)
        else:
            self.folderPath.set("No folder selected")

    @property
    def folder_path(self):
        return self.folderPath.get()


class ToolTip(object):

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(
            tw,
            text=self.text,
            justify=LEFT,
            background="#ffffe0",
            relief=SOLID,
            borderwidth=1,
            font=("tahoma", "8", "normal"),
        )
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)

    def enter(event):
        toolTip.showtip(text)

    def leave(event):
        toolTip.hidetip()

    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)


def CreateLabelFrameWithIcon(
    parent: Toplevel, header: str, icon_name: str, icon_link: str or None = None
):

    icon = PIL.Image.open(MENU_ICONS[icon_name]["icon_path"])
    icon = ImageTk.PhotoImage(icon)
    frm = Frame(parent)
    label_text = Label(frm, text=header, font=Formats.LABELFRAME_HEADER_FORMAT.value)
    label_text.grid(row=0, column=0)
    label_image = Label(frm, image=icon)
    label_image.image = icon
    if icon_link:
        label_image.bind("<Button-1>", lambda e: callback(icon_link))
    label_image.grid(row=0, column=1)
    return LabelFrame(parent, labelwidget=frm)


def callback(url):
    webbrowser.open_new(url)


def create_scalebar(
    parent: Frame, name: str, min: int, max: int, cmd: object or None = None
):

    scale = Scale(
        parent,
        from_=min,
        to=max,
        orient=HORIZONTAL,
        length=200,
        label=name,
        command=cmd,
    )
    scale.set(0)
    return scale


class TwoOptionQuestionPopUp(object):
    """
    Helpe to create a two-option question tkinter pop up window (e.g., YES/NO).

    :parameter str question: Question to present to the user. E.g., ``DO YOU WANT TO PROCEED?``.
    :parameter str option_one: The first user option. E.g., ``YES``.
    :parameter str option_one: The second user option. E.g., ``NO``.
    :parameter str title: The window titel in the top banner of the pop-up. E.g., ``A QUESTION FOR YOU!``.
    :parameter Optional[str] link: If not None, then a link to documentation presenting background info about the user choices.
    """

    def __init__(
        self,
        question: str,
        option_one: str,
        option_two: str,
        title: str,
        link: Optional[str] = None,
    ):

        self.main_frm = Toplevel()
        self.main_frm.geometry("600x200")
        self.main_frm.title(title)

        question_frm = Frame(self.main_frm)
        question_frm.pack(expand=True, fill="both")
        Label(
            question_frm, text=question, font=Formats.LABELFRAME_HEADER_FORMAT.value
        ).pack()
        button_one = Button(
            question_frm,
            text=option_one,
            fg="blue",
            command=lambda: self.run(option_one),
        )
        button_two = Button(
            question_frm,
            text=option_two,
            fg="red",
            command=lambda: self.run(option_two),
        )
        if link:
            link_lbl = Label(
                question_frm,
                text="Click here for more information.",
                cursor="hand2",
                fg="blue",
            )
            link_lbl.bind("<Button-1>", lambda e: callback(link))
            link_lbl.place(relx=0.5, rely=0.30, anchor=CENTER)
        button_one.place(relx=0.5, rely=0.50, anchor=CENTER)
        button_two.place(relx=0.5, rely=0.70, anchor=CENTER)
        self.main_frm.wait_window()

    def run(self, selected_option):
        self.selected_option = selected_option
        self.main_frm.destroy()

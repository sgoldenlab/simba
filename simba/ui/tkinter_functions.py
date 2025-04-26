__author__ = "Simon Nilsson"

import os.path
import platform
import threading
import tkinter
import tkinter as tk
import webbrowser
from copy import copy
from tkinter import *
from tkinter.filedialog import askdirectory, askopenfilename
from tkinter.ttk import Combobox
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import PIL.Image
from PIL import ImageTk

from simba.utils.enums import Defaults, Formats
from simba.utils.lookups import get_icons_paths
from simba.utils.read_write import get_fn_ext

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


def on_mouse_scroll(event, canvas):
    if event.delta:
        canvas.yview_scroll(-1 * (event.delta // 120), "units")
    elif event.num == 4:
        canvas.yview_scroll(-1, "units")
    elif event.num == 5:
        canvas.yview_scroll(1, "units")

def hxtScrollbar(master):
    canvas = tk.Canvas(master, borderwidth=0, bg="#f0f0f0", relief="flat")
    frame = tk.Frame(canvas, bg="#f0f0f0", bd=0)

    vsb = tk.Scrollbar(master, orient="vertical", command=canvas.yview, bg="#cccccc", width=20)
    hsb = tk.Scrollbar(master, orient="horizontal", command=canvas.xview, bg="#cccccc")

    canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    canvas.pack(side="left", fill="both", expand=True, padx=0, pady=0)
    vsb.pack(side="right", fill="y", padx=(0, 0))
    hsb.pack(side="bottom", fill="x", pady=(0, 0))

    canvas.create_window((0, 0), window=frame, anchor="nw")

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame.bind("<Configure>", on_frame_configure)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_key(event):
        if event.keysym == "Up":
            canvas.yview_scroll(-1, "units")
        elif event.keysym == "Down":
            canvas.yview_scroll(1, "units")
        elif event.keysym == "Page_Up":
            canvas.yview_scroll(-1, "pages")
        elif event.keysym == "Page_Down":
            canvas.yview_scroll(1, "pages")

    # Focus & bindings when mouse enters
    def _on_enter(_):
        frame.focus_set()
        frame.bind_all("<MouseWheel>", _on_mousewheel)
        frame.bind_all("<Key>", _on_key)

    # Clean up when mouse leaves
    # def _on_leave(_):
    #     frame.unbind_all("<MouseWheel>")
    #     frame.unbind_all("<Key>")

    frame.bind("<Enter>", _on_enter)
    #frame.bind("<Leave>", _on_leave)

    return frame

def form_validator_is_numeric(inStr, acttyp):
    if acttyp == "1":  # insert
        if not inStr.isdigit():
            return False
    return True


class DropDownMenu(Frame):
    def __init__(self,
                 parent=None,
                 dropdownLabel="",
                 choice_dict=None,
                 labelwidth="",
                 com=None,
                 val: Optional[Any] = None,
                 **kw):

        Frame.__init__(self, master=parent, **kw)
        self.dropdownvar = StringVar()
        self.lblName = Label(self, text=dropdownLabel, width=labelwidth, anchor=W, font=Formats.FONT_REGULAR.value)
        self.lblName.grid(row=0, column=0)
        self.choices = choice_dict
        self.popupMenu = OptionMenu(self, self.dropdownvar, *self.choices, command=com)
        self.popupMenu.grid(row=0, column=1)
        if val is not None:
            self.setChoices(val)

    def getChoices(self):
        return self.dropdownvar.get()

    def setChoices(self, choice):
        self.dropdownvar.set(choice)

    def enable(self):
        self.popupMenu.configure(state="normal")

    def disable(self):
        self.popupMenu.configure(state="disable")


class Entry_Box(Frame):
    def __init__(self,
                 parent=None,
                 fileDescription="",
                 labelwidth="",
                 label_bg_clr: Optional[str] = None,
                 status=None,
                 validation=None,
                 entry_box_width=None,
                 value: Optional[Any] = None,
                 label_font: tuple = Formats.FONT_REGULAR.value,
                 entry_font: tuple = Formats.FONT_REGULAR.value,
                 justify: str = 'left',
                 **kw):

        super(Entry_Box, self).__init__(master=parent)
        self.validation_methods = {"numeric": (self.register(form_validator_is_numeric), "%P", "%d")}
        self.status = status if status is not None else NORMAL
        self.labelname = fileDescription
        Frame.__init__(self, master=parent, **kw)
        self.filePath = StringVar()
        self.lblName = Label(self, text=fileDescription, width=labelwidth, anchor=W, font=label_font, bg=label_bg_clr)
        self.lblName.grid(row=0, column=0)
        if not entry_box_width:
            self.entPath = Entry(self, textvariable=self.filePath, state=self.status,  validate="key", validatecommand=self.validation_methods.get(validation, None), font=entry_font, justify=justify)
        else:
            self.entPath = Entry(self, textvariable=self.filePath, state=self.status, width=entry_box_width, validate="key", font=entry_font, justify=justify, validatecommand=self.validation_methods.get(validation, None))
        self.entPath.grid(row=0, column=1)
        if value is not None:
            self.entry_set(val=value)

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
    def __init__(self,
                 parent: Union[Frame, LabelFrame, Canvas, Toplevel],
                 folderDescription: Optional[str] = "",
                 color: Optional[str] = None,
                 label_bg_clr: Optional[str] = None,
                 font: Tuple = Formats.FONT_REGULAR.value,
                 title: Optional[str] = "",
                 lblwidth: Optional[int] = 0,
                 bg_clr: Optional[str] = 'white',
                 entry_width: Optional[int] = 20,
                 initialdir: Optional[Union[str, os.PathLike]] = None,
                 **kw):

        self.title, self.initialdir = title, initialdir
        self.color = color if color is not None else "black"
        self.lblwidth = lblwidth if lblwidth is not None else 0
        self.parent = parent
        Frame.__init__(self, master=parent, **kw)
        browse_icon = ImageTk.PhotoImage(image=PIL.Image.open(MENU_ICONS["browse_small"]["icon_path"]))
        self.folderPath = StringVar()
        self.lblName = Label(self, text=folderDescription, fg=str(self.color), width=str(self.lblwidth), anchor=W, font=font, bg=label_bg_clr)
        self.lblName.grid(row=0, column=0, sticky=NW)
        self.entPath = Label(self, textvariable=self.folderPath, relief=SUNKEN, font=font, bg=bg_clr, width=entry_width)
        self.entPath.grid(row=0, column=1, sticky=NW)
        self.btnFind = Button(self, text=Defaults.BROWSE_FOLDER_BTN_TEXT.value, compound="left", image=browse_icon, relief=RAISED, font=font, command=self.setFolderPath)
        self.btnFind.image = browse_icon
        self.btnFind.grid(row=0, column=2, sticky=NW)
        self.folderPath.set("No folder selected")

    def setFolderPath(self):
        if self.initialdir is not None:
            if not os.path.isdir(self.initialdir):
                self.initialdir = None
            else:
                pass
        folder_selected = askdirectory(title=str(self.title), parent=self.parent, initialdir=self.initialdir)
        if folder_selected:
            self.folderPath.set(folder_selected)
            self.entPath.configure(width=len(folder_selected)+10)
        else:
            self.folderPath.set("No folder selected")
            self.entPath.configure(width=len("No folder selected")+10)


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
        x = x + self.widget.winfo_rootx() + 20
        y = y + cy + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(tw, text=self.text, justify=LEFT, background="#ffffe0", relief=SOLID, borderwidth=1, font=Formats.FONT_REGULAR.value)
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


def CreateLabelFrameWithIcon(parent: Union[Toplevel, LabelFrame, Canvas, Frame],
                             header: str,
                             icon_name: str,
                             padx: Optional[int] = None,
                             pady: Optional[int] = None,
                             relief: str = 'solid',
                             width: Optional[int] = None,
                             bg: Optional[str] = None,
                             font: tuple = Formats.FONT_HEADER.value,
                             icon_link: Optional[Union[str, None]] = LabelFrame):

    icon = PIL.Image.open(MENU_ICONS[icon_name]["icon_path"])
    icon = ImageTk.PhotoImage(icon)

    frm = Frame(parent, bg=bg)  # <- Apply bg here
    label_text = Label(frm, text=header, font=font, bg=bg)  # <- And here
    label_text.grid(row=0, column=0)

    label_image = Label(frm, image=icon, bg=bg)  # <- And here
    label_image.image = icon
    if icon_link:
        label_image.bind("<Button-1>", lambda e: callback(icon_link))
    label_image.grid(row=0, column=1)

    lbl_frm = LabelFrame(parent, labelwidget=frm, relief=relief, width=width, padx=padx or 0, pady=pady or 0)
    if bg is not None:
        lbl_frm.configure(bg=bg)
    return lbl_frm

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

    def __init__(self,
                 question: str,
                 option_one: str,
                 option_two: str,
                 title: str,
                 link: Optional[str] = None):

        self.main_frm = Toplevel()
        self.main_frm.geometry("600x200")
        self.main_frm.title(title)

        question_frm = Frame(self.main_frm)
        question_frm.pack(expand=True, fill="both")
        Label(question_frm, text=question, font=Formats.LABELFRAME_HEADER_FORMAT.value).pack()

        button_one = SimbaButton(parent=question_frm, txt=option_one, txt_clr="blue", bg_clr="lightgrey", img='check_blue', cmd=self.run, cmd_kwargs={'selected_option': lambda: option_one}, font=Formats.FONT_LARGE.value)
        button_two = SimbaButton(parent=question_frm, txt=option_two, txt_clr="red", bg_clr="lightgrey", img='close', cmd=self.run, cmd_kwargs={'selected_option': lambda: option_two}, font=Formats.FONT_LARGE.value)
        #button_one = Button(question_frm, text=option_one, fg="blue", command=lambda: self.run(option_one))
        #button_two = Button(question_frm, text=option_two, fg="red", command=lambda: self.run(option_two))
        if link:
            link_lbl = Label(question_frm, text="Click here for more information.", cursor="hand2", fg="blue")
            link_lbl.bind("<Button-1>", lambda e: callback(link))
            link_lbl.place(relx=0.5, rely=0.30, anchor=CENTER)
        button_one.place(relx=0.5, rely=0.50, anchor=CENTER)
        button_two.place(relx=0.5, rely=0.80, anchor=CENTER)
        self.main_frm.wait_window()

    def run(self, selected_option):
        self.selected_option = selected_option
        self.main_frm.destroy()

def SimbaButton(parent: Union[Frame, Canvas, LabelFrame, Toplevel],
                txt: str,
                txt_clr: Optional[str] = 'black',
                bg_clr: Optional[str] = '#f0f0f0',
                active_bg_clr: Optional[str] = None,
                hover_bg_clr: Optional[str] = Formats.BTN_HOVER_CLR.value,
                hover_font: Optional[Tuple] = Formats.FONT_REGULAR_BOLD.value,
                font: Optional[Tuple] = Formats.FONT_REGULAR.value,
                width: Optional[int] = None,
                height: Optional[int] = None,
                compound: Optional[str] = 'left',
                img: Optional[Union[ImageTk.PhotoImage, str]] = None,
                cmd: Optional[Callable] = None,
                cmd_kwargs: Optional[Dict[Any, Any]] = None,
                enabled: Optional[bool] = True,
                anchor: str = 'w',
                thread: Optional[bool] = False,
                tooltip_txt: Optional[str] = None) -> Button:
    def on_enter(e):
        e.widget.config(bg=hover_bg_clr, font=hover_font)

    def on_leave(e):
        e.widget.config(bg=bg_clr, font=font)

    if isinstance(img, str):
        img = ImageTk.PhotoImage(image=PIL.Image.open(MENU_ICONS[img]["icon_path"]))

    if cmd_kwargs is None:
        cmd_kwargs = {}

    def execute_command():
        if cmd:
            evaluated_kwargs = {k: (v() if callable(v) else v) for k, v in cmd_kwargs.items()}
            cmd(**evaluated_kwargs)

    if cmd is not None:
        if thread:
            command = lambda: threading.Thread(target=execute_command).start()
        else:
            command = execute_command
    else:
        command = None

    btn = Button(master=parent,
                 text=txt,
                 compound=compound,
                 image=img,
                 relief=RAISED,
                 fg=txt_clr,
                 activebackground=active_bg_clr,
                 font=font,
                 bg=bg_clr,
                 anchor=anchor,
                 command=command)

    if img is not None:
        btn.image = img
    if width is not None:
        btn.config(width=width)
    if height is not None:
        btn.config(height=height)
    if not enabled:
        btn.config(state=DISABLED)

    if hover_bg_clr is not None:
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    if tooltip_txt is not None:
        CreateToolTip(widget=btn, text=tooltip_txt)

    return btn


def SimbaCheckbox(parent: Union[Frame, Toplevel, LabelFrame, Canvas],
                  txt: str,
                  txt_clr: Optional[str] = 'black',
                  txt_img: Optional[str] = None,

                  txt_img_location: Literal['left', 'right', 'top', 'bottom'] = RIGHT,
                  font: Optional[Tuple[str, str, int]] = Formats.FONT_REGULAR.value,
                  val: Optional[bool] = False,
                  state: Literal["disabled", 'normal'] = NORMAL,
                  cmd: Optional[Callable] = None):

    var = BooleanVar(value=False)
    if val: var.set(True)
    if isinstance(txt_img, str):
        txt_img = ImageTk.PhotoImage(image=PIL.Image.open(MENU_ICONS[txt_img]["icon_path"]))
    if cmd is None:
        cb = Checkbutton(master=parent, font=font, fg=txt_clr, image=txt_img, text=txt, compound=txt_img_location, variable=var)
    else:
        cb = Checkbutton(master=parent, font=font, fg=txt_clr, image=txt_img, text=txt, compound=txt_img_location, variable=var, command=cmd)
    if txt_img is not None:
        cb.image = txt_img
    if state == DISABLED:
        cb.configure(state=DISABLED)
    return cb, var

def SimBALabel(parent: Union[Frame, Canvas, LabelFrame, Toplevel],
               txt: str,
               txt_clr: str = 'black',
               bg_clr: Optional[str] = None,
               font: tuple = Formats.FONT_REGULAR.value,
               relief: str = FLAT,
               justify: Optional[str] = None,
               link: Optional[str] = None,
               width: Optional[int] = None,
               cursor: Optional[str] = None):

    lbl = Label(parent,
                text=txt,
                font=font,
                fg=txt_clr,
                bg=bg_clr,
                justify=justify,
                relief=relief,
                cursor=cursor)

    if width is not None:
        lbl.configure(width=width)

    if link is not None:
        lbl.bind("<Button-1>", lambda e: callback(link))

    return lbl


def get_menu_icons():
    menu_icons = copy(MENU_ICONS)
    for k in menu_icons.keys():
        menu_icons[k]["img"] = ImageTk.PhotoImage(image=PIL.Image.open(os.path.join(os.path.dirname(__file__), menu_icons[k]["icon_path"])))
    return menu_icons


class SimBADropDown(Frame):
    def __init__(self,
                parent: Union[Frame, Canvas, LabelFrame, Toplevel, Tk],
                dropdown_options: Union[Iterable[Any], List[Any], Tuple[Any]],
                label: Optional[str] = None,
                label_width: Optional[int] = None,
                label_font: tuple = Formats.FONT_REGULAR.value,
                label_bg_clr: Optional[str] = None,
                dropdown_font_size: Optional[int] = None,
                justify: str = 'center',
                dropdown_width: Optional[int] = None,
                command: Callable = None,
                value: Optional[Any] = None,
                state: Optional[str] = None):

        super().__init__(master=parent)
        self.dropdown_var = StringVar()
        self.dropdown_lbl = Label(self, text=label, width=label_width, anchor="w", font=label_font, bg=label_bg_clr)
        self.dropdown_lbl.grid(row=0, column=0)
        self.dropdown_options = dropdown_options
        self.command = command
        if dropdown_font_size is None:
            drop_down_font = None
        else:
            drop_down_font = ("Poppins", dropdown_font_size)
        self.dropdown = Combobox(self, textvariable=self.dropdown_var, font=drop_down_font, values=self.dropdown_options, state="readonly", width=dropdown_width, justify=justify)
        self.dropdown.grid(row=0, column=1, sticky="nw")
        if value is not None: self.set_value(value=value)
        if command is not None:
            self.command = command
            self.dropdown.bind("<<ComboboxSelected>>", self.on_select)
        if state == 'disabled':
            self.disable()

    def set_value(self, value: Any):
        self.dropdown_var.set(value)

    def get_value(self):
        return self.dropdown_var.get()

    def enable(self):
        self.dropdown.configure(state="normal")

    def disable(self):
        self.dropdown.configure(state="disabled")

    def getChoices(self):
        return self.dropdown_var.get()

    def setChoices(self, choice):
        self.dropdown_var.set(choice)

    def on_select(self, event):
        selected_value = self.dropdown_var.get()
        self.command(selected_value)


class DropDownMenu(Frame):
    def __init__(self,
                 parent=None,
                 dropdownLabel="",
                 choice_dict=None,
                 labelwidth="",
                 com=None,
                 val: Optional[Any] = None,
                 **kw):
        Frame.__init__(self, master=parent, **kw)
        self.dropdownvar = StringVar()
        self.lblName = Label(self, text=dropdownLabel, width=labelwidth, anchor=W, font=Formats.FONT_REGULAR.value)
        self.lblName.grid(row=0, column=0)
        self.choices = choice_dict
        self.popupMenu = OptionMenu(self, self.dropdownvar, *self.choices, command=com)
        self.popupMenu.grid(row=0, column=1)
        if val is not None:
            self.setChoices(val)
    def getChoices(self):
        return self.dropdownvar.get()
    def setChoices(self, choice):
        self.dropdownvar.set(choice)
    def enable(self):
        self.popupMenu.configure(state="normal")
    def disable(self):
        self.popupMenu.configure(state="disable")



class FileSelect(Frame):
    def __init__(self,
                 parent=None,
                 fileDescription="",
                 color=None,
                 title=None,
                 lblwidth=None,
                 file_types=None,
                 bg_clr: Optional[str] = 'white',
                 dropdown: Union[DropDownMenu, SimBADropDown] = None,
                 entry_width: Optional[int] = 20,
                 status: Optional[str] = None,
                 initialdir: Optional[Union[str, os.PathLike]] = None,
                 initial_path: Optional[Union[str, os.PathLike]] = None, **kw):

        self.title, self.dropdown, self.initialdir = title, dropdown, initialdir
        self.file_type = file_types
        self.color = color if color is not None else "black"
        self.lblwidth = lblwidth if lblwidth is not None else 0
        self.parent = parent
        Frame.__init__(self, master=parent, **kw)
        browse_icon = ImageTk.PhotoImage(image=PIL.Image.open(MENU_ICONS["browse"]["icon_path"]))
        self.filePath = StringVar()
        self.lblName = Label(self, text=fileDescription, fg=str(self.color), width=str(self.lblwidth), anchor=W, font=Formats.FONT_REGULAR.value)
        self.lblName.grid(row=0, column=0, sticky=W)
        self.entPath = Label(self, textvariable=self.filePath, relief=SUNKEN, font=Formats.FONT_REGULAR.value, bg=bg_clr, width=entry_width)
        self.entPath.grid(row=0, column=1)
        self.btnFind = Button(self, text=Defaults.BROWSE_FILE_BTN_TEXT.value, compound="left", image=browse_icon, relief=RAISED, command=self.setFilePath, font=Formats.FONT_REGULAR.value)
        self.btnFind.image = browse_icon
        self.btnFind.grid(row=0, column=2)
        self.filePath.set(Defaults.NO_FILE_SELECTED_TEXT.value)
        if initial_path is not None:
            self.filePath.set(initial_path)
        if status is not None:
            self.set_state(setstatus=status)


    def setFilePath(self):
        if self.initialdir is not None:
            if not os.path.isdir(self.initialdir):
                self.initialdir = None
            else:
                pass

        if self.file_type:
            file_selected = askopenfilename(title=self.title, parent=self.parent, filetypes=self.file_type, initialdir=self.initialdir)
        else:
            file_selected = askopenfilename(title=self.title, parent=self.parent, initialdir=self.initialdir)
        if file_selected:
            if self.dropdown is not None:
                _, name, _ = get_fn_ext(filepath=file_selected)
                self.dropdown.setChoices(name)
                self.filePath.set(name)
            else:
                self.filePath.set(file_selected)
            self.entPath.configure(width=len(file_selected)+10)

        else:
            self.filePath.set(Defaults.NO_FILE_SELECTED_TEXT.value)
            self.entPath.configure(width=len(Defaults.NO_FILE_SELECTED_TEXT.value) + 10)

    @property
    def file_path(self):
        return self.filePath.get()

    def set_state(self, setstatus):
        self.entPath.config(state=setstatus)
        self.btnFind["state"] = setstatus





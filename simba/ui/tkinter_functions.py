__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os.path
import platform
import threading
import tkinter
import tkinter as tk
import webbrowser
from copy import copy, deepcopy
from tkinter import *
from tkinter.filedialog import askdirectory, askopenfilename
from tkinter.ttk import Combobox
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np
import PIL.Image
from PIL import ImageTk

from simba.utils.checks import check_if_valid_img
from simba.utils.enums import Defaults, Formats, TkBinds
from simba.utils.lookups import get_icons_paths, get_tooltips
from simba.utils.read_write import get_fn_ext

MENU_ICONS = get_icons_paths()
TOOLTIPS = get_tooltips()

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
    """
    Legacy, use :func:`simba.ui.tkinter_functions.SimBADropDown`.
    """
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


class SimBAScaleBar(Frame):
    def __init__(self,
                 parent: Union[Frame, Canvas, LabelFrame, Toplevel, Tk],
                 label: Optional[str] = None,
                 label_width: Optional[int] = None,
                 orient: Literal['horizontal', 'vertical'] = HORIZONTAL,
                 length: int = 200,
                 value: Optional[int] = 95,
                 showvalue: bool = True,
                 label_clr: str = 'black',
                 lbl_font: tuple = Formats.FONT_REGULAR.value,
                 scale_font: tuple = Formats.FONT_REGULAR_ITALICS.value,
                 lbl_img: Optional[str] = None,
                 from_: int = -1,
                 resolution: int = 1,
                 to: int = 100,
                 tickinterval: Optional[int] = None,
                 troughcolor: Optional[str] = None,
                 activebackground: Optional[str] = None,
                 sliderrelief: Literal["raised", "sunken", "flat", "ridge", "solid", "groove"] = 'flat'):

        super().__init__(master=parent)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)
        if lbl_img is not None:
            self.columnconfigure(2, weight=0)
            self.lbl_lbl = SimBALabel(parent=self, txt='', txt_clr='black', bg_clr=None, font=lbl_font, width=None, anchor='w', img=lbl_img, compound=None)
            self.lbl_lbl.grid(row=0, column=0, sticky=SW)
        else:
            self.lbl_lbl = None

        self.scale = Scale(self,
                           from_=from_,
                           to=to,
                           orient=orient,
                           length=length,
                           font=scale_font,
                           sliderrelief=sliderrelief,
                           troughcolor=troughcolor,
                           tickinterval=tickinterval,
                           resolution=resolution,
                           showvalue=showvalue,
                           activebackground=activebackground)

        if label is not None:
            self.lbl = SimBALabel(parent=self, txt=label, font=lbl_font, txt_clr=label_clr, width=label_width)
            self.lbl.grid(row=0, column=1, sticky=SW)

        self.scale.grid(row=0, column=2, sticky=NW)
        if value is not None:
            self.set_value(value=value)

    def set_value(self, value: int):
        self.scale.set(value)

    def get_value(self) -> Union[int, float]:
        return self.scale.get()

    def get(self) -> Union[int, float]:
        ## Alternative for ``get_value`` for legacy reasons.
        return self.scale.get()




class Entry_Box(Frame):
    def __init__(self,
                 parent: Union[Frame, Canvas, LabelFrame, Toplevel, Tk],
                 fileDescription: Optional[str] = "",
                 labelwidth: Union[int, str] = None,
                 label_bg_clr: Optional[str] = None,
                 status: Literal["normal", "disabled", "readonly"] = 'normal',
                 validation: Optional[Union[Callable, str]] = None,
                 trace: Optional[Callable] = None,
                 entry_box_width: Union[int, str] = None,
                 entry_box_clr: Optional[str] = 'white',
                 img: Optional[str] = None,
                 value: Optional[Any] = None,
                 label_font: tuple = Formats.FONT_REGULAR.value,
                 entry_font: tuple = Formats.FONT_REGULAR.value,
                 tooltip_key: Optional[str] = None,
                 justify: Literal["left", "center", "right"] = 'left',
                 cmd: Optional[Callable] = None,
                 allow_blank: bool = False,
                 **kw):

        super(Entry_Box, self).__init__(master=parent)
        self.validation_methods = {"numeric": (self.register(form_validator_is_numeric), "%P", "%d")}
        self.status = status if status is not None else NORMAL
        self.labelname = fileDescription
        Frame.__init__(self, master=parent, **kw)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)
        if img is not None:
            self.columnconfigure(2, weight=0)
            self.entrybox_img_lbl = SimBALabel(parent=self, txt='', txt_clr='black', bg_clr=label_bg_clr, font=label_font, width=None, anchor='w', img=img, compound=None)
            self.entrybox_img_lbl.grid(row=0, column=0, sticky="w")
        else:
            self.entrybox_img_lbl = None
        self.filePath = StringVar()
        self.lblName = Label(self, text=fileDescription, width=labelwidth, anchor=W, font=label_font, bg=label_bg_clr)
        self.lblName.grid(row=0, column=1)
        if tooltip_key in TOOLTIPS.keys():
            CreateToolTip(widget=self.lblName, text=TOOLTIPS[tooltip_key])
        if not entry_box_width:
            self.entPath = Entry(self, textvariable=self.filePath, state=self.status,  validate="key", validatecommand=self.validation_methods.get(validation, None), font=entry_font, justify=justify, bg=entry_box_clr)
        else:
            self.entPath = Entry(self, textvariable=self.filePath, state=self.status, width=entry_box_width, validate="key", font=entry_font, justify=justify, validatecommand=self.validation_methods.get(validation, None), bg=entry_box_clr)
        self.entPath.grid(row=0, column=2)
        if value is not None:
            self.entry_set(val=value)
        if cmd is not None:
            self.bind_combobox_keys()
            self.cmd = cmd
        self.allow_blank = allow_blank
        if trace is not None:
            self.filePath.trace_add('write', lambda *a: trace(self))


    def bind_combobox_keys(self):
        self.entPath.bind("<KeyRelease>", self.run_cmd)

    @property
    def entry_get(self):
        return self.entPath.get()

    def entry_set(self, val):
        self.filePath.set(val)

    def set_state(self, setstatus):
        self.entPath.config(state=setstatus)

    def destroy(self):
        try:
            if self.entrybox_img_lbl is not None:
                self.entrybox_img_lbl.destroy()
            self.lblName.destroy()
            self.entPath.destroy()
        except:
            pass

    def run_cmd(self, x):
        self.cmd(self.entry_get.strip())


    def set_bg_clr(self, clr: str):
        self.entPath.configure(bg=clr)



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
                 lbl_icon: Optional[str] = None,
                 initialdir: Optional[Union[str, os.PathLike]] = None,
                 tooltip_txt: Optional[str] = None,
                 tooltip_key: Optional[str] = None,
                 **kw):

        self.title, self.initialdir = title, initialdir
        self.color = color if color is not None else "black"
        self.lblwidth = lblwidth if lblwidth is not None else 0
        self.parent = parent
        Frame.__init__(self, master=parent, **kw)
        browse_icon = ImageTk.PhotoImage(image=PIL.Image.open(MENU_ICONS["browse_small"]["icon_path"]))
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=0)
        if lbl_icon is not None:
            self.columnconfigure(3, weight=0)
            self.lbl_icon = SimBALabel(parent=self, txt='', txt_clr='black', bg_clr=label_bg_clr, font=font, width=None, anchor='w', img=lbl_icon, compound=None)
            self.lbl_icon.grid(row=0, column=0, sticky="w")
        else:
            self.lbl_icon = None
        self.folderPath = StringVar()
        self.lblName = Label(self, text=folderDescription, fg=str(self.color), width=str(self.lblwidth), anchor=W, font=font, bg=label_bg_clr)
        self.lblName.grid(row=0, column=1, sticky=NW)
        self.entPath = Label(self, textvariable=self.folderPath, relief=SUNKEN, font=font, bg=bg_clr, width=entry_width)
        self.entPath.grid(row=0, column=2, sticky=NW)
        self.btnFind = SimbaButton(parent=self, txt=Defaults.BROWSE_FOLDER_BTN_TEXT.value, font=font, cmd=self.setFolderPath, img=browse_icon)
        self.btnFind.image = browse_icon
        self.btnFind.grid(row=0, column=3, sticky=NW)
        self.folderPath.set("No folder selected")
        if tooltip_txt is not None and isinstance(tooltip_txt, str):
            CreateToolTip(widget=self.lblName, text=tooltip_txt)
        elif tooltip_key in TOOLTIPS.keys():
            CreateToolTip(widget=self.lblName, text=TOOLTIPS[tooltip_key])

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
                             icon_link: Optional[Union[str, None]] = LabelFrame,
                             tooltip_key: Optional[str] = None):

    if icon_name in MENU_ICONS.keys():
        icon = PIL.Image.open(MENU_ICONS[icon_name]["icon_path"])
        icon = ImageTk.PhotoImage(icon)
    else:
        icon = None
    frm = Frame(parent, bg=bg)

    label_image = Label(frm, image=icon, bg=bg)
    label_image.image = icon
    if icon_link:
        label_image.bind("<Button-1>", lambda e: callback(icon_link))
    label_image.grid(row=0, column=0)


    label_text = Label(frm, text=header, font=font, bg=bg)
    label_text.grid(row=0, column=1)



    lbl_frm = LabelFrame(parent, labelwidget=frm, relief=relief, width=width, padx=padx or 0, pady=pady or 0)

    if tooltip_key in TOOLTIPS.keys():
        CreateToolTip(widget=label_text, text=TOOLTIPS[tooltip_key])

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
        menu_icons = get_menu_icons()
        self.main_frm.iconphoto(False, menu_icons['question_mark']["img"])

        #img = ImageTk.PhotoImage(image=PIL.Image.open(MENU_ICONS['question_mark']["icon_path"]))

        question_frm = Frame(self.main_frm)
        question_frm.pack(expand=True, fill="both")
        Label(question_frm, text=question, font=Formats.LABELFRAME_HEADER_FORMAT.value).pack()

        button_one = SimbaButton(parent=question_frm, txt=option_one, txt_clr="blue", bg_clr="lightgrey", img='check_blue', cmd=self.run, cmd_kwargs={'selected_option': lambda: option_one}, font=Formats.FONT_LARGE.value, hover_font=Formats.FONT_LARGE_BOLD.value)
        button_two = SimbaButton(parent=question_frm, txt=option_two, txt_clr="red", bg_clr="lightgrey", img='close', cmd=self.run, cmd_kwargs={'selected_option': lambda: option_two}, font=Formats.FONT_LARGE.value, hover_font=Formats.FONT_LARGE_BOLD.value)
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
                tooltip_txt: Optional[str] = None,
                tooltip_key: Optional[str] = None) -> Button:
    def on_enter(e):
        e.widget.config(bg=hover_bg_clr, font=hover_font)

    def on_leave(e):
        e.widget.config(bg=bg_clr, font=font)

    # if hover_font != font:
    #     hover_font = copy(font)



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
    elif tooltip_key is not None and tooltip_key in TOOLTIPS.keys():
        CreateToolTip(widget=btn, text=TOOLTIPS[tooltip_key])

    return btn


def SimbaCheckbox(parent: Union[Frame, Toplevel, LabelFrame, Canvas],
                  txt: str,
                  txt_clr: Optional[str] = 'black',
                  bg_clr: Optional[str] = None,
                  txt_img: Optional[str] = None,
                  txt_img_location: Literal['left', 'right', 'top', 'bottom'] = RIGHT,
                  font: Optional[Tuple[str, str, int]] = Formats.FONT_REGULAR.value,
                  val: Optional[bool] = False,
                  state: Literal["disabled", 'normal'] = NORMAL,
                  indicatoron: bool = True,
                  cmd: Optional[Callable] = None,
                  tooltip_txt: Optional[str] = None,
                  tooltip_key: Optional[str] = None):

    var = BooleanVar(value=False)
    if val: var.set(True)
    if isinstance(txt_img, str):
        txt_img = ImageTk.PhotoImage(image=PIL.Image.open(MENU_ICONS[txt_img]["icon_path"]))
    if cmd is None:
        cb = Checkbutton(master=parent, font=font, fg=txt_clr, image=txt_img, text=txt, compound=txt_img_location, variable=var, indicatoron=indicatoron, bg=bg_clr)
    else:
        cb = Checkbutton(master=parent, font=font, fg=txt_clr, image=txt_img, text=txt, compound=txt_img_location, variable=var, command=cmd, indicatoron=indicatoron, bg=bg_clr)
    if txt_img is not None:
        cb.image = txt_img
    if state == DISABLED:
        cb.configure(state=DISABLED)
    if isinstance(tooltip_txt, str):
        CreateToolTip(widget=cb, text=tooltip_txt)
    elif isinstance(tooltip_key, str) and tooltip_key in TOOLTIPS.keys():
        CreateToolTip(widget=cb, text=TOOLTIPS[tooltip_key])

    return cb, var

def SimBALabel(parent: Union[Frame, Canvas, LabelFrame, Toplevel],
               txt: str,
               txt_clr: str = 'black',
               bg_clr: Optional[str] = None,
               hover_fg_clr: Optional[str] = None,
               font: tuple = Formats.FONT_REGULAR.value,
               hover_font: Optional[Tuple] = None,
               relief: str = FLAT,
               compound: Optional[Literal['left', 'right', 'top', 'bottom', 'center']] = 'left',
               justify: Optional[str] = None,
               link: Optional[str] = None,
               width: Optional[int] = None,
               padx: Optional[int] = None,
               pady: Optional[int] = None,
               cursor: Optional[str] = None,
               img: Optional[str] = None,
               anchor: Optional[str] = None,
               tooltip_key: Optional[str] = None,
               hover_img: Optional[np.ndarray] = None):


    def _hover_enter(e):
        w = e.widget
        if hover_img is not None and check_if_valid_img(data=hover_img, raise_error=False):
            arr = np.asarray(hover_img, dtype=np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = arr[:, :, ::-1]
            pil_img = PIL.Image.fromarray(arr)
            photo = ImageTk.PhotoImage(pil_img)
            tw = Toplevel(w)
            tw.wm_overrideredirect(True)
            tw.attributes("-topmost", True)
            tw.wm_geometry("+%d+%d" % (w.winfo_rootx() + w.winfo_width() + 4, w.winfo_rooty()))
            frm = Frame(tw, relief="solid", bd=2, bg="#f0f0f0")
            frm.pack(padx=2, pady=2)
            lbl_hover = Label(frm, image=photo, bg="#f0f0f0")
            lbl_hover.image = photo
            lbl_hover.pack(padx=4, pady=(4, 2))
            caption_lbl = Label(frm, text=txt, font=font, fg=txt_clr, bg=bg_clr)
            caption_lbl.pack(pady=(0, 4))
            tw.lift()
            w._hover_toplevel = tw
        elif hover_fg_clr is not None or hover_font is not None:
            w.config(fg=hover_fg_clr, font=hover_font)

    def _hover_leave(e):
        w = e.widget
        if getattr(w, "_hover_toplevel", None) is not None:
            try:
                w._hover_toplevel.destroy()
            except tkinter.TclError:
                pass
            w._hover_toplevel = None
        if hover_fg_clr is not None or hover_font is not None:
            w.config(fg=txt_clr, bg=bg_clr, font=font)

    def on_enter(e):
        if hover_img is not None:
            _hover_enter(e)
        else:
            e.widget.config(fg=hover_fg_clr, font=hover_font)

    def on_leave(e):
        if hover_img is not None:
            _hover_leave(e)
        else:
            e.widget.config(fg=txt_clr, bg=bg_clr, font=font)

    anchor = 'w' if anchor is None else anchor
    if isinstance(img, str) and img in MENU_ICONS.keys():
        img = ImageTk.PhotoImage(image=PIL.Image.open(MENU_ICONS[img]["icon_path"]))
    else:
        img = None
    if img is not None and compound is None:
        compound = 'left'

    lbl = Label(parent,
                text=txt,
                font=font,
                fg=txt_clr,
                bg=bg_clr,
                justify=justify,
                relief=relief,
                cursor=cursor,
                anchor=anchor,
                image=img,
                compound=compound,
                padx=padx,
                pady=pady)

    if img is not None:
        lbl.image = img

    if width is not None:
        lbl.configure(width=width)

    if link is not None:
        lbl.bind("<Button-1>", lambda e: callback(link))

    elif tooltip_key in TOOLTIPS.keys():
        CreateToolTip(widget=lbl, text=TOOLTIPS[tooltip_key])

    if hover_font is not None or hover_fg_clr is not None or hover_img is not None:
        lbl.bind(TkBinds.ENTER.value, on_enter)
        lbl.bind(TkBinds.LEAVE.value, on_leave)

    return lbl


def get_menu_icons():
    menu_icons = copy(MENU_ICONS)
    for k in menu_icons.keys():
        menu_icons[k]["img"] = ImageTk.PhotoImage(image=PIL.Image.open(os.path.join(os.path.dirname(__file__), menu_icons[k]["icon_path"])))
    return menu_icons



class SimBASeperator(Frame):

    def __init__(self,
                 parent: Union[Frame, Canvas, LabelFrame, Toplevel, Tk],
                 color: Optional[str] = 'black',
                 orient: Literal['horizontal', 'vertical'] = 'horizontal',
                 cursor: Optional[str] = None,
                 borderwidth: Optional[int] = None,
                 takefocus: Optional[int] = 0,
                 height: int = 1,
                 relief: Literal["raised", "sunken", "flat", "ridge", "solid", "groove"] = "flat"):

        super().__init__(master=parent, height=height, bg=color if color is not None else "black", relief=relief, bd=borderwidth if borderwidth is not None else None)
        style = tk.ttk.Style()
        style_name = "SimBA.TSeparator"

        style.configure(style_name,
                        background=color if color is not None else "black")

        seperator = tk.ttk.Separator(self,
                                     orient=orient,
                                     style=style_name,
                                     cursor=cursor,
                                     takefocus=takefocus)

        if orient == "horizontal":
            seperator.pack(fill="x", expand=True)
        else:
            seperator.pack(fill="y", expand=True)


class SimBADropDown(Frame):
    """
    Create a dropdown menu widget with optional searchable functionality.

    This class creates a ttk.Combobox dropdown menu with a label, supporting both readonly and searchable modes.
    When searchable mode is enabled, users can type to filter the dropdown options.

    :param parent (Frame | Canvas | LabelFrame | Toplevel | Tk): The parent widget container.
    :param dropdown_options (Iterable[Any] | List[Any] | Tuple[Any]): List of options to display in the dropdown menu.
    :param label (str, optional): Text label displayed next to the dropdown. Default: None.
    :param label_width (int, optional): Width of the label in characters. Default: None.
    :param label_font (tuple, optional): Font tuple for the label. Default: Formats.FONT_REGULAR.value.
    :param label_bg_clr (str, optional): Background color for the label. Default: None.
    :param dropdown_font_size (int, optional): Font size for the dropdown text. Default: None.
    :param justify (str): Text justification in the dropdown ('left', 'center', 'right'). Default: 'center'.
    :param dropdown_width (int, optional): Width of the dropdown widget in characters. Default: None.
    :param command (Callable, optional): Callback function to execute when an option is selected. Default: None.
    :param value (Any, optional): Initial selected value for the dropdown. Default: None.
    :param state (str, optional): Initial state of the dropdown ('normal', 'disabled'). Default: None.
    :param searchable (bool): If True, allows typing to filter dropdown options. Default: False.
    :param tooltip_txt (str, optional): Tooltip text to display on hover. Default: None.
    :param tooltip_key (str, optional): Key for tooltip lookup in TOOLTIPS dictionary. For dictionary, see `simba.assets.lookups.tooptips.json`. Default: None.

    :example:
    >>> dropdown = SimBADropDown(parent=parent_frm, dropdown_options=['Option 1', 'Option 2', 'Option 3'], label='Select option:', searchable=True)
    >>> selected = dropdown.get_value()
    """
    def __init__(self,
                parent: Union[Frame, Canvas, LabelFrame, Toplevel, Tk],
                dropdown_options: Union[Iterable[Any], List[Any], Tuple[Any]],
                label: Optional[str] = None,
                label_width: Optional[int] = None,
                label_font: tuple = Formats.FONT_REGULAR.value,
                label_bg_clr: Optional[str] = None,
                dropdown_font_size: Optional[int] = None,
                justify: Literal['left', 'right', 'center'] = 'center',
                img: Optional[str] = None,
                dropdown_width: Optional[int] = None,
                command: Callable = None,
                value: Optional[Any] = None,
                state: Optional[str] = None,
                searchable: bool = False,
                tooltip_txt: Optional[str] = None,
                tooltip_key: Optional[str] = None):


        super().__init__(master=parent)
        self.dropdown_var = StringVar()
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)
        if img is not None:
            self.columnconfigure(2, weight=0)
            self.dropdown_img_lbl = SimBALabel(parent=self, txt='', txt_clr='black', bg_clr=label_bg_clr, font=label_font, width=None, anchor='w', img=img, compound='left')
            self.dropdown_img_lbl.grid(row=0, column=0, sticky="w")
        else:
            self.dropdown_img_lbl = None
        self.dropdown_lbl = SimBALabel(parent=self, txt=label, txt_clr='black', bg_clr=label_bg_clr, font=label_font, width=label_width, anchor='w')
        self.dropdown_lbl.grid(row=0, column=1, sticky=NW)
        self.dropdown_options = dropdown_options
        self.original_options = deepcopy(self.dropdown_options)
        self.command, self.searchable = command, searchable
        if dropdown_font_size is None:
            drop_down_font = None
        else:
            drop_down_font = ("Poppins", dropdown_font_size)
        self.combobox_state = 'normal' if searchable else "readonly"
        self.dropdown = Combobox(self, textvariable=self.dropdown_var, font=drop_down_font, values=self.dropdown_options, state=self.combobox_state, width=dropdown_width, justify=justify)
        if searchable: self.bind_combobox_keys()
        self.dropdown.grid(row=0, column=2, sticky="w")
        if value is not None: self.set_value(value=value)
        if command is not None:
            self.command = command
            self.dropdown.bind("<<ComboboxSelected>>", self.on_select)
        if state == 'disabled':
            self.disable()
        if isinstance(tooltip_txt, str):
            CreateToolTip(widget=self.dropdown_lbl, text=tooltip_txt)
        elif tooltip_key in TOOLTIPS.keys():
            CreateToolTip(widget=self.dropdown_lbl, text=TOOLTIPS[tooltip_key])
            if img is not None:
                CreateToolTip(widget=self.dropdown_img_lbl, text=TOOLTIPS[tooltip_key])


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
        if self.searchable: self.dropdown['values'] = self.original_options

    def set_width(self, width: int):
        self.dropdown.configure(width=width)

    def change_options(self, values: List[str], set_index: Optional[int] = None, set_str: Optional[str] = None, auto_change_width: Optional[bool] = True):
        self.dropdown_var.set('')
        self.dropdown['values'] = values
        if isinstance(set_index, int) and (0 <= set_index <= len(values) - 1):
            self.dropdown_var.set(values[set_index])
        elif (set_str is not None) and (set_str in values):
            self.dropdown_var.set(set_str)
        elif self.searchable and (set_str is not None):
            self.dropdown_var.set(set_str)
        else:
            self.dropdown_var.set(values[0])
        if auto_change_width: self.set_width(width=max(5, max(len(s) for s in values)))
        if self.dropdown['values'] == ('',) or len(values) == 0 and not self.searchable:
            self.disable()
        else:
            self.enable()
        self.dropdown.configure(state='normal' if self.searchable else "readonly")

    def bind_combobox_keys(self):
        self.dropdown.bind("<KeyRelease>", self._on_key_release)
        if self.searchable:
            self.dropdown.bind("<Button-1>", lambda e: self._reset_to_all_options())

    def _on_key_release(self, event):
        if event.keysym in ['Up', 'Down', 'Left', 'Right', 'Tab', 'Return']:
            return
        try:
            cursor_pos = self.dropdown.index(INSERT)
        except:
            cursor_pos = len(self.dropdown_var.get())
        current_text = self.dropdown_var.get()
        filtered_options = [x for x in self.original_options if current_text.lower() in x.lower()]
        if current_text != '':
            self.dropdown['values'] = filtered_options
            # Restore the text and cursor position
            self.dropdown_var.set(current_text)
            self.dropdown.icursor(cursor_pos)
        else:
            self.dropdown['values'] = self.original_options
            self.dropdown_var.set('')
            self.dropdown.icursor(0)

    def _reset_to_all_options(self):
        if self.searchable and self.dropdown_var.get() == '':
            self.dropdown['values'] = self.original_options

class DropDownMenu(Frame):

    """
    Legacy, use :func:`simba.ui.tkinter_functions.SimBADropDown`.
    """
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
                 title: Optional[str] = None,
                 lblwidth: Optional[int] = None,
                 file_types=None,
                 bg_clr: Optional[str] = 'white',
                 dropdown: Union[DropDownMenu, SimBADropDown] = None,
                 entry_width: Optional[int] = 20,
                 status: Optional[str] = None,
                 lbl_icon: Optional[str] = None,
                 font: Tuple = Formats.FONT_REGULAR.value,
                 initialdir: Optional[Union[str, os.PathLike]] = None,
                 initial_path: Optional[Union[str, os.PathLike]] = None,
                 tooltip_txt: Optional[str] = None,
                 tooltip_key: Optional[str] = None,
                 **kw):

        self.title, self.dropdown, self.initialdir = title, dropdown, initialdir
        self.file_type = file_types
        self.color = color if color is not None else "black"
        self.lblwidth = lblwidth if lblwidth is not None else 0
        self.parent = parent
        Frame.__init__(self, master=parent, **kw)
        browse_icon = ImageTk.PhotoImage(image=PIL.Image.open(MENU_ICONS["browse"]["icon_path"]))
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=0)
        if lbl_icon is not None:
            self.columnconfigure(3, weight=0)
            self.lbl_icon = SimBALabel(parent=self, txt='', txt_clr='black', font=font, width=None, anchor='w', img=lbl_icon, compound=None)
            self.lbl_icon.grid(row=0, column=0, sticky="w")
        else:
            self.lbl_icon = None
        self.filePath = StringVar()
        self.lblName = Label(self, text=fileDescription, fg=str(self.color), width=str(self.lblwidth), anchor=W, font=Formats.FONT_REGULAR.value)
        self.lblName.grid(row=0, column=1, sticky=W)
        self.entPath = Label(self, textvariable=self.filePath, relief=SUNKEN, font=Formats.FONT_REGULAR.value, bg=bg_clr, width=entry_width)
        self.entPath.grid(row=0, column=2)
        self.btnFind = SimbaButton(parent=self, txt=Defaults.BROWSE_FILE_BTN_TEXT.value, font=font, cmd=self.setFilePath, img=browse_icon)
        self.btnFind.grid(row=0, column=3)
        self.filePath.set(Defaults.NO_FILE_SELECTED_TEXT.value)
        if initial_path is not None:
            self.filePath.set(initial_path)
        if status is not None:
            self.set_state(setstatus=status)
        if tooltip_txt is not None and isinstance(tooltip_txt, str):
            CreateToolTip(widget=self.lblName, text=tooltip_txt)
        elif tooltip_key in TOOLTIPS.keys():
            CreateToolTip(widget=self.lblName, text=TOOLTIPS[tooltip_key])

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


def SimBARadioButton(parent: Union[Frame, Canvas, LabelFrame, Toplevel],
                     txt: str,
                     variable: Union[BooleanVar, StringVar],
                     txt_clr: Optional[str] = 'black',
                     font: Optional[Tuple] = Formats.FONT_REGULAR.value,
                     compound: Optional[str] = 'left',
                     img: Optional[Union[ImageTk.PhotoImage, str]] = None,
                     enabled: Optional[bool] = True,
                     tooltip_txt: Optional[str] = None,
                     value: bool = False,
                     cmd: Optional[Callable] = None,
                     cmd_kwargs: Optional[Dict[Any, Any]] = None) -> Radiobutton:

    if isinstance(img, str):
        img = ImageTk.PhotoImage(image=PIL.Image.open(MENU_ICONS[img]["icon_path"]))

    if cmd_kwargs is None:
        cmd_kwargs = {}

    def execute_command():
        if cmd:
            evaluated_kwargs = {k: (v() if callable(v) else v) for k, v in cmd_kwargs.items()}
            cmd(**evaluated_kwargs)

    if cmd is not None:
        command = execute_command
    else:
        command = None

    btn = Radiobutton(parent,
                      text=txt,
                      font=font,
                      image=img,
                      fg=txt_clr,
                      variable=variable,
                      value=value,
                      compound=compound,
                      command=command)

    if img is not None:
        btn.image = img

    if not enabled:
        btn.config(state=DISABLED)

    if tooltip_txt is not None:
        CreateToolTip(widget=btn, text=tooltip_txt)

    return btn

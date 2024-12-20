__author__ = "Simon Nilsson"

import os
from tkinter import *
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import PIL.Image
from PIL import ImageTk

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (DropDownMenu, Entry_Box, FileSelect, hxtScrollbar)
from simba.utils.checks import check_float, check_int, check_valid_lst, check_instance
from simba.utils.enums import Formats, Options
from simba.utils.errors import CountError, NoFilesFoundError
from simba.utils.lookups import (get_color_dict, get_icons_paths, get_named_colors)
from simba.utils.read_write import find_core_cnt


def create_cb_frame(cb_titles: List[str],
                    main_frm: Optional[Union[Frame, Canvas, LabelFrame, ttk.Frame]] = None,
                    frm_title: Optional[str] = '',
                    idx_row: Optional[int] = -1,
                    command: Optional[Callable[[str], Any]] = None) -> Dict[str, BooleanVar]:
    """
    Creates a labelframe with checkboxes and inserts the labelframe into a window.

    .. image:: _static/img/create_cb_frame.png
       :width: 200
       :align: center

    .. note::
       One checkbox will be created per ``cb_titles``. The checkboxes will be labeled according to the ``cb_titles``.
       If checking/un-checking the box should have some effect, pass that function as ``command`` which takes the name of the checked/unchecked box.

    :param Optional[Union[Frame, Canvas, LabelFrame, ttk.Frame]] main_frm: The pop-up window to insert the labelframe into.
    :param List[str] cb_titles: List of strings representing the names of the checkboxes.
    :param Optional[str] frm_title: Title of the frame.
    :param Optional[int] idx_row: The location in main_frm to create the LabelFrame. If -1, then at the bottom.
    :param Optional[Callable[[str], Any]] frm_title: Optional function callable associated with checking/unchecking the checkboxes.
    :return Dict[str, BooleanVar]: Dictionary holding the ``cb_titles`` as keys and the BooleanVar representing if the checkbox is ticked or not.

    :example:
    >>> create_cb_frame(cb_titles=['Attack', 'Sniffing', 'Rearing'], frm_title='My classifiers')
    """

    check_valid_lst(data=cb_titles, source=f'{create_cb_frame.__name__} cb_titles', valid_dtypes=(str,), min_len=1)
    check_int(name=f'{create_cb_frame.__name__} idx_row', value=idx_row, min_value=-1)

    if main_frm is not None:
        check_instance(source=f'{create_cb_frame.__name__} parent_frm', accepted_types=(Frame, Canvas, LabelFrame, ttk.Frame), instance=main_frm)
    else:
        main_frm = Toplevel(); main_frm.minsize(960, 720); main_frm.lift()
    if idx_row == -1:
        idx_row = int(len(list(main_frm.children.keys())))
    cb_frm = LabelFrame(main_frm, text=frm_title, font=Formats.LABELFRAME_HEADER_FORMAT.value)
    cb_dict = {}
    for cnt, title in enumerate(cb_titles):
        cb_dict[title] = BooleanVar(value=False)
        if command is not None:
            cb = Checkbutton(cb_frm, text=title, variable=cb_dict[title], command=lambda k=cb_titles[cnt]: command(k))
        else:
            cb = Checkbutton(cb_frm, text=title, variable=cb_dict[title])
        cb.grid(row=cnt, column=0, sticky=NW)
    cb_frm.grid(row=idx_row, column=0, sticky=NW)

    #main_frm.mainloop()

    return cb_dict

def create_dropdown_frame(drop_down_titles: List[str],
                          drop_down_options: List[str],
                          frm_title: Optional[str] = '',
                          idx_row: Optional[int] = -1,
                          main_frm: Optional[Union[Frame, Canvas, LabelFrame, ttk.Frame]] = None) -> Dict[str, DropDownMenu]:

    """
    Creates a labelframe with dropdowns.

    .. image:: _static/img/create_dropdown_frame.png
       :width: 300
       :align: center

    :param Optional[Union[Frame, Canvas, LabelFrame, ttk.Frame]] main_frm: The pop-up window to insert the labelframe into. If None, one will be created.
    :param List[str] drop_down_titles: The titles of the dropdown menus.
    :param List[str] drop_down_options: The options in each dropdown. Note: All dropdowns must have the same options.
    :param Optional[str] frm_title: Title of the frame.
    :return Dict[str, BooleanVar]: Dictionary holding the ``drop_down_titles`` as keys and the drop-down menus as values.

    :example:
    >>> create_dropdown_frame(drop_down_titles=['Dropdown 1', 'Dropdown 2', 'Dropdown 2'], drop_down_options=['Option 1', 'Option 2'], frm_title='My dropdown frame')
    """

    check_valid_lst(data=drop_down_titles, source=f'{create_dropdown_frame.__name__} drop_down_titles', valid_dtypes=(str,), min_len=1)
    check_valid_lst(data=drop_down_options, source=f'{create_dropdown_frame.__name__} drop_down_options', valid_dtypes=(str,), min_len=2)
    check_int(name=f'{create_cb_frame.__name__} idx_row', value=idx_row, min_value=-1)
    if main_frm is not None:
        check_instance(source=f'{create_cb_frame.__name__} parent_frm', accepted_types=(Frame, Canvas, LabelFrame, ttk.Frame), instance=main_frm)
    else:
        main_frm = Toplevel(); main_frm.minsize(960, 720); main_frm.lift()
    if idx_row == -1:
        idx_row = int(len(list(main_frm.children.keys())))
    dropdown_frm = LabelFrame(main_frm, text=frm_title, font=Formats.LABELFRAME_HEADER_FORMAT.value)
    dropdown_dict = {}
    for cnt, title in enumerate(drop_down_titles):
        dropdown_dict[title] = DropDownMenu(dropdown_frm, title, drop_down_options, "35")
        dropdown_dict[title].setChoices(drop_down_options[0])
        dropdown_dict[title].grid(row=cnt, column=0, sticky=NW)
    dropdown_frm.grid(row=idx_row, column=0, sticky=NW)
    #main_frm.mainloop()
    return dropdown_dict

def create_run_frm(run_function: Callable,
                   title: Optional[str] = "RUN",
                   btn_txt_clr: Optional[str] = "black") -> None:
    """
    Create a label frame with a single button with a specified callback.

    :param Callable run_function: The function/method callback of the button.
    :param str title: The title of the frame.
    """
    if hasattr(self, "run_frm"):
        self.run_frm.destroy()
        self.run_btn.destroy()
    self.run_frm = LabelFrame(
        self.main_frm,
        text=title,
        font=Formats.LABELFRAME_HEADER_FORMAT.value,
        fg=btn_txt_clr,
    )
    self.run_btn = Button(self.run_frm, text=title, fg="blue", command=lambda: run_function()
    )
    self.run_frm.grid(row=self.children_cnt_main() + 1, column=0, sticky=NW)
    self.run_btn.grid(row=0, column=0, sticky=NW)
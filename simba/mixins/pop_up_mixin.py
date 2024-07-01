__author__ = "Simon Nilsson"

import os
from tkinter import *
from tkinter import ttk
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL.Image
from PIL import ImageTk

from simba.mixins.config_reader import ConfigReader
from simba.ui.tkinter_functions import (DropDownMenu, Entry_Box, FileSelect,
                                        hxtScrollbar)
from simba.utils.checks import (check_float, check_instance, check_int,
                                check_valid_lst)
from simba.utils.enums import Formats, Options
from simba.utils.errors import CountError, NoFilesFoundError
from simba.utils.lookups import (get_color_dict, get_icons_paths,
                                 get_named_colors)
from simba.utils.read_write import find_core_cnt


class PopUpMixin(object):
    """
    Methods for pop-up windows in SimBA. E.g., common methods for creating pop-up windows with drop-downs,
    checkboxes, entry-boxes, listboxes etc.

    :param str title: Pop-up window title
    :param Optional[configparser.Configparser] config_path: path to SimBA project_config.ini. If path, the project config is read in. If None, the project config is not read in.
    :param tuple size: HxW of the pop-up window. The size of the pop-up window in pixels.
    :param bool main_scrollbar: If True, the pop-up window is scrollable.
    """

    def __init__(self,
                 title: str,
                 config_path: Optional[str] = None,
                 main_scrollbar: Optional[bool] = True,
                 size: Tuple[int, int] = (960, 720)):

        self.root = Toplevel()
        self.root.minsize(size[0], size[1])
        self.root.wm_title(title)
        self.root.lift()
        if main_scrollbar:
            self.main_frm = Canvas(hxtScrollbar(self.root))
        else:
            self.main_frm = Canvas(self.root)
        self.main_frm.pack(fill="both", expand=True)
        self.palette_options = Options.PALETTE_OPTIONS.value
        self.resolutions = Options.RESOLUTION_OPTIONS.value
        self.shading_options = Options.HEATMAP_SHADING_OPTIONS.value
        self.heatmap_bin_size_options = Options.HEATMAP_BIN_SIZE_OPTIONS.value
        self.dpi_options = Options.DPI_OPTIONS.value
        self.colors = get_named_colors()
        self.colors_dict = get_color_dict()
        self.cpu_cnt, _ = find_core_cnt()
        self.menu_icons = get_icons_paths()
        for k in self.menu_icons.keys():
            self.menu_icons[k]["img"] = ImageTk.PhotoImage(image=PIL.Image.open(os.path.join(os.path.dirname(__file__), self.menu_icons[k]["icon_path"])))
        if config_path:
            ConfigReader.__init__(self, config_path=config_path, read_video_info=False)

    def create_clf_checkboxes(self,
                              main_frm: Frame,
                              clfs: List[str],
                              title: str = "SELECT CLASSIFIER ANNOTATIONS"):
        """
        Creates a labelframe with one checkbox per classifier, and inserts the labelframe into the bottom of the pop-up window.

        .. note::
           Legacy. Use ``create_cb_frame`` instead.

        """

        self.choose_clf_frm = LabelFrame(self.main_frm, text=title, font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.clf_selections = {}
        for clf_cnt, clf in enumerate(clfs):
            self.clf_selections[clf] = BooleanVar(value=False)
            self.calculate_distance_moved_cb = Checkbutton(self.choose_clf_frm, text=clf, variable=self.clf_selections[clf])
            self.calculate_distance_moved_cb.grid(row=clf_cnt, column=0, sticky=NW)
        self.choose_clf_frm.grid(row=self.children_cnt_main(), column=0, sticky=NW)

    def create_cb_frame(self,
                        cb_titles: List[str],
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
        >>> PopUpMixin.create_cb_frame(cb_titles=['Attack', 'Sniffing', 'Rearing'], frm_title='My classifiers')
        """

        check_valid_lst(data=cb_titles, source=f'{PopUpMixin.create_cb_frame.__name__} cb_titles', valid_dtypes=(str,), min_len=1)
        check_int(name=f'{PopUpMixin.create_cb_frame.__name__} idx_row', value=idx_row, min_value=-1)

        if main_frm is not None:
            check_instance(source=f'{PopUpMixin.create_cb_frame.__name__} parent_frm', accepted_types=(Frame, Canvas, LabelFrame, ttk.Frame), instance=main_frm)
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
        # main_frm.mainloop()
        return cb_dict

    def place_frm_at_top_right(self, frm: Toplevel):
        """
        Place a TopLevel tkinter pop-up at the top right of the monitor. Note: call before putting scrollbars or converting to Canvas.
        """
        screen_width, screen_height = frm.winfo_screenwidth(), frm.winfo_screenheight()
        window_width, window_height = frm.winfo_width(), frm.winfo_height()
        x_position = screen_width - window_width
        frm.geometry(f"{window_width}x{window_height}+{x_position}+{0}")

    def create_dropdown_frame(self,
                              drop_down_titles: List[str],
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
        >>> PopUpMixin.create_dropdown_frame(drop_down_titles=['Dropdown 1', 'Dropdown 2', 'Dropdown 2'], drop_down_options=['Option 1', 'Option 2'], frm_title='My dropdown frame')
        """

        check_valid_lst(data=drop_down_titles, source=f'{PopUpMixin.create_dropdown_frame.__name__} drop_down_titles',
                        valid_dtypes=(str,), min_len=1)
        check_valid_lst(data=drop_down_options, source=f'{PopUpMixin.create_dropdown_frame.__name__} drop_down_options', valid_dtypes=(str,), min_len=2)
        check_int(name=f'{PopUpMixin.create_cb_frame.__name__} idx_row', value=idx_row, min_value=-1)
        if main_frm is not None:
            check_instance(source=f'{PopUpMixin.create_cb_frame.__name__} parent_frm', accepted_types=(Frame, Canvas, LabelFrame, ttk.Frame), instance=main_frm)
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
        # main_frm.mainloop()
        return dropdown_dict

    def create_time_bin_entry(self):
        if hasattr(self, "time_bin_frm"):
            self.time_bin_frm.destroy()
        self.time_bin_frm = LabelFrame(
            self.main_frm, text="TIME BIN", font=Formats.LABELFRAME_HEADER_FORMAT.value
        )
        self.time_bin_entrybox = Entry_Box(
            self.time_bin_frm, "Time-bin size (s): ", "12"
        )
        self.time_bin_entrybox.grid(row=0, column=0, sticky=NW)
        self.time_bin_frm.grid(row=self.children_cnt_main(), column=0, sticky=NW)

    def create_run_frm(self,run_function: Callable,
        title: Optional[str] = "RUN",
        btn_txt_clr: Optional[str] = "black",
    ) -> None:
        """
        Create a label frame with a single button with a specified callback.

        :param object run_function: The function/method callback of the button.
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
        self.run_btn = Button(
            self.run_frm, text=title, fg="blue", command=lambda: run_function()
        )
        self.run_frm.grid(row=self.children_cnt_main() + 1, column=0, sticky=NW)
        self.run_btn.grid(row=0, column=0, sticky=NW)

    def create_choose_number_of_body_parts_frm(self, project_body_parts: List[str], run_function: object):
        """
        Many menus depend on how many animals the user choose to compute metrics for. Thus, we need to populate the menus
        dynamically. This function creates a single drop-down menu where the user select the number of animals the
        user choose to compute metrics for. It inserts this drop-down iat the bottom of the pop-up window, and ties this dropdown menu
        choice to a callback.

        :param List[str] project_body_parts: Options of the dropdown menu.
        :param object run_function: Function tied to the choice in the dropdown menu.
        """

        self.bp_cnt_frm = LabelFrame(
            self.main_frm,
            text="SELECT NUMBER OF BODY-PARTS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        self.bp_cnt_dropdown = DropDownMenu(
            self.bp_cnt_frm,
            "# of body-parts",
            list(range(1, len(project_body_parts) + 1)),
            "12",
        )
        self.bp_cnt_dropdown.setChoices(1)
        self.bp_cnt_confirm_btn = Button(
            self.bp_cnt_frm,
            text="Confirm",
            command=lambda: self.create_choose_bp_frm(project_body_parts, run_function),
        )
        self.bp_cnt_frm.grid(row=0, sticky=NW)
        self.bp_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        self.bp_cnt_confirm_btn.grid(row=0, column=1, sticky=NW)

    def add_to_listbox_from_entrybox(self, list_box: Listbox, entry_box: Entry_Box):
        """
        Add a value that populates a tkinter entry_box to a tkinter listbox.

        :param Listbox list_box: The tkinter Listbox to add the value to.
        :param Entry_Box entry_box: The tkinter Entry_Box containing the value that should be added to the list_box.
        """

        value = entry_box.entry_get
        check_float(name="VALUE", value=value)
        list_box_content = [float(x) for x in list_box.get(0, END)]
        if float(value) not in list_box_content:
            list_box.insert(0, value)

    def add_value_to_listbox(self, list_box: Listbox, value: float):
        """
        Add a float value to a tkinter listbox.

        :param Listbox list_box: The tkinter Listbox to add the value to.
        :param float value: Value to add to the listbox.
        """
        list_box.insert(0, value)

    def add_values_to_several_listboxes(
        self, list_boxes: List[Listbox], values: List[float]
    ):
        """
        Add N values to N listboxes. E.g., values[0] will be added to list_boxes[0].

        :param List[Listbox] list_boxes: List of Listboxes that the values should be added to.
        :param List[float] values: List of floats that will be added to the list_boxes.
        """

        if len(list_boxes) != len(values):
            raise CountError(
                msg="Value count and list-boxes count are not equal",
                source=self.__class__.__name__,
            )
        for i in range(len(list_boxes)):
            list_boxes[i].insert(0, values[i])

    def remove_from_listbox(self, list_box: Listbox):
        """
        Remove the current selection in a listbox from a listbox.

        :param Listbox list_box: The listbox that the current selection should be removed from.
        """

        selection = list_box.curselection()
        if selection:
            list_box.delete(selection[0])

    def update_file_select_box_from_dropdown(
        self, filename: str, fileselectbox: FileSelect
    ):
        """
        Updates the text inside a tkinter FileSelect entrybox with a new string.
        """
        fileselectbox.filePath.set(filename)

    def check_if_selected_video_path_exist_in_project(
        self, video_path: Union[str, os.PathLike]
    ):
        if not os.path.isfile(video_path):
            raise NoFilesFoundError(
                msg=f"Selected video {os.path.basename(video_path)} is not a video file in the SimBA project video directory."
            )

    def create_choose_bp_frm(self, project_body_parts, run_function):
        if hasattr(self, "body_part_frm"):
            self.body_part_frm.destroy()
        self.body_parts_dropdowns = {}
        self.body_part_frm = LabelFrame(
            self.main_frm,
            text="CHOOSE ANIMAL BODY-PARTS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            name="choose animal body-parts",
        )
        self.body_part_frm.grid(row=self.children_cnt_main(), sticky=NW)
        for bp_cnt in range(int(self.bp_cnt_dropdown.getChoices())):
            self.body_parts_dropdowns[bp_cnt] = DropDownMenu(
                self.body_part_frm,
                f"Body-part {str(bp_cnt+1)}:",
                project_body_parts,
                "25",
            )
            self.body_parts_dropdowns[bp_cnt].grid(row=bp_cnt, column=0, sticky=NW)
            self.body_parts_dropdowns[bp_cnt].setChoices(project_body_parts[bp_cnt])
        self.create_run_frm(run_function=run_function)

    def choose_bp_frm(self, parent: LabelFrame, bp_options: list):
        self.body_parts_dropdowns = {}
        self.body_part_frm = LabelFrame(
            parent,
            text="CHOOSE ANIMAL BODY-PARTS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            name="choose animal body-parts",
        )
        self.body_part_frm.grid(row=self.frame_children(frame=parent), sticky=NW)
        for bp_cnt in range(int(self.animal_cnt_dropdown.getChoices())):
            self.body_parts_dropdowns[bp_cnt] = DropDownMenu(
                self.body_part_frm, f"Body-part {str(bp_cnt + 1)}:", bp_options, "20"
            )
            self.body_parts_dropdowns[bp_cnt].grid(row=bp_cnt, column=0, sticky=NW)
            self.body_parts_dropdowns[bp_cnt].setChoices(bp_options[bp_cnt])

    def children_cnt_main(self) -> int:
        """
        Find the number of children (e.g., labelframes) currently exist within a main pop-up window. Useful for finding the
        row at which a new frame within the window should be inserted.
        """

        return int(len(list(self.main_frm.children.keys())))

    def frame_children(self, frame: Frame) -> int:
        """
        Find the number of children (e.g., labelframes) currently exist within specified frame.Similar to ``children_cnt_main``,
        but accepts a specific frame rather than the main frame beeing hardcoded.
        """

        return int(len(list(frame.children.keys())))

    def update_config(self) -> None:
        """Helper to update the SimBA project config file"""
        with open(self.config_path, "w") as f:
            self.config.write(f)

    def show_smoothing_entry_box_from_dropdown(self, choice: str):
        if choice == "None":
            self.smoothing_time_eb.grid_forget()
        if (choice == "Gaussian") or (choice == "Savitzky Golay"):
            self.smoothing_time_eb.grid(row=0, column=1, sticky=E)

    def choose_bp_threshold_frm(self, parent: LabelFrame):
        self.probability_frm = LabelFrame(
            parent,
            text="PROBABILITY THRESHOLD",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        self.probability_frm.grid(
            row=self.frame_children(frame=parent), column=0, sticky=NW
        )
        self.probability_entry = Entry_Box(
            self.probability_frm, "Probability threshold: ", labelwidth=20
        )
        self.probability_entry.entry_set("0.00")
        self.probability_entry.grid(row=0, column=0, sticky=NW)

    def enable_dropdown_from_checkbox(
        self, check_box_var: BooleanVar, dropdown_menus: List[DropDownMenu]
    ):
        """
        Given a single checkbox, enable a bunch of dropdowns if the checkbox is ticked, and disable the dropdowns if
        the checkbox is un-ticked.

        :param BooleanVar check_box_var: The checkbox associated tkinter BooleanVar.
        :param List[DropDownMenu] dropdown_menus: List of dropdowns which status is controlled by the ``check_box_var``.
        """

        if check_box_var.get():
            for menu in dropdown_menus:
                menu.enable()
        else:
            for menu in dropdown_menus:
                menu.disable()

    def create_entry_boxes_from_entrybox(self, count: int, parent: Frame, current_entries: list):
        check_int(name="CLASSIFIER COUNT", value=count, min_value=1)
        for entry in current_entries:
            entry.destroy()
        for clf_cnt in range(int(count)):
            entry = Entry_Box(parent, f"Classifier {str(clf_cnt+1)}:", labelwidth=15)
            current_entries.append(entry)
            entry.grid(row=clf_cnt + 2, column=0, sticky=NW)

    def create_animal_names_entry_boxes(self, animal_cnt: str):
        check_int(name="NUMBER OF ANIMALS", value=animal_cnt, min_value=0)
        if hasattr(self, "animal_names_frm"):
            self.animal_names_frm.destroy()
        if not hasattr(self, "multi_animal_id_list"):
            self.multi_animal_id_list = []
            for i in range(int(animal_cnt)):
                self.multi_animal_id_list.append(f"Animal {i+1}")
        self.animal_names_frm = Frame(self.animal_settings_frm, pady=5, padx=5)
        self.animal_name_entry_boxes = {}
        for i in range(int(animal_cnt)):
            self.animal_name_entry_boxes[i + 1] = Entry_Box(
                self.animal_names_frm, f"Animal {str(i+1)} name: ", "25"
            )
            if i <= len(self.multi_animal_id_list) - 1:
                self.animal_name_entry_boxes[i + 1].entry_set(
                    self.multi_animal_id_list[i]
                )
            self.animal_name_entry_boxes[i + 1].grid(row=i, column=0, sticky=NW)

        self.animal_names_frm.grid(row=1, column=0, sticky=NW)

    def enable_entrybox_from_checkbox(
        self,
        check_box_var: BooleanVar,
        entry_boxes: List[Entry_Box],
        reverse: bool = False,
    ):
        """
        Given a single checkbox, enable or disable a bunch of entry-boxes based on the status of the checkbox.

        :param BooleanVar check_box_var: The checkbox associated tkinter BooleanVar.
        :param List[Entry_Box] entry_boxes: List of entry-boxes which status is controlled by the ``check_box_var``.
        :param bool reverse: If False, the entry-boxes are enabled with the checkbox is ticked. Else, the entry-boxes are enabled if checkbox is unticked. Default: False.
        """

        if reverse:
            if check_box_var.get():
                for box in entry_boxes:
                    box.set_state("disable")
            else:
                for box in entry_boxes:
                    box.set_state("normal")
        else:
            if check_box_var.get():
                for box in entry_boxes:
                    box.set_state("normal")
            else:
                for box in entry_boxes:
                    box.set_state("disable")


    # def quit(self, e):
    #     self.main_frm.quit()
    #     self.main_frm.destroy()
    #
    # def callback(self, url):
    #     webbrowser.open_new(url)
    #
    # def move_app(self, e):
    #     print(f'+{e.x_root}+{e.y_root}')
    #     self.main_frm.geometry(f'+{e.x_root}+{e.y_root}')
    #     #print(f'+{e.x_root}x{e.y_root}')
    #     #self.main_frm.config(width=e.x_root, height=e.y_root)
    #     #self.main_frm.update()


# test = PopUpMixin(config_path='/Users/simon/Desktop/envs/troubleshooting/two_animals_16bp_032023/project_folder/project_config.ini',
#                   title='ss')
# test.create_import_pose_menu(parent_frm=test.main_frm)

# test = PopUpMixin(config_path='/Users/simon/Desktop/envs/troubleshooting/two_animals_16bp_032023/project_folder/project_config.ini',
#                   title='ss')
# test.create_import_videos_menu(parent_frm=test.main_frm)

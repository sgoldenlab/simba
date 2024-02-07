__author__ = "Simon Nilsson"

import os
from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.data_plotter import DataPlotter
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, DropDownMenu
from simba.utils.enums import Formats, Keys, Links, Paths
from simba.utils.errors import InvalidInputError
from simba.utils.read_write import get_file_name_info_in_directory


class DataPlotterPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):

        PopUpMixin.__init__(self, title="CREATE DATA PLOTS")
        ConfigReader.__init__(self, config_path=config_path)
        self.color_lst = list(self.colors_dict.keys())
        self.data_path = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
        self.files_found_dict = get_file_name_info_in_directory(
            directory=self.data_path, file_type=self.file_type
        )
        self.animal_cnt_options = list(range(1, self.animal_cnt + 1))
        self.style_settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="STYLE SETTINGS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.DATA_TABLES.value,
        )
        self.rounding_decimals_options = list(range(0, 11))
        self.font_thickness_options = list(range(1, 11))
        self.resolution_dropdown = DropDownMenu(
            self.style_settings_frm, "RESOLUTION:", self.resolutions, "18"
        )
        self.rounding_decimals_dropdown = DropDownMenu(
            self.style_settings_frm,
            "DECIMAL ACCURACY:",
            self.rounding_decimals_options,
            "18",
        )
        self.background_color_dropdown = DropDownMenu(
            self.style_settings_frm, "BACKGROUND COLOR: ", self.color_lst, "18"
        )
        self.font_color_dropdown = DropDownMenu(
            self.style_settings_frm, "HEADER COLOR: ", self.color_lst, "18"
        )
        self.font_thickness_dropdown = DropDownMenu(
            self.style_settings_frm,
            "FONT THICKNESS: ",
            self.font_thickness_options,
            "18",
        )

        self.background_color_dropdown.setChoices(choice="White")
        self.font_color_dropdown.setChoices(choice="Black")
        self.resolution_dropdown.setChoices(self.resolutions[1])
        self.rounding_decimals_dropdown.setChoices(2)
        self.font_thickness_dropdown.setChoices(1)

        self.body_parts_frm = LabelFrame(
            self.main_frm,
            text="CHOOSE BODY-PARTS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
        )
        self.number_of_animals_dropdown = DropDownMenu(
            self.body_parts_frm,
            "# Animals:",
            self.animal_cnt_options,
            "16",
            com=self.__populate_body_parts_menu,
        )
        self.number_of_animals_dropdown.setChoices(self.animal_cnt_options[0])
        self.__populate_body_parts_menu(self.animal_cnt_options[0])

        self.settings_frm = LabelFrame(
            self.main_frm,
            text="VISUALIZATION SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
        )
        self.data_frames_var = BooleanVar()
        self.data_videos_var = BooleanVar()
        data_frames_cb = Checkbutton(
            self.settings_frm, text="Create frames", variable=self.data_frames_var
        )
        data_videos_cb = Checkbutton(
            self.settings_frm, text="Create videos", variable=self.data_videos_var
        )

        self.run_frm = LabelFrame(
            self.main_frm,
            text="RUN",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
            fg="black",
        )
        self.run_single_video_frm = LabelFrame(
            self.run_frm,
            text="SINGLE VIDEO",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
            fg="black",
        )
        self.run_single_video_btn = Button(
            self.run_single_video_frm,
            text="Create single video",
            fg="blue",
            command=lambda: self.__create_data_plots(multiple_videos=False),
        )
        self.single_video_dropdown = DropDownMenu(
            self.run_single_video_frm,
            "Video:",
            list(self.files_found_dict.keys()),
            "12",
        )
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])
        self.run_multiple_videos = LabelFrame(
            self.run_frm,
            text="MULTIPLE VIDEO",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
            fg="black",
        )
        self.run_multiple_video_btn = Button(
            self.run_multiple_videos,
            text="Create multiple videos ({} video(s) found)".format(
                str(len(list(self.files_found_dict.keys())))
            ),
            fg="blue",
            command=lambda: self.__create_data_plots(multiple_videos=True),
        )

        self.style_settings_frm.grid(row=0, sticky=NW)
        self.resolution_dropdown.grid(row=0, sticky=NW)
        self.rounding_decimals_dropdown.grid(row=1, sticky=NW)
        self.background_color_dropdown.grid(row=2, sticky=NW)
        self.font_color_dropdown.grid(row=3, sticky=NW)
        self.font_thickness_dropdown.grid(row=4, sticky=NW)

        self.body_parts_frm.grid(row=1, sticky=NW)
        self.number_of_animals_dropdown.grid(row=0, sticky=NW)

        self.settings_frm.grid(row=2, sticky=NW)
        data_frames_cb.grid(row=0, sticky=NW)
        data_videos_cb.grid(row=1, sticky=NW)

        self.run_frm.grid(row=3, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)
        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

        self.main_frm.mainloop()

    def __populate_body_parts_menu(self, choice):
        if hasattr(self, "bp_dropdowns"):
            for k, v in self.bp_dropdowns.items():
                self.bp_dropdowns[k].destroy()
                self.bp_colors[k].destroy()
        self.bp_dropdowns, self.bp_colors = {}, {}
        for animal_cnt in range(int(self.number_of_animals_dropdown.getChoices())):
            self.bp_dropdowns[animal_cnt] = DropDownMenu(
                self.body_parts_frm,
                "Body-part {}:".format(str(animal_cnt + 1)),
                self.body_parts_lst,
                "16",
            )
            self.bp_dropdowns[animal_cnt].setChoices(self.body_parts_lst[animal_cnt])
            self.bp_dropdowns[animal_cnt].grid(row=animal_cnt + 1, column=0, sticky=NW)

            self.bp_colors[animal_cnt] = DropDownMenu(
                self.body_parts_frm, "", self.color_lst, "2"
            )
            self.bp_colors[animal_cnt].setChoices(self.color_lst[animal_cnt])
            self.bp_colors[animal_cnt].grid(row=animal_cnt + 1, column=1, sticky=NW)

    def __create_data_plots(self, multiple_videos: bool):

        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [
                self.files_found_dict[self.single_video_dropdown.getChoices()]
            ]

        width = int(self.resolution_dropdown.getChoices().split("×")[0])
        height = int(self.resolution_dropdown.getChoices().split("×")[1])
        body_part_attr = []
        for k, v in self.bp_dropdowns.items():
            body_part_attr.append([v.getChoices(), self.bp_colors[k].getChoices()])
        selected_bps = [x[0] for x in body_part_attr]
        if len(list(set(selected_bps))) != len(body_part_attr):
            raise InvalidInputError(
                msg=f"Please choose unique only body-parts for plotting. Got {len(body_part_attr)} body-parts but only {len(list(set(selected_bps)))} unique",
                source=DataPlotterPopUp.__class__.__name__,
            )
        style_attr = {
            "bg_color": self.background_color_dropdown.getChoices(),
            "header_color": self.font_color_dropdown.getChoices(),
            "font_thickness": int(self.font_thickness_dropdown.getChoices()),
            "size": (int(width), int(height)),
            "data_accuracy": int(self.rounding_decimals_dropdown.getChoices()),
        }

        print(data_paths)
        data_plotter = DataPlotter(
            config_path=self.config_path,
            body_part_attr=body_part_attr,
            data_paths=data_paths,
            style_attr=style_attr,
            frame_setting=self.data_frames_var.get(),
            video_setting=self.data_videos_var.get(),
        )

        _ = data_plotter.run()


# _ = DataPlotterPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')
# _ = DataPlotterPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')

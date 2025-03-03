import multiprocessing
import os
import platform
import threading
from tkinter import *
from typing import Union

from simba.data_processors.spontaneous_alternation_calculator import \
    SpontaneousAlternationCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.spontaneous_alternation_plotter import \
    SpontaneousAlternationsPlotter
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        FileSelect)
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.errors import (AnimalNumberError, CountError,
                                InvalidInputError, NoFilesFoundError,
                                NoROIDataError)
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    find_video_of_file, get_fn_ext)


class SpontaneousAlternationPopUp(ConfigReader, PopUpMixin):
    """
    Pop-up window for setting spontaneous alternation parameters and running spontaneous alternation analysis and visualizations.

    :example:
    >>> _ = SpontaneousAlternationPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/spontenous_alternation/project_folder/project_config.ini')
    """

    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if not os.path.isfile(self.roi_coordinates_path):
            raise NoROIDataError(
                msg=f"Spontaneous alternation requires user-defined ROIs. No ROI data found at {self.roi_coordinates_path}",
                source=self.__class__.__name__,
            )
        if self.animal_cnt != 1:
            raise AnimalNumberError(
                msg=f"Spontaneous alternation 1 animal project, found {self.animal_cnt}.",
                source=self.__class__.__name__,
            )
        self.read_roi_data()
        if len(self.shape_names) < 4:
            raise CountError(
                msg=f"Spontaneous alternation requires at least {4} defined ROIs. Got {len(self.shape_names)}.",
                source=self.__class__.__name__,
            )
        if len(self.outlier_corrected_paths) == 0:
            raise NoFilesFoundError(
                msg=f"No data found in {self.outlier_corrected_dir} directory.",
                source=self.__class__.__name__,
            )
        video_files = find_files_of_filetypes_in_directory(
            directory=self.video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value
        )
        self.video_names = [get_fn_ext(x)[1] for x in video_files]
        PopUpMixin.__init__(self, title="SPONTANEOUS ALTERNATION CALCULATOR")
        self.config_path = config_path
        self.arm_dict = self.create_dropdown_frame(
            main_frm=self.main_frm,
            drop_down_titles=[
                "MAZE ARM 1:",
                "MAZE ARM 2:",
                "MAZE ARM 3:",
                "MAZE CENTER:",
            ],
            drop_down_options=self.shape_names,
            frm_title="ARM DEFINITIONS",
        )
        self.animal_settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="ANIMAL SETTINGS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.ROI_FEATURES_PLOT.value,
        )
        self.threshold_dropdown = DropDownMenu(
            self.animal_settings_frm,
            "POSE-ESTIMATION THRESHOLD: ",
            list(range(0, 105, 5)),
            labelwidth=35,
        )
        self.animal_area_dropdown = DropDownMenu(
            self.animal_settings_frm,
            "ANIMAL AREA (%): ",
            list(range(51, 101, 1)),
            labelwidth=35,
        )
        self.buffer_dropdown = DropDownMenu(
            self.animal_settings_frm,
            "ANIMAL BUFFER (MM): ",
            list(range(0, 105, 5)),
            labelwidth=35,
        )
        self.detailed_data_dropdown = DropDownMenu(
            self.animal_settings_frm,
            "SAVE DETAILED DATA: ",
            ["True", "False"],
            labelwidth=35,
        )
        self.verbose_dropdown = DropDownMenu(
            self.animal_settings_frm, "VERBOSE: ", ["True", "False"], labelwidth=35
        )

        self.threshold_dropdown.setChoices(0)
        self.animal_area_dropdown.setChoices(80)
        self.buffer_dropdown.setChoices(0)
        self.detailed_data_dropdown.setChoices("False")
        self.verbose_dropdown.setChoices("False")

        self.animal_settings_frm.grid(row=2, column=0, sticky=NW)
        self.threshold_dropdown.grid(row=0, column=0, sticky=NW)
        self.animal_area_dropdown.grid(row=1, column=0, sticky=NW)
        self.buffer_dropdown.grid(row=2, column=0, sticky=NW)
        self.detailed_data_dropdown.grid(row=3, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=4, column=0, sticky=NW)
        # self.create_run_frm(run_function=self.run_analysis, title="RUN ANALYSIS")

        self.run_analysis_frm = LabelFrame(
            self.main_frm,
            text="RUN ANALYSIS",
            font=Formats.FONT_HEADER.value,
            fg="black",
        )
        self.run_analysis_btn = Button(
            self.run_analysis_frm,
            text="RUN ANALYSIS",
            fg="blue",
            font=Formats.FONT_REGULAR.value,
            command=lambda: threading.Thread(target=self.run_analysis()).start(),
        )

        self.run_analysis_frm.grid(row=5, column=0, sticky=NW)
        self.run_analysis_btn.grid(row=0, column=0, sticky=NW)

        self.run_visualization_frm = LabelFrame(
            self.main_frm,
            text="RUN VISUALIZATION",
            font=Formats.FONT_HEADER.value,
            fg="black",
        )
        self.video_run_btn = Button(
            self.run_visualization_frm,
            text="CREATE VIDEO",
            fg="blue",
            font=Formats.FONT_REGULAR.value,
            command=lambda: self.run_visualization(),
        )
        self.single_video_dropdown = DropDownMenu(
            self.run_visualization_frm,
            "Video:",
            self.video_names,
            "12",
            com=lambda x: self.__update_single_video_file_path(filename=x),
        )
        self.select_video_file_select = FileSelect(
            self.run_visualization_frm,
            "",
            lblwidth="1",
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
            dropdown=self.single_video_dropdown,
        )
        self.single_video_dropdown.setChoices(self.video_names[0])
        self.select_video_file_select.filePath.set(self.video_names[0])
        self.run_visualization_frm.grid(row=6, column=0, sticky=NW)
        self.video_run_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)
        self.select_video_file_select.grid(row=0, column=2, sticky=NW)
        self.select_video_file_select.grid(row=0, column=3, sticky=NW)
        self.main_frm.mainloop()

    def __update_single_video_file_path(self, filename: str):
        self.select_video_file_select.filePath.set(filename)

    def __checks(self):
        self.arm_names = [
            self.arm_dict["MAZE ARM 1:"].getChoices(),
            self.arm_dict["MAZE ARM 2:"].getChoices(),
            self.arm_dict["MAZE ARM 3:"].getChoices(),
        ]
        self.center_name = self.arm_dict["MAZE CENTER:"].getChoices()
        if self.center_name in self.arm_names:
            raise InvalidInputError(
                msg=f"One ROI has been defined both as an ARM, and as the CENTER: {self.center_name}",
                source=self.__class__.__name__,
            )
        if len(list(set(self.arm_names))) != len(self.arm_names):
            raise InvalidInputError(
                msg=f"Each arm has to be unique but got {self.arm_names}",
                source=self.__class__.__name__,
            )
        self.animal_area = int(self.animal_area_dropdown.getChoices())
        self.threshold = float(self.threshold_dropdown.getChoices()) / 100
        self.buffer = int(self.buffer_dropdown.getChoices()) + 1
        if self.detailed_data_dropdown.getChoices() == "True":
            self.detailed_data = True
        else:
            self.detailed_data = False
        if self.verbose_dropdown.getChoices() == "True":
            self.verbose = True
        else:
            self.verbose = False

    def run_analysis(self):
        self.__checks()
        calculator = SpontaneousAlternationCalculator(
            config_path=self.config_path,
            arm_names=self.arm_names,
            center_name=self.center_name,
            animal_area=self.animal_area,
            threshold=self.threshold,
            buffer=self.buffer,
            verbose=self.verbose,
            detailed_data=self.detailed_data,
        )
        # calculator.run()

        threading.Thread(target=calculator.run()).start()
        calculator.save()

    def run_visualization(self):
        self.__checks()
        file_name = self.single_video_dropdown.getChoices()
        data_path = os.path.join(
            self.outlier_corrected_dir, f"{file_name}.{self.file_type}"
        )
        _ = find_video_of_file(
            video_dir=self.video_dir, filename=file_name, raise_error=True
        )
        plotter = SpontaneousAlternationsPlotter(
            config_path=self.config_path,
            arm_names=self.arm_names,
            center_name=self.center_name,
            buffer=self.buffer,
            verbose=self.verbose,
            data_path=data_path,
        )
        plotter.run()


# _ = SpontaneousAlternationPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/spontenous_alternation/project_folder/project_config.ini')

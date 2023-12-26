import os
from tkinter import *

from simba.data_processors.directing_other_animals_calculator import \
    DirectingOtherAnimalsAnalyzer
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon
from simba.utils.enums import Keys, Links
from simba.utils.errors import (AnimalNumberError, InvalidInputError,
                                NoFilesFoundError)


class AnimalDirectingAnimalPopUp(ConfigReader, PopUpMixin):
    def __init__(self, config_path: str):
        ConfigReader.__init__(self, config_path=config_path)
        if self.animal_cnt < 2:
            raise AnimalNumberError(
                msg=f"Directionality between animals require at least two animals. The SimBA project is set to use {self.animal_cnt} animal.",
                source=self.__class__.__name__,
            )
        if len(self.outlier_corrected_paths) == 0:
            raise NoFilesFoundError(
                msg=f"No data files found in {self.outlier_corrected_dir}.",
                source=self.__class__.__name__,
            )
        PopUpMixin.__init__(
            self, title="ANALYZE DIRECTIONALITY BETWEEN ANIMALS", size=(400, 400)
        )
        self.settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="OUTPUT FORMATS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.DIRECTING_ANIMALS_PLOTS.value,
        )
        self.boolean_tables_var = BooleanVar(value=False)
        self.summary_tables_var = BooleanVar(value=False)
        self.aggregate_statistics_var = BooleanVar(value=False)
        boolean_tables_cb = Checkbutton(
            self.settings_frm,
            text="CREATE BOOLEAN TABLES",
            variable=self.boolean_tables_var,
        )
        summary_table_cb = Checkbutton(
            self.settings_frm,
            text="CREATE DETAILED SUMMARY TABLES (INCLUDING COORDINATES)",
            variable=self.summary_tables_var,
        )
        aggregate_statistics_cb = Checkbutton(
            self.settings_frm,
            text="CREATE AGGREGATE STATISTICS TABLE",
            variable=self.aggregate_statistics_var,
        )

        self.settings_frm.grid(row=0, column=0, sticky="NW")
        boolean_tables_cb.grid(row=0, column=0, sticky="NW")
        summary_table_cb.grid(row=1, column=0, sticky="NW")
        aggregate_statistics_cb.grid(row=2, column=0, sticky="NW")
        self.create_run_frm(run_function=self.run)
        # self.main_frm.mainloop()

    def run(self):
        if (
            not self.boolean_tables_var.get()
            and not self.summary_tables_var.get()
            and not self.aggregate_statistics_var.get()
        ):
            raise InvalidInputError(
                "Please select at least one output format.",
                source=self.__class__.__name__,
            )

        directing_animals_analyzer = DirectingOtherAnimalsAnalyzer(
            config_path=self.config_path,
            bool_tables=self.boolean_tables_var.get(),
            summary_tables=self.summary_tables_var.get(),
            aggregate_statistics_tables=self.aggregate_statistics_var.get(),
        )
        directing_animals_analyzer.run()


# test = AnimalDirectingAnimalPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')

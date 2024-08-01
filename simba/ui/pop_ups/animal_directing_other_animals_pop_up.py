import os
from tkinter import *
from typing import Union

from simba.data_processors.directing_other_animals_calculator import \
    DirectingOtherAnimalsAnalyzer
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, SimbaCheckbox
from simba.utils.enums import Formats, Keys, Links
from simba.utils.errors import (AnimalNumberError, InvalidInputError,
                                NoFilesFoundError)


class AnimalDirectingAnimalPopUp(ConfigReader, PopUpMixin):

    """
    :example:
    >>> test = AnimalDirectingAnimalPopUp(config_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini")
    """
    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if self.animal_cnt < 2:
            raise AnimalNumberError(msg=f"Directionality between animals require at least two animals. The SimBA project is set to use {self.animal_cnt} animal.", source=self.__class__.__name__,)
        if len(self.outlier_corrected_paths) == 0:
            raise NoFilesFoundError(msg=f"No data files found in {self.outlier_corrected_dir}.", source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="ANALYZE DIRECTIONALITY BETWEEN ANIMALS", size=(400, 400))
        self.settings_frm = CreateLabelFrameWithIcon( parent=self.main_frm, header="OUTPUT FORMATS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DIRECTING_ANIMALS_PLOTS.value)

        boolean_tables_cb, self.boolean_tables_var = SimbaCheckbox(parent=self.settings_frm, txt="CREATE BOOLEAN TABLES", txt_img='table')
        summary_table_cb, self.summary_tables_var = SimbaCheckbox(parent=self.settings_frm, txt="CREATE DETAILED SUMMARY TABLES (INCLUDING COORDINATES)", txt_img='table')
        aggregate_statistics_cb, self.aggregate_statistics_var = SimbaCheckbox(parent=self.settings_frm, txt="CREATE AGGREGATE STATISTICS TABLE", txt_img='table')

        self.settings_frm.grid(row=0, column=0, sticky="NW")
        boolean_tables_cb.grid(row=0, column=0, sticky="NW")
        summary_table_cb.grid(row=1, column=0, sticky="NW")
        aggregate_statistics_cb.grid(row=2, column=0, sticky="NW")
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

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


#test = AnimalDirectingAnimalPopUp(config_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini")

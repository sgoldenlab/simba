import os
from typing import Union

from simba.data_processors.directing_other_animals_calculator import \
    DirectingOtherAnimalsAnalyzer
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon,
                                        SimbaCheckbox, SimBADropDown)
from simba.utils.enums import Keys, Links
from simba.utils.errors import (AnimalNumberError, CountError,
                                InvalidInputError, NoFilesFoundError)
from simba.utils.lookups import find_closest_string
from simba.utils.warnings import SkippingRuleWarning

NOSE, EAR_LEFT, EAR_RIGHT = Keys.NOSE.value, Keys.EAR_LEFT.value, Keys.EAR_RIGHT.value

class AnimalDirectingAnimalPopUp(ConfigReader, PopUpMixin):

    """
    :example:
    >>> test = AnimalDirectingAnimalPopUp(config_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini")
    """
    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if self.animal_cnt < 2:
            raise AnimalNumberError(msg=f"Directionality between animals require at least two animals. The SimBA project is set to use {self.animal_cnt} animal.", source=self.__class__.__name__,)
        if len(self.outlier_corrected_paths) == 0:
            raise NoFilesFoundError(msg=f"No data files found in {self.outlier_corrected_dir}.", source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="ANALYZE DIRECTIONALITY BETWEEN ANIMALS", size=(600, 500), icon='direction')

        bp_names = list(set([x[:-2] for x in self.body_parts_lst]))
        nose_guess = find_closest_string(target=NOSE, string_list=bp_names)[0]
        ear_left_guess = find_closest_string(target=EAR_LEFT, string_list=bp_names)[0]
        ear_right_guess = find_closest_string(target=EAR_RIGHT, string_list=bp_names)[0]

        self.bp_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT BODY-PARTS", icon_name='pose', icon_link=Links.DIRECTING_ANIMALS_PLOTS.value)
        self.ear_left_dropdown = SimBADropDown(parent=self.bp_frm, dropdown_options=bp_names, label_width=30, dropdown_width=25, value=ear_left_guess, label='LEFT EAR BODY-PART NAME:', img='left_ear')
        self.ear_right_dropdown = SimBADropDown(parent=self.bp_frm, dropdown_options=bp_names, label_width=30, dropdown_width=25, value=ear_right_guess, label='RIGHT EAR BODY-PART NAME:', img='ear_right')
        self.nose_dropdown = SimBADropDown(parent=self.bp_frm, dropdown_options=bp_names, label_width=30, dropdown_width=25, value=nose_guess, label='NOSE BODY-PART NAME:', img='nose')

        self.settings_frm = CreateLabelFrameWithIcon( parent=self.main_frm, header="OUTPUT FORMATS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DIRECTING_ANIMALS_PLOTS.value)
        boolean_tables_cb, self.boolean_tables_var = SimbaCheckbox(parent=self.settings_frm, txt="CREATE BOOLEAN TABLES", txt_img='table', val=False)
        summary_table_cb, self.summary_tables_var = SimbaCheckbox(parent=self.settings_frm, txt="CREATE DETAILED SUMMARY TABLES (INCLUDING COORDINATES)", txt_img='table', val=True)
        aggregate_statistics_cb, self.aggregate_statistics_var = SimbaCheckbox(parent=self.settings_frm, txt="CREATE AGGREGATE STATISTICS TABLE", txt_img='table', val=True)
        append_cb, self.append_var = SimbaCheckbox(parent=self.settings_frm, txt="APPEND BOOLEAN TABLES TO FEATURES", txt_img='table', val=False)

        self.bp_frm.grid(row=0, column=0, sticky="NW")
        self.ear_left_dropdown.grid(row=0, column=0, sticky="NW")
        self.ear_right_dropdown.grid(row=1, column=0, sticky="NW")
        self.nose_dropdown.grid(row=2, column=0, sticky="NW")

        self.settings_frm.grid(row=1, column=0, sticky="NW")
        boolean_tables_cb.grid(row=0, column=0, sticky="NW")
        summary_table_cb.grid(row=1, column=0, sticky="NW")
        aggregate_statistics_cb.grid(row=2, column=0, sticky="NW")
        append_cb.grid(row=3, column=0, sticky="NW")
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        if (not self.boolean_tables_var.get() and not self.summary_tables_var.get() and not self.aggregate_statistics_var.get()):
            raise InvalidInputError("Boolean tables, summary tables, and aggregate statistics options are all UNCHECKED. Please select at least one of these output format.", source=self.__class__.__name__,)
        append = self.append_var.get()
        if not self.boolean_tables_var.get() and self.append_var.get():
            SkippingRuleWarning(msg='To append boolean tables to features, select create boolean tables as well as the checkbox for appending boolean tables to features.')
            append = False

        nose_name = self.nose_dropdown.get_value()
        left_ear = self.ear_left_dropdown.get_value()
        ear_right = self.ear_right_dropdown.get_value()

        if len(list(set(list([nose_name, left_ear, ear_right])))) != 3:
            raise CountError(msg=f'The three chosen body-parts have to be unique: Got {nose_name, left_ear, ear_right}', source=self.__class__.__name__)


        directing_animals_analyzer = DirectingOtherAnimalsAnalyzer(config_path=self.config_path,
                                                                   bool_tables=self.boolean_tables_var.get(),
                                                                   summary_tables=self.summary_tables_var.get(),
                                                                   aggregate_statistics_tables=self.aggregate_statistics_var.get(),
                                                                   append_bool_tables_to_features=append,
                                                                   nose_name=nose_name,
                                                                   left_ear_name=left_ear,
                                                                   right_ear_name=ear_right)
        directing_animals_analyzer.run()

#_ = AnimalDirectingAnimalPopUp(config_path=r"D:\troubleshooting\maplight_ri\project_folder\project_config.ini")
#test = AnimalDirectingAnimalPopUp(config_path=r"C:\troubleshooting\two_animals_16_bp_JAG\project_folder\project_config.ini")

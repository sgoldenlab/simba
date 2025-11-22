from typing import Union
import os

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.utils.errors import InvalidInputError
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, SimbaButton, SimBALabel, SimBADropDown, FileSelect)
from simba.utils.enums import Links, Formats
from simba.utils.read_write import str_2_bool
from simba.roi_tools.import_roi_csvs import ROIDefinitionsCSVImporter


class ROIDefinitionsCSVImporterPopUp(ConfigReader, PopUpMixin):
    """
    :example:
    >>> ROIDefinitionsCSVImporterPopUp(config_path=r"C:\troubleshooting\mouse_open_field\project_folder\project_config.ini")
    """
    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        PopUpMixin.__init__(self, title="IMPORT SIMBA ROI CSV DEFINITIONS TO PROJECT", size=(720, 960), icon='data_table')
        self.append_activated = None if os.path.isfile(self.roi_coordinates_path) else 'disabled'
        self.config_path = config_path
        self.paths_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header=f"FILE PATHS", icon_name='csv_black', icon_link=Links.ROI.value)
        instruct_lbl = SimBALabel(parent=self.paths_frm, txt='Import previously exported SimBA ROI data in CSV format to \n SimBA ROI H5 format', txt_clr='black', bg_clr=None, font=Formats.FONT_REGULAR_ITALICS.value)
        self.rectangle_file_select = FileSelect(self.paths_frm, "RECTANGLE CSV PATH", title="SELECT CSV FILE", lblwidth=35, file_types=[("CSV FILE", (".csv", ".CSV"))])
        self.circle_file_select = FileSelect(self.paths_frm, "CIRCLE CSV PATH", title="SELECT CSV FILE", lblwidth=35, file_types=[("CSV FILE", (".csv", ".CSV"))])
        self.polygon_file_select = FileSelect(self.paths_frm, "POLYGON CSV PATH", title="SELECT CSV FILE", lblwidth=35, file_types=[("CSV FILE", (".csv", ".CSV"))])

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header=f"SETTINGS", icon_name='settings', icon_link=Links.ROI.value, pady=10)
        self.append_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label='APPEND TO EXISTING ROI DATA: ', label_width=35, dropdown_width=30, value='FALSE', state=self.append_activated)

        self.run_btn = SimbaButton(parent=self.main_frm, txt='RUN', img='rocket', txt_clr='blue', font=Formats.FONT_LARGE.value, hover_font=Formats.FONT_LARGE_BOLD.value, cmd=self.run)
        self.paths_frm.grid(row=0, column=0, sticky='NW')
        instruct_lbl.grid(row=0, column=0, sticky='NW')
        self.rectangle_file_select.grid(row=1, column=0, sticky='NW')
        self.circle_file_select.grid(row=2, column=0, sticky='NW')
        self.polygon_file_select.grid(row=3, column=0, sticky='NW')

        self.settings_frm.grid(row=1, column=0, sticky='NW')
        self.append_dropdown.grid(row=0, column=0, sticky='NW')
        self.run_btn.grid(row=2, column=0, sticky='NW')

        self.main_frm.mainloop()

    def run(self):
        rectangle_path = None if not os.path.isfile(self.rectangle_file_select.file_path) else self.rectangle_file_select.file_path
        circle_path = None if not os.path.isfile(self.circle_file_select.file_path) else self.circle_file_select.file_path
        polygon_path = None if not os.path.isfile(self.polygon_file_select.file_path) else self.polygon_file_select.file_path
        append = str_2_bool(self.append_dropdown.get_value())

        if rectangle_path is None and circle_path is None and polygon_path is None:
            raise InvalidInputError(msg='Please pass at path to rectangles, circles, and/or polygon CSVs. They are all not defined. Define at least ONE path.', source=self.__class__.__name__)

        importer = ROIDefinitionsCSVImporter(config_path=self.config_path,
                                             rectangles_path=rectangle_path,
                                             circles_path=circle_path,
                                             polygon_path=polygon_path,
                                             append=append)
        importer.run()

#ROIDefinitionsCSVImporterPopUp(config_path=r"C:\troubleshooting\mouse_open_field\project_folder\project_config.ini")
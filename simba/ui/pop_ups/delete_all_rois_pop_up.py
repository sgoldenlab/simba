import os
from typing import Union

from simba.ui.tkinter_functions import TwoOptionQuestionPopUp
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.enums import ConfigKey, Links, Paths
from simba.utils.errors import NoROIDataError
from simba.utils.printing import stdout_trash
from simba.utils.read_write import read_config_file, remove_files


def delete_all_rois_pop_up(config_path: Union[str, os.PathLike]) -> None:
    """
    Launches a pop-up asking if to delete all SimBA roi definitions. If click yes, then the ``/project_folder/logs/measures\ROI_definitions.h5`` of the SimBA project is deleted.

    :param config_path: Path to SimBA project config file.
    :return: None

    :example:
    >>> delete_all_rois_pop_up(config_path=r"C:\troubleshooting\ROI_movement_test\project_folder\project_config.ini")
    """

    question = TwoOptionQuestionPopUp(title="WARNING!", question="Do you want to delete all defined ROIs in the project?", option_one="YES", option_two="NO", link=Links.ROI.value)
    if question.selected_option == "YES":
        check_file_exist_and_readable(file_path=config_path)
        config = read_config_file(config_path=config_path)
        project_path = config.get(ConfigKey.GENERAL_SETTINGS.value, ConfigKey.PROJECT_PATH.value)
        roi_coordinates_path = os.path.join(project_path, "logs", Paths.ROI_DEFINITIONS.value)
        if not os.path.isfile(roi_coordinates_path):
            raise NoROIDataError(msg=f"Cannot delete ROI definitions: no ROI definitions exist in SimBA project. Could find find a file at expected location {roi_coordinates_path}. Create ROIs before deleting ROIs.", source=reset_video_ROIs.__name__)
        else:
            remove_files(file_paths=[roi_coordinates_path], raise_error=True)
            stdout_trash(msg=f"Deleted all ROI records for video for the SimBA project (Deleted file {roi_coordinates_path}). USe the Define ROIs menu to create new ROIs.")
    else:
        pass
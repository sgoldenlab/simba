
from tkinter import *
import os
from configparser import ConfigParser
import pandas as pd
from simba.drop_bp_cords import get_fn_ext
from simba.enums import ReadConfig, Keys, Paths
from simba.read_config_unit_tests import read_config_file
from simba.utils.errors import NoROIDataError
from simba.enums import Formats

def reset_video_ROIs(config_path, filename):

    _, file_name_wo_ext, VideoExtension = get_fn_ext(filename)
    config = ConfigParser()
    configFile = str(config_path)
    config.read(configFile)
    vidInfPath = config.get(ReadConfig.GENERAL_SETTINGS.value, ReadConfig.PROJECT_PATH.value)
    logFolderPath = os.path.join(vidInfPath, 'logs')
    ROIcoordinatesPath = os.path.join(logFolderPath, Paths.ROI_DEFINITIONS.value)
    if not os.path.isfile(ROIcoordinatesPath):
        raise NoROIDataError(msg='Cannot delete ROI definitions: no definitions exist to delete')

    else:
        rectanglesInfo = pd.read_hdf(ROIcoordinatesPath, key=Keys.ROI_RECTANGLES.value)
        circleInfo = pd.read_hdf(ROIcoordinatesPath, key=Keys.ROI_CIRCLES.value)
        polygonInfo = pd.read_hdf(ROIcoordinatesPath, key=Keys.ROI_POLYGONS.value)
        store = pd.HDFStore(ROIcoordinatesPath, mode='w')

    try:
        rectanglesInfo = rectanglesInfo[rectanglesInfo['Video'] != file_name_wo_ext]
    except KeyError:
        pass
    store['rectangles'] = rectanglesInfo

    try:
        circleInfo = circleInfo[circleInfo['Video'] != file_name_wo_ext]
    except KeyError:
        pass
    store['circleDf'] = circleInfo

    try:
        polygonInfo = polygonInfo[polygonInfo['Video'] != file_name_wo_ext]
    except KeyError:
        pass
    store['polygons'] = polygonInfo

    print('Deleted ROI record: ' + str(file_name_wo_ext))
    store.close()

def delete_all_ROIs(config_path: str):

    def delete_file(config_path):
        config = read_config_file(ini_path=config_path)
        project_path = config.get(ReadConfig.GENERAL_SETTINGS.value, ReadConfig.PROJECT_PATH.value)
        roi_data_path = os.path.join(project_path, 'logs', Paths.ROI_DEFINITIONS.value)

        if not os.path.isfile(roi_data_path):
            raise NoROIDataError(msg=f'No ROI definitions exist in this SimBA project. Expected file at path {roi_data_path}')
        else:
            os.remove(roi_data_path)
            close_window()
            print('SIMBA COMPLETE: All ROI definitions deleted in this SimBA project')

    def close_window():
        delete_confirm_win.destroy()
        delete_confirm_win.update()

    delete_confirm_win = Toplevel()
    delete_confirm_win.minsize(200, 200)

    question_frame = LabelFrame(delete_confirm_win, text='Confirm', font=("Arial", 16, "bold"), padx=5, pady=5)
    question_lbl = Label(question_frame, text="Do you want to delete all defined ROIs in the project?", font=Formats.LABELFRAME_HEADER_FORMAT.value)

    yes_button = Button(question_frame, text='YES', fg='black', command=lambda: delete_file(config_path))
    no_button = Button(question_frame, text='NO', fg='black', command=lambda: close_window())

    question_frame.grid(row=0, sticky=W)
    question_lbl.grid(row=1, column=0, sticky=W, pady=10, padx=10)
    yes_button.grid(row=2, column=1, sticky=W, pady=10, padx=10)
    no_button.grid(row=2, column=2, sticky=W, pady=10, padx=10)

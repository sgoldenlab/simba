from tkinter import *

from simba.ui.tkinter_functions import FileSelect, Entry_Box, CreateLabelFrameWithIcon, DropDownMenu
from simba.utils.lookups import get_color_dict
from simba.utils.enums import Keys, Links
from simba.plotting.ez_lineplot import DrawPathPlot
from simba.mixins.pop_up_mixin import PopUpMixin

class MakePathPlotPopUp(PopUpMixin):
    def __init__(self):

        PopUpMixin.__init__(self, title="CREATE PATH PLOT", size=(200, 200))
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        video_path = FileSelect(settings_frm, 'VIDEO PATH: ', lblwidth='30')
        body_part = Entry_Box(settings_frm, 'BODY PART: ', '30')
        data_path = FileSelect(settings_frm, 'DATA PATH (e.g., H5 or CSV file): ', lblwidth='30')
        color_lst = list(get_color_dict().keys())
        background_color = DropDownMenu(settings_frm,'BACKGROUND COLOR: ',color_lst,'18')
        background_color.setChoices(choice='White')
        line_color = DropDownMenu(settings_frm, 'LINE COLOR: ', color_lst, '18')
        line_color.setChoices(choice='Red')
        line_thickness = DropDownMenu(settings_frm, 'LINE THICKNESS: ', list(range(1, 11)), '18')
        line_thickness.setChoices(choice=1)
        circle_size = DropDownMenu(settings_frm, 'CIRCLE SIZE: ', list(range(1, 11)), '18')
        circle_size.setChoices(choice=5)
        run_btn = Button(settings_frm,text='CREATE PATH PLOT VIDEO',command = lambda: DrawPathPlot(data_path=data_path.file_path,
                                                                                                                video_path=video_path.file_path,
                                                                                                                body_part=body_part.entry_get,
                                                                                                                bg_color=background_color.getChoices(),
                                                                                                                line_color=line_color.getChoices(),
                                                                                                                line_thinkness=line_thickness.getChoices(),
                                                                                                                circle_size=circle_size.getChoices()))
        settings_frm.grid(row=0,sticky=W)
        video_path.grid(row=0,sticky=W)
        data_path.grid(row=1,sticky=W)
        body_part.grid(row=2,sticky=W)
        background_color.grid(row=3,sticky=W)
        line_color.grid(row=4, sticky=W)
        line_thickness.grid(row=5, sticky=W)
        circle_size.grid(row=6, sticky=W)
        run_btn.grid(row=7,pady=10)
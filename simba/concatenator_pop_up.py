from simba.read_config_unit_tests import read_config_file, read_config_entry
from tkinter import *
import os, glob
from simba.tkinter_functions import DropDownMenu, FileSelect
from simba.misc_tools import get_fn_ext, find_video_of_file, get_video_meta_data
from frame_mergerer_ffmpeg import FrameMergererFFmpeg
import simba

class ConcatenatorPopUp(object):
    def __init__(self,
                 config_path: str):
        self.config, self.config_path = read_config_file(ini_path=config_path), config_path
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.icons_path = os.path.join(os.path.dirname(simba.__file__), 'assets', 'icons')
        self.icons_path = '/Users/simon/Desktop/simbapypi_dev/simba/assets/icons'
        self.main_frm = Toplevel()
        self.main_frm.minsize(500, 800)
        self.main_frm.wm_title('MERGE (CONCATENATE) VIDEOS')
        self.select_video_cnt_frm = LabelFrame(self.main_frm, text='VIDEOS #', pady=5, padx=5, font=("Helvetica", 12, 'bold'), fg='black')
        self.select_video_cnt_dropdown = DropDownMenu(self.select_video_cnt_frm, 'VIDEOS #', list(range(2,21)), '15')
        self.select_video_cnt_dropdown.setChoices(2)
        self.select_video_cnt_btn = Button(self.select_video_cnt_frm, text='SELECT', command=lambda: self.populate_table())
        self.select_video_cnt_frm.grid(row=0, column=0, sticky=NW)
        self.select_video_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        self.select_video_cnt_btn.grid(row=0, column=1, sticky=NW)



    def populate_table(self):
        if hasattr(self, 'video_table_frm'):
            self.video_table_frm.destroy()
            self.join_type_frm.destroy()
        self.video_table_frm = LabelFrame(self.main_frm, text='VIDEO PATHS', pady=5, padx=5, font=("Helvetica", 12, 'bold'), fg='black')
        self.video_table_frm.grid(row=1, sticky=NW)
        self.join_type_frm = LabelFrame(self.main_frm, text='JOIN TYPE', pady=5, padx=5, font=("Helvetica", 12, 'bold'),fg='black')
        self.join_type_frm.grid(row=2, sticky=NW)
        self.videos_dict = {}
        for cnt in range(int(self.select_video_cnt_dropdown.getChoices())):
            self.videos_dict[cnt] = FileSelect(self.video_table_frm, "Video {}: ".format(str(cnt+1)), title='Select a video file')
            self.videos_dict[cnt].grid(row=cnt, column=0, sticky=NW)

        self.join_type_var = StringVar()
        self.icons_dict = {}
        for file_cnt, file_path in enumerate(glob.glob(self.icons_path + '/*')):
            _, file_name, _ = get_fn_ext(file_path)
            self.icons_dict[file_name] = {}
            self.icons_dict[file_name]['img'] = PhotoImage(file=file_path)
            self.icons_dict[file_name]['btn'] = Radiobutton(self.join_type_frm, text=file_name, variable=self.join_type_var, value=file_name)
            self.icons_dict[file_name]['btn'].config(image=self.icons_dict[file_name]['img'])
            self.icons_dict[file_name]['btn'].image = self.icons_dict[file_name]['img']
            self.icons_dict[file_name]['btn'].grid(row=0, column=file_cnt, sticky=NW)
        self.join_type_var.set(value='Mosaic')

        self.resolution_frm = LabelFrame(self.main_frm, text='RESOLUTION', pady=5, padx=5, font=("Helvetica", 12, 'bold'), fg='black')
        self.resolution_width = DropDownMenu(self.resolution_frm, 'Resolution width', ['480', '640', '1280', '1920', '2560'], '15')
        self.resolution_width.setChoices('640')
        self.resolution_height = DropDownMenu(self.resolution_frm, 'Resolution height', ['480', '640', '1280', '1920', '2560'], '15')
        self.resolution_height.setChoices('480')
        self.resolution_frm.grid(row=3, column=0, sticky=NW)
        self.resolution_width.grid(row=0, column=0, sticky=NW)
        self.resolution_height.grid(row=1, column=0, sticky=NW)

        run_btn = Button(self.main_frm, text='RUN', command=lambda: self.run())
        run_btn.grid(row=4, column=0, sticky=NW)

    def run(self):
        videos_info = {}
        for cnt, (video_name, video_data) in enumerate(self.videos_dict.items()):
            _ = get_video_meta_data(video_path=video_data.file_path)
            videos_info['Video {}'.format(str(cnt+1))] = video_data.file_path

        if (len(videos_info.keys()) < 3) & (self.join_type_var.get() == 'mixed_mosaic'):
            print('SIMBA ERROR: if using the mixed mosaic join type, please tick check-boxes for at leasr three video types.')
            raise ValueError()

        print(videos_info)

        _ = FrameMergererFFmpeg(config_path=self.config_path,
                                frame_types=videos_info,
                                video_height=int(self.resolution_height.getChoices()),
                                video_width=int(self.resolution_width.getChoices()),
                                concat_type=self.join_type_var.get())

test = ConcatenatorPopUp(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini')
test.main_frm.mainloop()
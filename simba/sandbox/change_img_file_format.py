
try:
    from typing import Literal
except:
    from typing_extensions import Literal
from typing import Union, Optional
import os
from tkinter import *
from datetime import datetime
from PIL import Image
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, DropDownMenu, FolderSelect


from simba.utils.checks import check_if_dir_exists, check_str, check_int
from simba.utils.enums import Options, Keys, Links
from simba.utils.read_write import find_files_of_filetypes_in_directory, get_fn_ext
from simba.utils.printing import SimbaTimer, stdout_success
from simba.video_processors.video_processing import convert_to_jpeg



# class Convert2jpegPopUp(PopUpMixin):
#     def __init__(self):
#         super().__init__(title="CONVERT IMAGE DIRECTORY TO JPEG")
#         settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
#         self.selected_frame_dir = FolderSelect(settings_frm, "IMAGE DIRECTORY PATH:", title="Select a image directory", lblwidth=25)
#         self.quality_scale = Scale(settings_frm, from_=1, to=100, orient=HORIZONTAL, length=200, label='JPEG QUALITY', fg='blue')
#         self.quality_scale.set(95)
#         run_btn = Button(settings_frm, text="RUN JPEG CONVERSION", command=lambda: self.run())
#         settings_frm.grid(row=0, column=0, sticky="NW")
#         self.selected_frame_dir.grid(row=0, column=0, sticky="NW")
#         self.quality_scale.grid(row=1, column=0, sticky="NW")
#         run_btn.grid(row=2, column=0, sticky="NW")
#         self.main_frm.mainloop()
#
#     def run(self):
#         folder_path = self.selected_frame_dir.folder_path
#         check_if_dir_exists(in_dir=folder_path)
#         _ = convert_to_jpeg(directory=folder_path, quality=int(self.quality_scale.get()), verbose=True)


# class Convert2bmpPopUp(PopUpMixin):
#     def __init__(self):
#         super().__init__(title="CONVERT IMAGE DIRECTORY TO BMP")
#         settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
#         self.selected_frame_dir = FolderSelect(settings_frm, "IMAGE DIRECTORY PATH:", title="Select a image directory", lblwidth=25)
#         self.bits_dropdown = DropDownMenu(settings_frm, "BMP BITS:", [1, 4, 8, 24], labelwidth=25)
#         self.bits_dropdown.setChoices(24)
#         self.compression_drop_down = DropDownMenu(settings_frm, "COMPRESSION LEVEL:", list(range(0, 10)), labelwidth=25)
#         self.compression_drop_down.setChoices(0)


def convert_to_png(directory: Union[str, os.PathLike],
                   verbose: Optional[bool] = False) -> None:

    timer = SimbaTimer(start=True)
    check_if_dir_exists(in_dir=directory, source=convert_to_png.__name__)
    file_paths = find_files_of_filetypes_in_directory(directory=directory, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, raise_error=True)
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"{len(file_paths)} image file(s) found in {directory}...")
    save_dir = os.path.join(directory, f'png_{datetime_}')
    os.makedirs(save_dir)
    for file_cnt, file_path in enumerate(file_paths):
        dir, file_name, _ = get_fn_ext(filepath=file_path)
        save_path = os.path.join(save_dir, f'{file_name}.png')
        if verbose:
            print(f"Converting file {file_cnt+1}/{len(file_paths)} ...")
        img = Image.open(file_path)
        if img.mode in ('RGBA', 'LA'): img = img.convert('RGB')
        img.save(save_path, 'PNG')
        timer.stop_timer()
    stdout_success(msg=f"SIMBA COMPLETE: {len(file_paths)} image file(s) in {directory} directory converted to PNG and stored in {save_dir} directory", source=convert_to_png.__name__, elapsed_time=timer.elapsed_time_str)


convert_to_png(directory='/Users/simon/Desktop/imgs')

# def convert_to_jpeg(directory: Union[str, os.PathLike],
#                     file_type_in: Literal['.bmp', '.jpg', '.jpeg', '.png'],
#                     quality: Optional[int] = 95,
#                     verbose: Optional[bool] = False) -> None:
#
#     """
#     Convert the file type of all image files within a directory to jpeg format of the passed quality.
#
#     .. note::
#        Quality above 95 should be avoided; 100 disables portions of the JPEG compression algorithm, and results in large files with hardly any gain in image quality
#
#     :parameter Union[str, os.PathLike] directory: Path to directory holding image files
#     :parameter str file_type_in: Input file type, e.g., 'bmp' or 'png.
#     :parameter str file_type_out: Output file type, e.g., 'bmp' or 'png.
#     :parameter Optional[bool] verbose: If True, prints progress. Default False.
#
#     :example:
#     >>> convert_to_jpeg(directory='/Users/simon/Desktop/imgs', file_type_in='.png', quality=15)
#     """
#     timer = SimbaTimer(start=True)
#     check_if_dir_exists(in_dir=directory, source=convert_to_jpeg.__name__)
#     check_str(name=f'{convert_to_jpeg.__name__} file_type_in', value=file_type_in, options=Options.ALL_IMAGE_FORMAT_OPTIONS.value)
#     check_int(name=f'{convert_to_jpeg.__name__} quality', value=quality, min_value=1, max_value=100)
#     file_paths = find_files_of_filetypes_in_directory(directory=directory, extensions=[file_type_in], raise_error=True)
#     datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
#     print(f"{len(file_paths)} {file_type_in} image file(s) found in {directory}...")
#     save_dir = os.path.join(directory, f'png_{datetime_}')
#     os.makedirs(save_dir)
#     for file_cnt, file_path in enumerate(file_paths):
#         dir, file_name, _ = get_fn_ext(filepath=file_path)
#         save_path = os.path.join(save_dir, f'{file_name}.jpeg')
#         if verbose:
#             print(f"Converting file {file_cnt+1}/{len(file_paths)} ...")
#         img = Image.open(file_path)
#         if img.mode in ('RGBA', 'LA'): img = img.convert('RGB')
#         img.save(save_path, 'JPEG', quality=quality)
#     timer.stop_timer()
#     stdout_success(msg=f"SIMBA COMPLETE: {len(file_paths)} {file_type_in} image files in {directory} directory converted to jpeg and stored in {save_dir} directory", source=convert_to_jpeg.__name__, elapsed_time=timer.elapsed_time_str)

#convert_to_jpeg(directory='/Users/simon/Desktop/imgs', file_type_in='.png', quality=15)
import os
import webbrowser
from tkinter import *
from typing import Union

import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageTk

import simba
from simba.ui.tkinter_functions import SimBALabel
from simba.utils.enums import OS, Formats, Links, Paths
from simba.utils.lookups import rgb_to_hex
from simba.utils.read_write import get_video_meta_data

LINKS = [
         ("GitHub", "https://github.com/sgoldenlab/simba", "github_2"),
         ("API", "https://simba-uw-tf-dev.readthedocs.io/en/latest/api.html", "documentation_large"),
         ("Gitter", "https://app.gitter.im/#/room/#sgoldenlab_simba:gitter.im", "gitter_large"),
         ("bioRxiv", "https://www.biorxiv.org/content/10.1101/2020.04.19.049452v1", "documentation_large"),
         ("Nature Neuroscience", "https://www.nature.com/articles/s41593-024-01649-9", "pdf_large"),
         ("OSF data buckets", "https://osf.io/user/mutws", "osf_large"),
         ("PyPI", "https://pypi.org/project/simba-uw-tf-dev/", "rocket")
         ]

DEVELOPER_URL = Links.SIMON_WEBSITE.value
DEVELOPER_IMG = os.path.join(os.path.dirname(simba.__file__), Paths.SIMON_SMALL_IMG.value)
VERSION_TXT = f"SimBA v{OS.SIMBA_VERSION.value}" if OS.SIMBA_VERSION.value else "SimBA"
LANDING_MOVIE_PATH = os.path.join(os.path.dirname(simba.__file__), Paths.LANDING_MOVIE.value)

# RGB color definitions
LINK_COLOR_RGB = (91, 163, 245)
LINK_HOVER_COLOR_RGB = (123, 181, 247)
BG_COLOR_RGB = (10, 10, 10)
SEPARATOR_COLOR_RGB = (42, 42, 42)
UNDERLINE_COLOR_RGB = (58, 58, 58)
BLACK_RGB = (0, 0, 0)
WHITE_RGB = (255, 255, 255)

LINK_COLOR = rgb_to_hex(LINK_COLOR_RGB)
LINK_HOVER_COLOR = rgb_to_hex(LINK_HOVER_COLOR_RGB)
BG_COLOR = rgb_to_hex(BG_COLOR_RGB)
SEPARATOR_COLOR = rgb_to_hex(SEPARATOR_COLOR_RGB)
UNDERLINE_COLOR = rgb_to_hex(UNDERLINE_COLOR_RGB)
BLACK = rgb_to_hex(BLACK_RGB)
WHITE = rgb_to_hex(WHITE_RGB)


class AboutSimBAPopUp:
    def __init__(self,
                 video_path: Union[str, os.PathLike] = LANDING_MOVIE_PATH,
                 title: str = "ABOUT SIMBA"):

        self.video_meta = get_video_meta_data(video_path=video_path, fps_as_int=False)
        video_width, video_height = int(self.video_meta['width']), int(self.video_meta['height'])
        window_height = video_height * 2

        self.root = Toplevel()
        icon_path = os.path.join(os.path.dirname(simba.__file__), "assets", "icons", "SimBA_logo_3_small.png")
        try:
            icon_img = PhotoImage(file=icon_path)
            self.root.iconphoto(False, icon_img)
            self.root.iconimage = icon_img
        except Exception:
            pass
        self.root.title(title)
        self.root.geometry(f"{video_width}x{window_height}")
        self.root.resizable(False, False)
        self.root.attributes('-topmost', True)

        self.root.grid_rowconfigure(0, minsize=video_height)
        self.root.grid_rowconfigure(1, minsize=video_height)
        self.root.grid_columnconfigure(0, weight=1)

        self.video_frame = Frame(self.root, bg=BLACK, height=video_height)
        self.video_frame.grid(row=0, column=0, sticky="nsew")
        self.video_frame.grid_propagate(False)
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.bottom_frame = Frame(self.root, bg=BG_COLOR, height=video_height)
        self.bottom_frame.grid(row=1, column=0, sticky="nsew")
        self.bottom_frame.grid_propagate(False)
        self.bottom_frame.grid_rowconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(1, weight=0)  # Separator column
        self.bottom_frame.grid_columnconfigure(2, weight=1)

        left_container = Frame(self.bottom_frame, bg=BG_COLOR)
        left_container.grid(row=0, column=0, sticky="nsew", padx=(40, 20), pady=25)
        left_container.grid_rowconfigure(0, weight=1)
        left_container.grid_columnconfigure(0, weight=1)

        separator = Frame(self.bottom_frame, bg=SEPARATOR_COLOR, width=1)
        separator.grid(row=0, column=1, sticky="ns", pady=25)

        self.right_container = Frame(self.bottom_frame, bg=BG_COLOR)
        self.right_container.grid(row=0, column=2, sticky="nsew", padx=(20, 40), pady=25)
        self.right_container.grid_rowconfigure(0, weight=1)
        self.right_container.grid_columnconfigure(0, weight=1)

        header_label = SimBALabel(parent=left_container, txt="R E S O U R C E S", txt_clr=WHITE, bg_clr=BG_COLOR,
                                  font=Formats.FONT_REGULAR.value)
        header_label.pack(pady=(0, 5))

        header_underline = Frame(left_container, bg=UNDERLINE_COLOR, height=1)
        header_underline.pack(fill=X, pady=(0, 5))

        developer_header = SimBALabel(parent=self.right_container, txt="D E V E L O P E R", txt_clr=WHITE,
                                      bg_clr=BG_COLOR, font=Formats.FONT_REGULAR.value)
        developer_header.pack(pady=(0, 5))

        dev_underline = Frame(self.right_container, bg=UNDERLINE_COLOR, height=1)
        dev_underline.pack(fill=X, pady=(0, 5))

        def add_rounded_corners_with_shadow(img, radius=10, shadow_offset=5):
            shadow = Image.new('RGBA', (img.width + shadow_offset * 2, img.height + shadow_offset * 2), (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow)

            try:
                shadow_draw.rounded_rectangle(
                    [(shadow_offset, shadow_offset), (img.width + shadow_offset, img.height + shadow_offset)],
                    radius=radius, fill=(0, 0, 0, 100))
            except AttributeError:
                width, height = img.size
                shadow_draw.rectangle(
                    [shadow_offset + radius, shadow_offset, width + shadow_offset - radius, height + shadow_offset],
                    fill=(0, 0, 0, 100))
                shadow_draw.rectangle(
                    [shadow_offset, shadow_offset + radius, width + shadow_offset, height + shadow_offset - radius],
                    fill=(0, 0, 0, 100))
                shadow_draw.ellipse(
                    [shadow_offset, shadow_offset, shadow_offset + radius * 2, shadow_offset + radius * 2],
                    fill=(0, 0, 0, 100))
                shadow_draw.ellipse([width + shadow_offset - radius * 2, shadow_offset, width + shadow_offset,
                                     shadow_offset + radius * 2], fill=(0, 0, 0, 100))
                shadow_draw.ellipse([shadow_offset, height + shadow_offset - radius * 2, shadow_offset + radius * 2,
                                     height + shadow_offset], fill=(0, 0, 0, 100))
                shadow_draw.ellipse(
                    [width + shadow_offset - radius * 2, height + shadow_offset - radius * 2, width + shadow_offset,
                     height + shadow_offset], fill=(0, 0, 0, 100))

            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=8))
            mask = Image.new('L', img.size, 0)
            mask_draw = ImageDraw.Draw(mask)
            try:
                mask_draw.rounded_rectangle([(0, 0), img.size], radius=radius, fill=255)
            except AttributeError:
                width, height = img.size
                mask_draw.rectangle([radius, 0, width - radius, height], fill=255)
                mask_draw.rectangle([0, radius, width, height - radius], fill=255)
                mask_draw.ellipse([0, 0, radius * 2, radius * 2], fill=255)
                mask_draw.ellipse([width - radius * 2, 0, width, radius * 2], fill=255)
                mask_draw.ellipse([0, height - radius * 2, radius * 2, height], fill=255)
                mask_draw.ellipse([width - radius * 2, height - radius * 2, width, height], fill=255)

            output = Image.new('RGBA', img.size, (0, 0, 0, 0))
            output.paste(img, (0, 0), mask)

            final = Image.new('RGBA', shadow.size, (0, 0, 0, 0))
            final.paste(shadow, (0, 0))
            final.paste(output, (shadow_offset, shadow_offset), output)

            return final

        dev_image_container = Frame(self.right_container, bg=BG_COLOR)
        dev_image_container.pack(pady=(0, 10))

        if os.path.exists(DEVELOPER_IMG):
            try:
                dev_image = Image.open(DEVELOPER_IMG)
                if dev_image.mode != 'RGBA':
                    dev_image = dev_image.convert('RGBA')
                max_width = 180
                aspect_ratio = dev_image.width / dev_image.height
                new_width = min(max_width, dev_image.width)
                new_height = int(new_width / aspect_ratio)
                dev_image = dev_image.resize((new_width, new_height), Image.LANCZOS)

                dev_image = add_rounded_corners_with_shadow(dev_image, radius=20, shadow_offset=20)
                dev_photo = ImageTk.PhotoImage(dev_image)

                dev_image_label = Label(dev_image_container, image=dev_photo, bg=BG_COLOR, cursor="hand2")
                dev_image_label.image = dev_photo
                dev_image_label.pack()
                dev_image_label.bind("<Button-1>", lambda e, u=DEVELOPER_URL: webbrowser.open(u))
            except Exception as r:
                print(r.args)
                pass

        developer_link_frame = Frame(self.right_container, bg=BG_COLOR, cursor="hand2")
        developer_link_frame.pack(pady=(0, 0))
        developer_link = SimBALabel(parent=developer_link_frame, txt='SIMON NILSSON', txt_clr=LINK_COLOR, font=Formats.FONT_HEADER.value, bg_clr=BG_COLOR, hover_fg_clr=WHITE, hover_font=Formats.FONT_LARGE_BOLD.value, cursor="hand2")
        developer_link.pack()

        def dev_on_enter(e):
            developer_link.config(fg=LINK_HOVER_COLOR)
            if not hasattr(developer_link, 'underline_widget'):
                underline = Frame(developer_link_frame, bg=LINK_HOVER_COLOR, height=2)
                underline.pack(fill=X, pady=(3, 0))
                developer_link.underline_widget = underline

        def dev_on_leave(e):
            developer_link.config(fg=LINK_COLOR)
            if hasattr(developer_link, 'underline_widget'):
                developer_link.underline_widget.destroy()
                del developer_link.underline_widget

        developer_link.bind("<Button-1>", lambda e, u=DEVELOPER_URL: webbrowser.open(u))
        developer_link.bind("<Enter>", dev_on_enter)
        developer_link.bind("<Leave>", dev_on_leave)
        developer_link_frame.bind("<Button-1>", lambda e, u=DEVELOPER_URL: webbrowser.open(u))
        developer_link_frame.bind("<Enter>", dev_on_enter)
        developer_link_frame.bind("<Leave>", dev_on_leave)
        for widget in developer_link_frame.winfo_children():
            widget.bind("<Button-1>", lambda e, u=DEVELOPER_URL: webbrowser.open(u))
            widget.bind("<Enter>", dev_on_enter)
            widget.bind("<Leave>", dev_on_leave)

        links_container = Frame(left_container, bg=BG_COLOR)
        links_container.pack(fill=BOTH, expand=True)

        for i, (text, url, icon) in enumerate(LINKS):
            link_frame = Frame(links_container, bg=BG_COLOR, cursor="hand2")
            link_frame.pack(pady=2)

            icon_label = SimBALabel(parent=link_frame, txt='', txt_clr=LINK_COLOR, bg_clr=BG_COLOR, font=Formats.FONT_REGULAR.value, hover_font=Formats.FONT_LARGE_ITALICS.value, cursor="hand2", link=url, img=icon, compound='left')
            icon_label.pack(side=LEFT, padx=(0, 3))

            link_label = SimBALabel(parent=link_frame, txt=text, txt_clr=LINK_COLOR, font=Formats.FONT_REGULAR.value, bg_clr=BG_COLOR, hover_font=Formats.FONT_LARGE_ITALICS.value, cursor="hand2", link=url)
            link_label.pack(side=LEFT)

        self.video_canvas = Canvas(self.video_frame, bg=BLACK, highlightthickness=0)
        self.video_canvas.grid(row=0, column=0, sticky="nsew")

        version_label = SimBALabel(parent=self.video_frame, txt=VERSION_TXT, txt_clr=WHITE, bg_clr=BLACK,
                                   font=Formats.FONT_HEADER.value)
        version_label.place(relx=0.5, rely=0.90, anchor="center")

        self.cap = cv2.VideoCapture(video_path)
        self.frame_delay = int(1000 / self.video_meta['fps'])
        self.video_frame_width, self.video_frame_height = video_width, video_height

        self.root.bind("<Escape>", lambda e: self.on_closing())
        self.root.focus_set()

        self.root.update_idletasks()
        self.play_video()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def play_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_canvas.update_idletasks()
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()

            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = self.video_frame_width, self.video_frame_height

            frame_height, frame_width = frame_rgb.shape[:2]
            aspect_ratio = frame_width / frame_height

            if canvas_width / canvas_height > aspect_ratio:
                new_height = canvas_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = canvas_width
                new_height = int(new_width / aspect_ratio)

            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))

            pil_image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=pil_image)

            self.video_canvas.delete("all")
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            self.video_canvas.create_image(x, y, anchor=NW, image=photo)
            self.video_canvas.image = photo
            self.root.after(self.frame_delay, self.play_video)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.root.after(self.frame_delay, self.play_video)

    def on_closing(self, event=None):
        if hasattr(self, 'cap'):
            self.cap.release()
        self.root.destroy()

#
# if __name__ == "__main__":
#     root = Tk()
#     root.withdraw()
#     AboutSimBAPopUp()
#     root.mainloop()
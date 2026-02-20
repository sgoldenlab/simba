try:
    from typing import Literal
except:
    from typing_extensions import Literal

from tkinter import Toplevel

from simba.utils.checks import check_instance, check_int, check_str
from simba.utils.lookups import get_monitor_info


def position_window(window: Toplevel,
                    position: Literal["top_left", "top_right", "bottom_left", "bottom_right", "middle_top", "middle_bottom", "center"],
                    offset_x: int = 0,
                    offset_y: int = 0) -> None:
    """
    Place the window at a screen position. Placement is deferred (via after(0, ...)) so the window
    is laid out first; call right after building the window.

    :param win: A tk.Toplevel or tk.Tk window.
    :param position: One of "top_left", "top_right", "bottom_left", "bottom_right", "middle_top", "middle_bottom", "center".
    :param offset_x: Pixels from the position horizontally (positive = right for corners, right for center/middle).
    :param offset_y: Pixels from the position vertically (positive = down for top, up for bottom, down for center).
    """
    valid = ("top_left", "top_right", "bottom_left", "bottom_right", "middle_top", "middle_bottom", "center")
    check_instance(source=f'{position_window.__name__} window', instance=window, accepted_types=(Toplevel,), raise_error=True)
    check_int(name=f'{position_window.__name__} offset_x', value=offset_x, raise_error=True)
    check_int(name=f'{position_window.__name__} offset_y', value=offset_y, raise_error=True)
    check_str(name=f'{position_window.__name__} position', value=position, options=valid, raise_error=True)

    def do_place():
        window.update_idletasks()
        w = window.winfo_width()
        h = window.winfo_height()
        if w <= 1 or h <= 1:
            w = max(w, window.winfo_reqwidth())
            h = max(h, window.winfo_reqheight())
        _, (sw, sh) = get_monitor_info()
        if position == "top_left":
            x, y = offset_x, offset_y
        elif position == "top_right":
            x, y = sw - w - offset_x, offset_y
        elif position == "bottom_left":
            x, y = offset_x, sh - h - offset_y
        elif position == "bottom_right":
            x, y = sw - w - offset_x, sh - h - offset_y
        elif position == "middle_top":
            x, y = (sw - w) // 2 + offset_x, offset_y
        elif position == "middle_bottom":
            x, y = (sw - w) // 2 + offset_x, sh - h - offset_y
        else:  # center
            x, y = (sw - w) // 2 + offset_x, (sh - h) // 2 + offset_y
        window.geometry(f"+{x}+{y}")

    window.after(0, do_place)
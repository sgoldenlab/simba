try:
    from typing import Literal
except:
    from typing_extensions import Literal

from tkinter import Tk, Toplevel
from typing import Union

from simba.utils.checks import check_instance, check_int, check_str
from simba.utils.lookups import get_monitor_info

VALID_POSITIONS = ("top_left", "top_right", "bottom_left", "bottom_right", "middle_top", "middle_bottom", "middle_left", "middle_right", "center",)


def position_window(window: Union[Toplevel, Tk],
                    position: Literal["top_left", "top_right", "bottom_left", "bottom_right", "middle_top", "middle_bottom", "middle_left", "middle_right", "center" ],
                    offset_x: int = 0,
                    offset_y: int = 0,
                    lift: bool = True,
                    focus: bool = False,
                    constrain: bool = True,
                    verbose: bool = False,
                    after_ms: int = 10) -> None:
    """
    Place the window at a screen position. Placement is deferred (via after(after_ms)) so the window is mapped and has real dimensions; call right after building the window.

    :param window: A tk.Toplevel or tk.Tk window.
    :param position: One of "top_left", "top_right", "bottom_left", "bottom_right", "middle_top", "middle_bottom", "middle_left", "middle_right", "center".
    :param offset_x: Pixels from the position horizontally (positive = right for corners, right for center/middle).
    :param offset_y: Pixels from the position vertically (positive = down for top, up for bottom, down for center).
    :param lift: If True (default), bring the window to front after placing.
    :param focus: If True, give the window keyboard focus after placing (default False).
    :param constrain: If True (default), clamp position so the window stays fully on screen.
    :param after_ms: Milliseconds to wait before placing (default 10). Use 0 for next tick; increase if placement is sporadic.
    """

    check_instance(source=f'{position_window.__name__} window', instance=window, accepted_types=(Toplevel, Tk,), raise_error=True)
    check_int(name=f'{position_window.__name__} offset_x', value=offset_x, raise_error=True)
    check_int(name=f'{position_window.__name__} offset_y', value=offset_y, raise_error=True)
    check_str(name=f'{position_window.__name__} position', value=position, options=VALID_POSITIONS, raise_error=True)
    check_int(name=f'{position_window.__name__} after_ms', value=after_ms, raise_error=True)
    if after_ms < 0:
        raise ValueError(f'{position_window.__name__} after_ms must be >= 0, got {after_ms}.')

    def do_place():
        if verbose:
            print(f'Positioning window at {position}...')
        window.update_idletasks()
        window.update()
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
        elif position == "middle_left":
            x, y = offset_x, (sh - h) // 2 + offset_y
        elif position == "middle_right":
            x, y = sw - w - offset_x, (sh - h) // 2 + offset_y
        else:  # center
            x, y = (sw - w) // 2 + offset_x, (sh - h) // 2 + offset_y
        if constrain:
            x = max(0, min(x, sw - w))
            y = max(0, min(y, sh - h))
        window.geometry(f"+{x}+{y}")
        if lift:
            window.lift()
        if focus:
            try:
                window.focus_force()
            except Exception:
                pass

    # Defer placement so the window is mapped and has real dimensions. after(0) often
    # runs before the WM has sized the window, causing sporadic placement; a short
    # delay (after_ms) makes positioning reliable.
    window.after(after_ms, do_place)
# All credit to https://stackoverflow.com/questions/46571448/tkinter-and-a-html-file - thanks DELICA - https://stackoverflow.com/users/7027346/delica

from cefpython3 import cefpython as cef
import ctypes

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    import Tkinter as tk
import sys
import platform
import logging as _logging

# Fix for PyCharm hints warnings
WindowUtils = cef.WindowUtils()

# Platforms
WINDOWS = (platform.system() == "Windows")
LINUX = (platform.system() == "Linux")
MAC = (platform.system() == "Darwin")

# Globals
logger = _logging.getLogger("tkinter_.py")
url = "localhost:8050/"

class MainFrame(tk.Frame):

    def __init__(self, root):
        self.closing = False
        self.browser = None

        # Root
        root.geometry("900x640")
        tk.Grid.rowconfigure(root, 0, weight=1)
        tk.Grid.columnconfigure(root, 0, weight=1)

        # MainFrame
        tk.Frame.__init__(self, root)
        self.master.title('SimBA Dashboard')
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<Configure>", self.on_configure)
        self.bind("<FocusIn>", self.on_focus_in)
        self.bind("<FocusOut>", self.on_focus_out)
        self.focus_set()

        # Pack MainFrame
        self.pack(fill=tk.BOTH, expand=tk.YES)

    def embed_browser(self):
        window_info = cef.WindowInfo()
        rect = [0, 0, self.winfo_width(), self.winfo_height()]
        window_info.SetAsChild(self.get_window_handle(), rect)
        self.browser = cef.CreateBrowserSync(window_info,
                                             url=url) #todo
        assert self.browser
        self.browser.SetClientHandler(LoadHandler(self))
        self.browser.SetClientHandler(FocusHandler(self))
        self.message_loop_work()

    def get_window_handle(self):
        if self.winfo_id() > 0:
            return self.winfo_id()
        else:
            raise Exception("Couldn't obtain window handle")

    def message_loop_work(self):
        cef.MessageLoopWork()
        self.after(10, self.message_loop_work)

    def on_configure(self, event):

        width = event.width
        height = event.height
        if self.browser:
            if WINDOWS:
                ctypes.windll.user32.SetWindowPos(
                    self.browser.GetWindowHandle(), 0,
                    0, 0, width, height, 0x0002)
            elif LINUX:
                self.browser.SetBounds(0, 0, width, height)
            self.browser.NotifyMoveOrResizeStarted()
        if not self.browser:
            self.embed_browser()

    def on_focus_in(self, _):
        logger.debug("BrowserFrame.on_focus_in")
        if self.browser:
            self.browser.SetFocus(True)
            self.focus_set()

    def on_focus_out(self, _):
        logger.debug("BrowserFrame.on_focus_out")
        if self.browser:
            self.browser.SetFocus(False)

    def on_close(self):
        if self.browser:
            self.browser.CloseBrowser(True)
            self.clear_browser_references()
        self.destroy()
        self.master.destroy()

    def get_browser(self):
        if self.browser:
            return self.browser
        return None

    def clear_browser_references(self):
        self.browser = None


class LoadHandler(object):

    def __init__(self, browser_frame):
        self.browser_frame = browser_frame


class FocusHandler(object):

    def __init__(self, browser):
        self.browser = browser

    def OnTakeFocus(self, next_component, **_):
        logger.debug("FocusHandler.OnTakeFocus, next={next}"
                     .format(next=next_component))

    def OnSetFocus(self, source, **_):
        logger.debug("FocusHandler.OnSetFocus, source={source}"
                     .format(source=source))
        return False

    def OnGotFocus(self, **_):
        """Fix CEF focus issues (#255). Call browser frame's focus_set
           to get rid of type cursor in url entry widget."""
        logger.debug("FocusHandler.OnGotFocus")
        self.browser.focus_set()


# if __name__ == '__main__':
logger.setLevel(_logging.INFO)
stream_handler = _logging.StreamHandler()
formatter = _logging.Formatter("[%(filename)s] %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.info("CEF Python {ver}".format(ver=cef.__version__))
logger.info("Python {ver} {arch}".format(
        ver=platform.python_version(), arch=platform.architecture()[0]))
logger.info("Tk {ver}".format(ver=tk.Tcl().eval('info patchlevel')))
assert cef.__version__ >= "55.3", "CEF Python v55.3+ required to run this"
sys.excepthook = cef.ExceptHook  # To shutdown all CEF processes on error
root = tk.Tk()
app = MainFrame(root)
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Tk must be initialized before CEF otherwise fatal error (Issue #306)
cef.Initialize()
root.mainloop()
# app.mainloop()
cef.Shutdown()

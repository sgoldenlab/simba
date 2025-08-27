from tkinter import ttk
from tkinter.ttk import Style, Treeview
from typing import Any, Dict, Optional, Tuple

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.utils.checks import check_str, check_valid_dict, check_valid_tuple
from simba.utils.enums import Formats


class GetTreeView(PopUpMixin):
    """
    Create a hierarchical TreeView widget for displaying nested dictionary data in a popup window.

    This class creates a TreeView interface that displays nested dictionary data in an expandable tree format.
    Each top-level dictionary key becomes a parent node, and the nested dictionary entries become child nodes.
    The TreeView includes vertical and horizontal scrollbars and supports double-click expansion/collapse functionality.

    .. note::
       The TreeView automatically calculates its height based on the window size to maximize screen usage. Parent nodes can be expanded/collapsed by double-clicking.

    .. image:: _static/img/tree_view.png
       :width: 700
       :align: center

    :param Dict[str, Dict[str, Any]] data: Nested dictionary data where top-level keys become parent nodes  and nested dictionaries become child nodes.
    :param Optional[str] index_col_name: Header text for the tree column (first column). Default is empty string.
    :param Optional[Tuple[str, ...]] headers: Tuple of column headers for data columns. Default is ("VALUE",).
    :param str title: Window title for the popup. Default is 'DATA'.
    :param Optional[Tuple[int, int]] size: Window size as (width, height) tuple. Default is (800, 500).
    :param str theme: Tkinter theme name to use. Default is 'clam' for better visual separation.

    :example:
    >>> from simba.utils.read_write import find_all_videos_in_directory
    >>> from simba.utils.read_write import get_video_meta_data
    >>> videos = find_all_videos_in_directory(directory=r'C:\troubleshooting\mitra\project_folder\videos', as_dict=True)
    >>> results = {}
    >>> for video_name, video_path in videos.items():
    >>>     v = get_video_meta_data(video_path=video_path)
    >>>     results[v['video_name']] = v
    >>> GetTreeView(data=results, index_col_name='VIDEO')
    
    :example:
    >>> # Simple example with custom data
    >>> data = {
    >>>     'Video1': {'fps': 30, 'width': 1920, 'height': 1080},
    >>>     'Video2': {'fps': 25, 'width': 1280, 'height': 720}
    >>> }
    >>> tree_view = GetTreeView(data=data, index_col_name='Videos', title='Video Metadata')
    """

    def __init__(self,
                 data: Dict[str, Dict[str, Any]],
                 index_col_name: Optional[str] = '',
                 headers: Optional[Tuple[str, ...]] = ("VALUE",),
                 title: str = 'DATA',
                 size: Optional[Tuple[int, int]] = (800, 500),
                 theme: str = 'clam'):

        check_valid_dict(x=data, valid_key_dtypes=(str,), valid_values_dtypes=(dict,), min_len_keys=1, source=f'{self.__class__.__name__} data')
        check_valid_tuple(x=headers, source=f'{self.__class__.__name__} headers', valid_dtypes=(str,))
        check_str(name=f'{self.__class__.__name__} title', value=title, allow_blank=False, raise_error=True)
        for cnt, (i, x) in enumerate(data.items()):
            check_valid_dict(x=x, valid_key_dtypes=(str,), min_len_keys=1, source=f'{self.__class__.__name__} data {i}')
        PopUpMixin.__init__(self, title=title, size=size, icon='eye')

        self.main_frm.pack(fill="both", expand=True)

        window_height = size[1] if size else 500
        tree_height = int((window_height - 50) / 20)  # Estimate rows based on window height
        self.tree = Treeview(self.main_frm, columns=headers, show="tree headings", height=tree_height)
        self.tree.pack(fill="both", expand=True, side="left")
        style = Style()
        style.theme_use(theme)

        self.tree.heading('#0', text=index_col_name, anchor="center")
        self.tree.column('#0', anchor="w", width=450, stretch=True)

        for col in headers:
            self.tree.heading(col, text=col, anchor="center")
            self.tree.column(col, anchor="w", width=320, stretch=True)

        vsb = ttk.Scrollbar(self.main_frm, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.main_frm, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        
        self.insert_data(data)
        self.tree.bind("<Double-1>", self.toggle_node)
        #self.main_frm.mainloop()

    def insert_data(self, data: Dict[str, Dict[str, Any]]):
        for parent_key, child_dict in data.items():
            parent_id = self.tree.insert("", "end", text=parent_key, values=("",), tags=('parent',), open=False)
            for k, v in child_dict.items():
                formatted_value = self.format_value(v)
                self.tree.insert(parent_id, "end", text=k, values=(formatted_value,), tags=('child',))

        self.tree.tag_configure('parent', background='#f0f0f0', font=Formats.FONT_HEADER.value)
        self.tree.tag_configure('child', background='#ffffff', font=Formats.FONT_REGULAR.value)

    def format_value(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"{value:,}" if isinstance(value, int) else f"{value:.2f}"
        elif isinstance(value, str):
            return value
        else:
            return str(value)

    def toggle_node(self, event):
        item = self.tree.selection()[0]
        if self.tree.parent(item):
            return
        if self.tree.item(item, "open"):
            self.tree.item(item, open=False)
        else:
            self.tree.item(item, open=True)

    def expand_all(self):
        for item in self.tree.get_children():
            self.tree.item(item, open=True)

    def collapse_all(self):
        for item in self.tree.get_children():
            self.tree.item(item, open=False)


# from simba.utils.read_write import find_all_videos_in_directory
# from simba.utils.read_write import get_video_meta_data
# videos = find_all_videos_in_directory(directory=r'D:\EPM_4\original', as_dict=True)
# results = {}
# for video_name, video_path in videos.items():
#     v = get_video_meta_data(video_path=video_path)
#     results[v['video_name']] = v
# GetTreeView(data=results, index_col_name='VIDEO')
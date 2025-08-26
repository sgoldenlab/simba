from typing import Dict, Any, Optional, Tuple
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.utils.checks import check_valid_dict, check_valid_tuple, check_str
from tkinter.ttk import Treeview, Style
from tkinter import ttk
from simba.utils.enums import Formats


class GetTreeView(PopUpMixin):
    """
    :example:
    >>> from simba.utils.read_write import find_all_videos_in_directory
    >>> from simba.utils.read_write import get_video_meta_data
    >>> videos = find_all_videos_in_directory(directory=r'C:\troubleshooting\mitra\project_folder\videos', as_dict=True)
    >>> results = {}
    >>> for video_name, video_path in videos.items():
    >>>     v = get_video_meta_data(video_path=video_path)
    >>>     results[v['video_name']] = v
    >>> GetTreeView(data=results, index_col_name='VIDEO')
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
        self.tree = Treeview(self.main_frm, columns=headers, show="tree headings", height=20)
        style = Style()
        style.theme_use(theme)

        self.tree.heading('#0', text=index_col_name, anchor="center")
        self.tree.column('#0', anchor="w", minwidth=100, stretch=True)

        for col in headers:
            self.tree.heading(col, text=col, anchor="center")
            self.tree.column(col, anchor="w", minwidth=200, stretch=True)

        vsb = ttk.Scrollbar(self.main_frm, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.main_frm, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.main_frm.grid_rowconfigure(0, weight=1)
        self.main_frm.grid_columnconfigure(0, weight=1)
        self.main_frm.grid_rowconfigure(1, weight=0)
        self.main_frm.grid_columnconfigure(1, weight=0)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
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
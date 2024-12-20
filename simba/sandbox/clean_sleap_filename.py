

def clean_sleap_file_name(filename: str) -> str:
    """
    Clean a SLEAP input filename by removing '.analysis' suffix, the video  and project name prefix to match orginal video name.

     .. note::
       Modified from `vtsai881 <https://github.com/vtsai881>`_.

    :param str filename: The original filename to be cleaned to match video name.
    :returns str: The cleaned filename.

    :example:
    >>> clean_sleap_file_name("projectname.v00x.00x_videoname.analysis.csv")
    >>> 'videoname.csv'
    >>> clean_sleap_file_name("projectname.v00x.00x_videoname.analysis.h5")
    >>> 'videoname.h5'
    """

    if (".analysis" in filename.lower()) and ("_" in filename) and (filename.count('.') >= 3):
        filename_parts = filename.split('.')
        video_num_name = filename_parts[2]
        if '_' in video_num_name:
            return video_num_name.split('_', 1)[1]
        else:
            return filename
    else:
        return filename

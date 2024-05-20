import matplotlib.font_manager

from simba.utils.warnings import NoDataFoundWarning


def get_fonts():
    font_dict = {f.name: f.fname for f in matplotlib.font_manager.fontManager.ttflist if not f.name.startswith('.')}
    font_dict = dict(sorted(font_dict.items()))
    if len(font_dict) == 0:
        NoDataFoundWarning(msg='No fonts found on disk using matplotlib.font_manager', source=get_fonts.__name__)
    return font_dict


get_fonts()





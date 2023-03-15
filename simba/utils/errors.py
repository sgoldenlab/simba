from tkinter import messagebox as mb

WINDOW_TITLE = 'SIMBA ERROR'

class SimbaError(Exception):
    def __init__(self, msg: str, show_window: bool):
        self.msg = msg
        if show_window:
            mb.showerror(title=WINDOW_TITLE, message=msg)

class NoSpecifiedOutputError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')

class ROICoordinatesNotFoundError(SimbaError):
    def __init__(self, expected_file_path: str, show_window: bool = False):
        msg = f'No ROI coordinates found. Please use the [ROI] tab to define ROIs. Expected at location {expected_file_path}'
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')

class NoChoosenClassifierError(SimbaError):
    def __init__(self, show_window: bool = False):
        msg = f'Select at least one classifiers'
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')

class NoChoosenROIError(SimbaError):
    def __init__(self, show_window: bool = False):
        msg = f'Please select at least one ROI.'
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')

class NoChoosenMeasurementError(SimbaError):
    def __init__(self, show_window: bool = False):
        msg = 'Please select at least one measurement to calculate descriptive statistics for.'
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')


class NoROIDataError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(msg)


class MixedMosaicError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(msg)

class AnimalNumberError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')

class InvalidFilepathError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')


class NoFilesFoundError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')

class NotDirectoryError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')

class DirectoryExistError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')

class FileExistError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')


class FrameRangeError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')

class AdvancedLabellingError(SimbaError):
    def __init__(self, frame: str, lbl_lst: list, unlabel_lst: list, show_window: bool = False):
        self.msg = 'SIMBA ERROR: In advanced labelling of multiple behaviors, any annotated frame cannot have some ' \
                   'behaviors annotated as present/absent, while other behaviors are un-labelled. All behaviors need ' \
                   'labels for a frame with one or more labels. In frame {}, behaviors {} are labelled, while behaviors ' \
                   '{} are un-labelled.'.format(str(frame), lbl_lst, unlabel_lst)
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')

class InvalidHyperparametersFileError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')

class InvalidVideoFileError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')

class InvalidFileTypeError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')

class CountError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')

class DuplicationError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')






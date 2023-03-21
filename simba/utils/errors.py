from tkinter import messagebox as mb

WINDOW_TITLE = 'SIMBA ERROR'

class SimbaError(Exception):
    def __init__(self, msg: str, show_window: bool):
        self.msg = msg
        if show_window:
            mb.showerror(title=WINDOW_TITLE, message=msg)

    def __str__(self):
        return self.msg


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
        msg = 'SIMBA ERROR: In advanced labelling of multiple behaviors, any annotated frame cannot have some ' \
                   'behaviors annotated as present/absent, while other behaviors are un-labelled. All behaviors need ' \
                   'labels for a frame with one or more labels. In frame {}, behaviors {} are labelled, while behaviors ' \
                   '{} are un-labelled.'.format(str(frame), lbl_lst, unlabel_lst)
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA ERROR: {msg}')

class InvalidHyperparametersFileError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA HYPERPAREMETER FILE ERROR: {msg}')

class InvalidVideoFileError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA VIDEO FILE ERROR: {msg}')

class InvalidFileTypeError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA INVALID FILE TYPE ERROR: {msg}')

class FaultyTrainingSetError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA INVALID ML TRAINING SET ERROR: {msg}')

class CountError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA COUNT ERROR: {msg}')

class DuplicationError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA DUPLICATION ERROR: {msg}')

class InvalidInputError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA VALUE ERROR: {msg}')

class IntegerError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA INTEGER ERROR: {msg}')

class StringError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA STRING ERROR: {msg}')

class FloatError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA FLOAT ERROR: {msg}')

class MissingProjectConfigEntryError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA MISSING PROJECT CONFIG ENTRY ERROR: {msg}')

class CorruptedFileError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA CORRUPTED FILE ERROR: {msg}')

class ParametersFileError(SimbaError):
    def __init__(self, msg: str, show_window: bool = False):
        super().__init__(msg=msg, show_window=show_window)
        print(f'SIMBA VIDEO PARAMETERS FILE ERROR: {msg}')



#####

class ThirdPartyAnnotationEventCountError(SimbaError):
    def __init__(self, video_name: str, clf_name: str, start_event_cnt: int, stop_event_cnt: int, show_window: bool = False):
        msg = f'SIMBA THIRD-PARTY ANNOTATION ERROR: The annotations for behavior {clf_name} in video {video_name} contains {str(start_event_cnt)} start events and {str(stop_event_cnt)} stop events. SimBA requires the number of stop and start event counts to be equal'
        super().__init__(msg=msg, show_window=show_window)
        print(msg)

class ThirdPartyAnnotationOverlapError(SimbaError):
    def __init__(self, video_name: str, clf_name: str, show_window: bool = False):
        msg = f'SIMBA THIRD-PARTY ANNOTATION ERROR: The annotations for behavior {clf_name} in video {video_name} contains behavior start events that are initiated PRIOR to the PRECEDING behavior event ending. SimBA requires a specific behavior event to end before another behavior event can start.'
        super().__init__(msg=msg, show_window=show_window)
        print(msg)

class ColumnNotFoundError(SimbaError):
    def __init__(self, column_name: str, file_name: str, show_window: bool = False):
        msg = f'SIMBA ERROR: Field name {column_name} could not be found in file {file_name}'
        super().__init__(msg=msg, show_window=show_window)
        print(msg)

class AnnotationFileNotFoundError(SimbaError):
    def __init__(self, video_name: str, show_window: bool = False):
        msg = f'SIMBA THIRD-PARTY ANNOTATION ERROR: NO ANNOTATION DATA FOR VIDEO {video_name} FOUND'
        super().__init__(msg=msg, show_window=show_window)
        print(msg)




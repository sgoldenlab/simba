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



#####


class ThirdPartyAnnotationFileNotFoundError(SimbaError):
    def __init__(self, video_name: str, show_window: bool = False):
        msg = f'SIMBA ERROR: Could not find file in project_folder/csv/features_extracted directory representing annotations for video {video_name}'
        super().__init__(msg=msg, show_window=show_window)
        print(msg)

class ThirdPartyAnnotationsFpsConflictError(SimbaError):
    def __init__(self, video_name: str, annotation_fps: int, video_fps: int, show_window: bool = False):
        msg = f'SIMBA THIRD-PARTY ANNOTATION ERROR: The FPS for video {video_name} is set to {str(video_fps)} in SimBA and {str(annotation_fps)} in the annotation file'
        super().__init__(msg=msg, show_window=show_window)
        print(msg)


class ThirdPartyAnnotationsMissingAnnotationsError(SimbaError):
    def __init__(self, video_name: str, clf_names: list, show_window: bool = False):
        msg = f'SIMBA THIRD-PARTY ANNOTATION ERROR: No annotations detected for SimBA classifier(s) named {clf_names} for video {video_name}'
        super().__init__(msg=msg, show_window=show_window)
        print(msg)

class ThirdPartyAnnotationOverlapError(SimbaError):
    def __init__(self, video_name: str, clf_name: str, show_window: bool = False):
        msg = f'SIMBA THIRD-PARTY ANNOTATION ERROR: The annotations for behavior {clf_name} in video {video_name} contains behavior start events that are initiated PRIOR to the PRECEDING behavior event ending. SimBA requires a specific behavior event to end before another behavior event can start.'
        super().__init__(msg=msg, show_window=show_window)
        print(msg)

class ThirdPartyAnnotationsAdditionalClfError(SimbaError):
    def __init__(self, video_name: str, clf_names: list, show_window: bool = False):
        msg = f'SIMBA THIRD-PARTY ANNOTATION ERROR: Annotations file for video {video_name} has annotations for the following behaviors {clf_names} that are NOT classifiers named in the SimBA project.'
        super().__init__(msg=msg, show_window=show_window)
        print(msg)

class ThirdPartyAnnotationEventCountError(SimbaError):
    def __init__(self, video_name: str, clf_name: str, start_event_cnt: int, stop_event_cnt: int, show_window: bool = False):
        msg = f'SIMBA THIRD-PARTY ANNOTATION ERROR: The annotations for behavior {clf_name} in video {video_name} contains {str(start_event_cnt)} start events and {str(stop_event_cnt)} stop events. SimBA requires the number of stop and start event counts to be equal.'
        super().__init__(msg=msg, show_window=show_window)
        print(msg)

class ThirdPartyAnnotationsClfMissingError(SimbaError):
    def __init__(self, video_name: str, clf_name: str, show_window: bool = False):
        msg = f'SIMBA THIRD-PARTY ANNOTATION WARNING: No annotations detected for video {video_name} and behavior {clf_name}.'
        super().__init__(msg=msg, show_window=show_window)
        print(msg)

class ThirdPartyAnnotationsOutsidePoseEstimationDataError(SimbaError):
    def __init__(self,
                 video_name: str,
                 frm_cnt: int,
                 clf_name: str or None = None,
                 annotation_frms: int or None = None,
                 first_error_frm: int or None = None,
                 ambiguous_cnt: int or None = None,
                 show_window: bool = False):

        if clf_name:
            msg = (
                f'SIMBA THIRD-PARTY ANNOTATION WARNING: SimBA found THIRD-PARTY annotations for behavior {clf_name} in video '
                f'{video_name} that are annotated to occur at times which is not present in the '
                f'video data you imported into SIMBA. The video you imported to SimBA has {str(frm_cnt)} frames. '
                f'However, in BORIS, you have annotated {clf_name} to happen at frame number {str(first_error_frm)}. '
                f'These ambiguous annotations occur in {str(ambiguous_cnt)} different frames for video {video_name} that SimBA will **remove** by default. '
                f'Please make sure you imported the same video as you annotated in BORIS into SimBA and the video is registered with the correct frame rate.')
        else:
            msg = f'SIMBA THIRD-PARTY ANNOTATION WARNING: The annotations for video {video_name} contain data for {str(annotation_frms)} frames. The pose-estimation features for the same video contain data for {str(frm_cnt)} frames.'

        super().__init__(msg=msg, show_window=show_window)
        print(msg)

__author__ = "Simon Nilsson"

from tkinter import messagebox as mb

from simba.utils.enums import Defaults, TagNames
from simba.utils.printing import log_event

WINDOW_TITLE = "SIMBA ERROR"


class SimbaError(Exception):
    def __init__(self, msg: str, source: str = " ", show_window: bool = False):
        self.msg, self.source, self.show_window = msg, source, show_window
        self.print_and_log_error()

    def __str__(self):
        return self.msg

    def print_and_log_error(self):
        log_event(logger_name=f"{self.source}.{self.__class__.__name__}", log_type=TagNames.ERROR.value,msg=self.msg)
        print(f"{self.msg}{Defaults.STR_SPLIT_DELIMITER.value}{TagNames.ERROR.value}")
        if self.show_window:
            mb.showerror(title=WINDOW_TITLE, message=self.msg)


class NoSpecifiedOutputError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = True):
        super().__init__(msg=msg, source=source, show_window=show_window)


class ROICoordinatesNotFoundError(SimbaError):
    def __init__(self, expected_file_path: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA ROI COORDINATES ERROR: No ROI coordinates found. Please use the [ROI] tab to define ROIs. Expected at location {expected_file_path}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class NoChoosenClassifierError(SimbaError):
    def __init__(self, source: str = "", show_window: bool = False):
        msg = f"Select at least one classifier"
        super().__init__(msg=msg, source=source, show_window=show_window)


class NoChoosenROIError(SimbaError):
    def __init__(self, source: str = "", show_window: bool = False):
        msg = f"Please select at least one ROI."
        super().__init__(msg=msg, source=source, show_window=show_window)


class NoChoosenMeasurementError(SimbaError):
    def __init__(self, source: str = "", show_window: bool = False):
        msg = "SIMBA NoChoosenMeasurementError ERROR: Please select at least one measurement to calculate descriptive statistics for."
        super().__init__(msg=msg, source=source, show_window=show_window)


class NoDataError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA NO DATA ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class SamplingError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA SAMPLING ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class PermissionError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA PERMISSION ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class NoROIDataError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA NO ROI DATA ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class MixedMosaicError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA MixedMosaicError ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class ClassifierInferenceError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA CLASSIFIER INFERENCE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class AnimalNumberError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA ANIMAL NUMBER ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class InvalidFilepathError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA INVALID FILE PATH ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class NoFilesFoundError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA NO FILES FOUND ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class DataHeaderError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA DATA HEADER ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class NotDirectoryError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA NOT A DIRECTORY ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class DirectoryExistError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA DIRECTORY ALREADY EXIST ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class FileExistError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA FILE EXIST ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class FrameRangeError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA FRAME RANGE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class AdvancedLabellingError(SimbaError):
    def __init__(
        self,
        frame: str,
        lbl_lst: list,
        unlabel_lst: list,
        source: str = "",
        show_window: bool = False,
    ):
        msg = (
            "SIMBA ADVANCED LABELLING ERROR: In advanced labelling of multiple behaviors, any annotated frame cannot have some "
            "behaviors annotated as present/absent, while other behaviors are un-labelled. All behaviors need "
            "labels for a frame with one or more labels. In frame {}, behaviors {} are labelled, while behaviors "
            "{} are un-labelled.".format(str(frame), lbl_lst, unlabel_lst)
        )
        super().__init__(msg=msg, source=source, show_window=show_window)


class InvalidHyperparametersFileError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA HYPERPARAMETER FILE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class InvalidVideoFileError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA VIDEO FILE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class InvalidFileTypeError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA INVALID FILE TYPE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class FaultyTrainingSetError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA INVALID ML TRAINING SET ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class CountError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA COUNT ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class FeatureNumberMismatchError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA FEATURE NUMBER MISMATCH ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class DuplicationError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA DUPLICATION ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class InvalidInputError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA VALUE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class IntegerError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA INTEGER ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class StringError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA STRING ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class FloatError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA FLOAT ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class MissingProjectConfigEntryError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA MISSING PROJECT CONFIG ENTRY ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class MissingColumnsError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA MISSING COLUMN ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class CorruptedFileError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA READ FILE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class ParametersFileError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA VIDEO PARAMETERS FILE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class ColumnNotFoundError(SimbaError):
    def __init__(
        self,
        column_name: str,
        file_name: str,
        source: str = "",
        show_window: bool = False,
    ):
        msg = f"SIMBA ERROR: Field name {column_name} could not be found in file {file_name}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class BodypartColumnNotFoundError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA BODY_PART COLUMN NOT FOUND ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class AnnotationFileNotFoundError(SimbaError):
    def __init__(self, video_name: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA THIRD-PARTY ANNOTATION ERROR: NO ANNOTATION DATA FOR VIDEO {video_name} FOUND"
        super().__init__(msg=msg, source=source, show_window=show_window)


class DirectoryNotEmptyError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA DIRECTORY NOT EMPTY ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


#####


class ThirdPartyAnnotationFileNotFoundError(SimbaError):
    def __init__(self, video_name: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA ERROR: Could not find file in project_folder/csv/features_extracted directory representing annotations for video {video_name}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class ThirdPartyAnnotationsFpsConflictError(SimbaError):
    def __init__(
        self,
        video_name: str,
        annotation_fps: int,
        video_fps: int,
        source: str = "",
        show_window: bool = False,
    ):
        msg = f"SIMBA THIRD-PARTY ANNOTATION ERROR: The FPS for video {video_name} is set to {str(video_fps)} in SimBA and {str(annotation_fps)} in the annotation file"
        super().__init__(msg=msg, source=source, show_window=show_window)


class ThirdPartyAnnotationsMissingAnnotationsError(SimbaError):
    def __init__(
        self,
        video_name: str,
        clf_names: list,
        source: str = "",
        show_window: bool = False,
    ):
        msg = f"SIMBA THIRD-PARTY ANNOTATION ERROR: No annotations detected for SimBA classifier(s) named {clf_names} for video {video_name}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class ThirdPartyAnnotationOverlapError(SimbaError):
    def __init__(
        self,
        video_name: str,
        clf_name: str,
        source: str = "",
        show_window: bool = False,
    ):
        msg = f"SIMBA THIRD-PARTY ANNOTATION ERROR: The annotations for behavior {clf_name} in video {video_name} contains behavior start events that are initiated PRIOR to the PRECEDING behavior event ending. SimBA requires a specific behavior event to end before another behavior event can start."
        super().__init__(msg=msg, source=source, show_window=show_window)


class ThirdPartyAnnotationsAdditionalClfError(SimbaError):
    def __init__(
        self,
        video_name: str,
        clf_names: list,
        source: str = "",
        show_window: bool = False,
    ):
        msg = f"SIMBA THIRD-PARTY ANNOTATION ERROR: Annotations file for video {video_name} has annotations for the following behaviors {clf_names} that are NOT classifiers named in the SimBA project."
        super().__init__(msg=msg, source=source, show_window=show_window)


class ThirdPartyAnnotationEventCountError(SimbaError):
    def __init__(
        self,
        video_name: str,
        clf_name: str,
        start_event_cnt: int,
        stop_event_cnt: int,
        source: str = "",
        show_window: bool = False,
    ):
        msg = f"SIMBA THIRD-PARTY ANNOTATION ERROR: The annotations for behavior {clf_name} in video {video_name} contains {str(start_event_cnt)} start events and {str(stop_event_cnt)} stop events. SimBA requires the number of stop and start event counts to be equal."
        super().__init__(msg=msg, source=source, show_window=show_window)


class ThirdPartyAnnotationsClfMissingError(SimbaError):
    def __init__(
        self,
        video_name: str,
        clf_name: str,
        source: str = "",
        show_window: bool = False,
    ):
        msg = f"SIMBA THIRD-PARTY ANNOTATION WARNING: No annotations detected for video {video_name} and behavior {clf_name}."
        super().__init__(msg=msg, source=source, show_window=show_window)


class ThirdPartyAnnotationsOutsidePoseEstimationDataError(SimbaError):
    def __init__(
        self,
        video_name: str,
        frm_cnt: int,
        clf_name: str or None = None,
        annotation_frms: int or None = None,
        first_error_frm: int or None = None,
        ambiguous_cnt: int or None = None,
        source: str = "",
        show_window: bool = False,
    ):

        if clf_name:
            msg = (
                f"SIMBA THIRD-PARTY ANNOTATION WARNING: SimBA found THIRD-PARTY annotations for behavior {clf_name} in video "
                f"{video_name} that are annotated to occur at times which is not present in the "
                f"video data you imported into SIMBA. The video you imported to SimBA has {str(frm_cnt)} frames. "
                f"However, in BORIS, you have annotated {clf_name} to happen at frame number {str(first_error_frm)}. "
                f"These ambiguous annotations occur in {str(ambiguous_cnt)} different frames for video {video_name} that SimBA will **remove** by default. "
                f"Please make sure you imported the same video as you annotated in BORIS into SimBA and the video is registered with the correct frame rate."
            )
        else:
            msg = f"SIMBA THIRD-PARTY ANNOTATION WARNING: The annotations for video {video_name} contain data for {str(annotation_frms)} frames. The pose-estimation features for the same video contain data for {str(frm_cnt)} frames."

        super().__init__(msg=msg, source=source, show_window=show_window)


class FFMPEGCodecGPUError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA FFMPEG CODEC ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class FFMPEGNotFoundError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA FFMPEG NOT FOUND ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class ArrayError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA ARRAY SIZE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)

class ResolutionError(SimbaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SIMBA RESOLUTION ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


# test = NoSpecifiedOutputError(msg='test', source='test.method')
# test = FFMPEGNotFoundError(msg='323')

# class NoSpecifiedOutputError(SimbaError):
#     @log_error_decorator()
#     def __init__(self, msg: str, source: str = '', show_window: bool = True):

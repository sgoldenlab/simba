import logging

def ThirdPartyAnnotationsOutsidePoseEstimationDataWarning(video_name: str,
                                                          frm_cnt: int,
                                                          log_status: bool = False,
                                                          clf_name: str or None = None,
                                                          annotation_frms: int or None = None,
                                                          first_error_frm: int or None=None,
                                                          ambiguous_cnt: int or None=None,
                                                          ):
    if clf_name:
        msg = (f'SIMBA THIRD-PARTY ANNOTATION WARNING: SimBA found THIRD-PARTY annotations for behavior {clf_name} in video '
               f'{video_name} that are annotated to occur at times which is not present in the '
               f'video data you imported into SIMBA. The video you imported to SimBA has {str(frm_cnt)} frames. '
               f'However, in BORIS, you have annotated {clf_name} to happen at frame number {str(first_error_frm)}. '
               f'These ambiguous annotations occur in {str(ambiguous_cnt)} different frames for video {video_name} that SimBA will **remove** by default. '
               f'Please make sure you imported the same video as you annotated in BORIS into SimBA and the video is registered with the correct frame rate. '
               f'SimBA will only append annotations made to the frames present in the pose estimation data.')
    else:
        msg = f'SIMBA THIRD-PARTY ANNOTATION WARNING: The annotations for video {video_name} contain data for {str(annotation_frms)} frames. The pose-estimation features for the same video contain data for {str(frm_cnt)} frames. SimBA will use the annotations for the frames present in the pose-estimation data and discard the rest. If the annotation data is shorter than the pose-estimation data, SimBA will assume the missing annotation frames are all behavior absent.'
    if log_status: logging.warning(msg=msg)
    print(msg)

def ThirdPartyAnnotationsClfMissingWarning(video_name: str,
                                              clf_name: str):
    msg = f'SIMBA THIRD-PARTY ANNOTATION WARNING: No annotations detected for video {video_name} and behavior {clf_name}. ' \
          f'SimBA will set all frame annotations as absent.'
    print(msg)

def ThirdPartyAnnotationsAdditionalClfWarning(video_name: str,
                                              clf_names: list,
                                              log_status: bool=False):
    msg = f'SIMBA THIRD-PARTY ANNOTATION WARNING: Annotations file for video {video_name} has annotations for the following behaviors {clf_names} that are NOT classifiers named in the SimBA project. SimBA will OMIT appending the data for these {str(len(clf_names))} classifiers.'
    if log_status: logging.warning(msg=msg)
    print(msg)


def ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app: str,
                                                  file_path: str,
                                                  log_status: bool=False):
    msg = f'SIMBA WARNING: {file_path} is not a valid {annotation_app} file and is skipped. See the SimBA GitHub repository for expected file format'
    if log_status: logging.warning(msg=msg)
    print(msg)

def ThirdPartyAnnotationsMissingAnnotationsWarning(video_name: str,
                                                   clf_names: list,
                                                   log_status: bool=False):
    msg = f'SIMBA THIRD-PARTY ANNOTATION WARNING: No annotations detected for SimBA classifier(s) named {clf_names} for video {video_name}. All frame annotations will be set to behavior absent (0).'
    if log_status: logging.warning(msg=msg)
    print(msg)

def ThirdPartyAnnotationsFpsConflictWarning(video_name: str,
                                            annotation_fps: int,
                                            video_fps: int):
    msg = f'SIMBA THIRD-PARTY ANNOTATION WARNING: The FPS for video {video_name} is set to {str(video_fps)} in SimBA and {str(annotation_fps)} in the annotation file'
    print(msg)


def ThirdPartyAnnotationEventCountWarning(video_name: str, clf_name: str, start_event_cnt: int, stop_event_cnt: int, log_status: bool=False):
    msg = f'SIMBA THIRD-PARTY ANNOTATION WARNING: The annotations for behavior {clf_name} in video {video_name} contains {str(start_event_cnt)} start events and {str(stop_event_cnt)} stop events. SimBA requires the number of stop and start event counts to be equal. SimBA will try to find and delete the odd event stamps.'
    if log_status: logging.warning(msg=msg)
    print(msg)

def ThirdPartyAnnotationOverlapWarning(video_name: str,
                                       clf_name: str,
                                       log_status: bool=False):
    msg = f'SIMBA THIRD-PARTY ANNOTATION WARNING: The annotations for behavior {clf_name} in video {video_name} contains behavior start events that are initiated PRIOR to the PRECEDING behavior event ending. SimBA requires a specific behavior event to end before another behavior event can start. SimBA will try and delete these events.'
    if log_status: logging.warning(msg=msg)
    print(msg)

def ThirdPartyAnnotationFileNotFoundWarning(video_name: str, log_status: bool=False):
    msg = f'SIMBA THIRD-PARTY ANNOTATION WARNING: Could not find annotations for video features file {video_name} in the annotations directory.'
    if log_status: logging.warning(msg=msg)
    print(msg)
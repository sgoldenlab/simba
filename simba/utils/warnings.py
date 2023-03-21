import warnings

def ThirdPartyAnnotationsOutsidePoseEstimationDataWarning(video_name: str,
                                                          clf_name: str,
                                                          frm_cnt: int,
                                                          first_error_frm: int,
                                                          ambiguous_cnt: int):
    msg = (f'SIMBA THIRD-PARTY ANNOTATION WARNING: SimBA found THIRD-PARTY annotations for behavior {clf_name} in video '
           f'{video_name} that are annotated to occur at times which is not present in the '
           f'video data you imported into SIMBA. The video you imported to SimBA has {str(frm_cnt)} frames. '
           f'However, in BORIS, you have annotated {clf_name} to happen at frame number {str(first_error_frm)}. '
           f'These ambiguous annotations occur in {str(ambiguous_cnt)} different frames for video {video_name} that SimBA will **remove** by default. '
           f'Please make sure you imported the same video as you annotated in BORIS into SimBA and the video is registered with the correct frame rate.')
    print(msg)

def ThirdPartyAnnotationsClfMissingWarning(video_name: str,
                                              clf_name: str):
    msg = f'SIMBA THIRD-PARTY ANNOTATION WARNING: No annotations detected for video {video_name} and behavior {clf_name}. ' \
          f'SimBA will set all frame annotations as absent.'
    print(msg)

def ThirdPartyAnnotationsAdditionalClfWarning(video_name: str,
                                              clf_names: list):
    msg = f'SIMBA THIRD-PARTY ANNOTATION WARNING: Annotation file for video {video_name} has annotations for the following behaviors {clf_names} that are NOT classifiers named in the SimBA project. SimBA will OMIT appending the data for these classifiers'
    print(msg)



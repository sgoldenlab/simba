from simba.plotting.ROI_feature_visualizer_mp import \
    ROIfeatureVisualizerMultiprocess

## DEFINE A LIST OF ROI NAMES THAT WE WISH TO SEE ROI DIRECTIONALITY FOR.
# CHANGE My_first and My_second_poygon to the actual names of your ROIS
roi_subset = ["My_first_polygon", "My_second_poygon"]

# CHANGE THIS TO YOUR ACTUAL CONFIG PATH
CONFIG_PATH = (
    "/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini",
)

# CHANGE THIS TO YOUR ACTUAL VIDEO NAME
VIDEO_NAME = "Together_1.avi"


################################
STYLE_ATTIBUTES = {
    "ROI_centers": True,  # SHOW THE CENTROID OF THE ROIs AS A CIRCLE
    "ROI_ear_tags": True,  # SHOW THE EAR TAGS OF THE ROIs as CIRCLES
    "Directionality": True,  # IF VIABLE, SHOW DIRECTIONALITY TOWARDS ROIs
    "Directionality_style": "Funnel",  # LINE OR FUNNEL
    "Border_color": (0, 128, 0),  # COLOR OF THE SIDE-BAR
    "Pose_estimation": True,  # SHOW POSE-ESTIMATED LOCATIONS
    "Directionality_roi_subset": roi_subset,
}  # ONLY SHOW DIRECTIONALITY TO THE ROIS DEFINED IN LIST


roi_feature_visualizer = ROIfeatureVisualizerMultiprocess(
    config_path=CONFIG_PATH,
    video_name=VIDEO_NAME,
    style_attr=STYLE_ATTIBUTES,
    core_cnt=-1,
)
roi_feature_visualizer.run()

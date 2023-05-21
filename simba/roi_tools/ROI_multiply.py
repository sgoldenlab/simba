import glob
import os

import pandas as pd

from simba.utils.enums import ConfigKey, Keys, Paths
from simba.utils.errors import NoROIDataError
from simba.utils.printing import stdout_success
from simba.utils.read_write import get_fn_ext, read_config_file


def create_emty_df(shape_type):
    col_list = None
    if shape_type == Keys.ROI_RECTANGLES.value:
        col_list = [
            "Video",
            "Shape_type",
            "Name",
            "Color name",
            "Color BGR",
            "Thickness",
            "topLeftX",
            "topLeftY",
            "Bottom_right_X",
            "Bottom_right_Y",
            "width",
            "height",
            "Tags",
            "Ear_tag_size",
        ]
    if shape_type == Keys.ROI_CIRCLES.value:
        col_list = [
            "Video",
            "Shape_type",
            "Name",
            "Color name",
            "Color BGR",
            "Thickness",
            "centerX",
            "centerY",
            "radius",
            "Tags",
            "Ear_tag_size",
        ]
    if shape_type == Keys.ROI_POLYGONS.value:
        col_list = [
            "Video",
            "Shape_type",
            "Name",
            "Color name",
            "Color BGR",
            "Thickness",
            "Center_X",
            "Center_Y",
            "vertices",
            "Tags",
            "Ear_tag_size",
        ]
    return pd.DataFrame(columns=col_list)


def multiply_ROIs(config_path, filename):
    _, CurrVidName, ext = get_fn_ext(filename)
    config = read_config_file(config_path=config_path)
    projectPath = config.get(
        ConfigKey.GENERAL_SETTINGS.value, ConfigKey.PROJECT_PATH.value
    )
    videoPath = os.path.join(projectPath, "videos")
    ROIcoordinatesPath = os.path.join(projectPath, "logs", Paths.ROI_DEFINITIONS.value)

    if not os.path.isfile(ROIcoordinatesPath):
        raise NoROIDataError(
            msg="Cannot multiply ROI definitions: no ROI definitions exist in SimBA project"
        )
    rectanglesInfo = pd.read_hdf(ROIcoordinatesPath, key=Keys.ROI_RECTANGLES.value)
    circleInfo = pd.read_hdf(ROIcoordinatesPath, key=Keys.ROI_CIRCLES.value)
    polygonInfo = pd.read_hdf(ROIcoordinatesPath, key=Keys.ROI_POLYGONS.value)

    try:
        r_df = rectanglesInfo[rectanglesInfo["Video"] == CurrVidName]
    except KeyError:
        r_df = create_emty_df("rectangle")

    try:
        c_df = circleInfo.loc[circleInfo["Video"] == str(CurrVidName)]
    except KeyError:
        c_df = create_emty_df("circle")

    try:
        p_df = polygonInfo.loc[polygonInfo["Video"] == str(CurrVidName)]
    except KeyError:
        p_df = create_emty_df("polygon")

    if len(r_df) == 0 and len(c_df) == 0 and len(p_df) == 0:
        print(
            "Cannot replicate ROIs to all videos: no ROI records exist for "
            + str(CurrVidName)
        )

    else:
        videofilesFound = glob.glob(videoPath + "/*.mp4") + glob.glob(
            videoPath + "/*.avi"
        )
        duplicatedRec, duplicatedCirc, duplicatedPoly = (
            r_df.copy(),
            c_df.copy(),
            p_df.copy(),
        )
        for vids in videofilesFound:
            _, vid_name, ext = get_fn_ext(vids)
            duplicatedRec["Video"], duplicatedCirc["Video"], duplicatedPoly["Video"] = (
                vid_name,
                vid_name,
                vid_name,
            )
            r_df = r_df.append(duplicatedRec, ignore_index=True)
            c_df = c_df.append(duplicatedCirc, ignore_index=True)
            p_df = p_df.append(duplicatedPoly, ignore_index=True)
        r_df = r_df.drop_duplicates(subset=["Video", "Name"], keep="first")
        c_df = c_df.drop_duplicates(subset=["Video", "Name"], keep="first")
        p_df = p_df.drop_duplicates(subset=["Video", "Name"], keep="first")

        store = pd.HDFStore(ROIcoordinatesPath, mode="w")
        store["rectangles"] = r_df
        store["circleDf"] = c_df
        store["polygons"] = p_df
        store.close()
        stdout_success(msg=f"ROI(s) for {CurrVidName} applied to all videos")
        print()
        print(
            'Next, click on "draw" to modify ROI location(s) or click on "reset" to remove ROI drawing(s)'
        )

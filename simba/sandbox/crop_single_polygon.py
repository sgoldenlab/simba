import multiprocessing
import os
import platform
from typing import Union

from shapely.geometry import Polygon

from simba.mixins.image_mixin import ImageMixin
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_all_videos_in_directory, get_fn_ext,
                                    get_video_meta_data)
from simba.video_processors.roi_selector_polygon import ROISelectorPolygon


def crop_multiple_videos_circles(
    in_dir: Union[str, os.PathLike], out_dir: Union[str, os.PathLike]
) -> None:
    """
    Crop multiple videos based on circular regions of interest (ROIs) selected by the user.

    :param  Union[str, os.PathLike] in_dir: The directory containing input video files.
    :param  Union[str, os.PathLike] out_dir: The directory to save the cropped video files.

    .. note::
       This function crops multiple videos based on circular ROIs selected by the user.
       The user is asked to define a circle manually in one video within the input directory.
       The function then crops all the video in the input directory according to the shape defined
       using the first video and saves the videos in the ``out_dir`` with the same filenames as the original videos..

    :example:
    >>> crop_multiple_videos_circles(in_dir='/Users/simon/Desktop/edited/tests', out_dir='/Users/simon/Desktop')
    """

    check_if_dir_exists(in_dir=in_dir)
    check_if_dir_exists(in_dir=out_dir)
    video_files = find_all_videos_in_directory(directory=in_dir)
    polygon_selector = ROISelectorPolygon(path=os.path.join(in_dir, video_files[0]))
    polygon_selector.run()
    vertices = polygon_selector.polygon_vertices
    polygon = Polygon(vertices)
    timer = SimbaTimer(start=True)
    if (platform.system() == "Darwin") and (multiprocessing.get_start_method() is None):
        multiprocessing.set_start_method("spawn", force=True)
    for video_cnt, video_path in enumerate(video_files):
        print(
            f"Polygon cropping video {video_path} ({video_cnt+1}/{len(video_files)})..."
        )
        video_path = os.path.join(in_dir, video_path)
        _, video_name, _ = get_fn_ext(filepath=video_path)
        save_path = os.path.join(out_dir, f"{video_name}.mp4")
        video_meta_data = get_video_meta_data(video_path=video_path)
        polygons = [polygon for x in range(video_meta_data["frame_count"])]
        polygons = ImageMixin().slice_shapes_in_imgs(
            imgs=video_path, shapes=polygons, verbose=False
        )
        _ = ImageMixin.img_stack_to_video(
            imgs=polygons, save_path=save_path, fps=video_meta_data["fps"]
        )
    timer.stop_timer()
    stdout_success(
        msg=f"Polygon-based cropped {len(video_files)} files to directory {out_dir}",
        elapsed_time=timer.elapsed_time_str,
    )


crop_multiple_videos_circles(
    in_dir="/Users/simon/Desktop/edited/tests", out_dir="/Users/simon/Desktop"
)

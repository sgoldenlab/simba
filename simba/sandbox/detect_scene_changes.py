from typing import Union, Optional
import os
import subprocess


def detect_scene_changes(video_path: Union[str, os.PathLike], threshold: Optional[float] = 0.4):
    cmd = f"ffmpeg -i {video_path} -vf select='gt(scene\\,{threshold})',showinfo -vsync vfr -f null -"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, _ = process.communicate()
    output.decode("utf-8")
    print(output)

detect_scene_changes(video_path='/Users/simon/Desktop/video_test/test/concatenated.mp4', threshold=0.01)
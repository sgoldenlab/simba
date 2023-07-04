
#### GPU vs CPU runtimes by video processing function


| TASK                                 | CPU   | GPU  |
|---  |---|---|
| Downsample video from 30 to 15 fps   | 208s   | 60s   |
| Convert video to powerpoint accepted format  | 427s  |  178s |
| Re-encode to mp4 | 288s  |77s   |
| Re-encode to greyscale | 252s  | 303s  |
| Print frame count | 580s  |80s   |
| Remove first 10s of video  |294s   |76s   |
| Cut between 00:00:30 and 00:10:30 of video  |58s   |54s   |
| Downsample to 600x400 resolution  |127s   |83s   |
| Create 15s GIF  |20s   |1s   |
| Multi-split into two 50s clips  |10s   |4s   |
| Crop single video  |134s   |59s   |
| Horizontally concatenate two videos  |587s   |161s   |
| Create 2x2 mosaic concatenation 640x480 tiles |1536s  |503s   |


* Using single NVIDEA RTX2080, 54min video recorded at 30fps and 1002x640 resolution.

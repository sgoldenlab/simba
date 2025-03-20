## INTRODUCTION

Many of us use pose-estimation to track animal body-parts. However, I have noticed that in many scenorious this is not required and simpler, faster, background subtraction methods as those deployed by commercial tools (like noldus or anymaze) will suffice. These scenarious where background sutraction is sufficient is when we are tracking a single animal, and the 
body-parts we are intrested in are along the perimeters of the animal or body parts that can be heuristiallt infered by the perimeter and movement of the animal (e.g., snout, nose, centroid, lateral sides). We don't have to train supervised models to detect these body-parts, meaning no deep leraning, no annotations, and the tracking can be donew in minutes for many body-parts. 

EXAMPLE VIDEOS. 

> [!IMPORTANT]
> #### ANALYSIS SPEED
> 
> The speed of tracking is not only determined by your available hardware (i.e., you have a GPU, how many CPU cores you have, and how much RAM memory you have available). It is also determined by the resolution of your videos and the frame-rate of the videos.
> 
> For example, if your video resolutions are large (1000 x 1000), but you can visually detect the animals even at smaller resolutions (640 x 640), then consider downsampling your videos before performing blob detection.
> 
> Moreover, if you have collected your videos at a higher frame-rate (say 30 fps), but you really don't need to know where the animal is every 33ms of the video, then reduce the FPS of your videos to say 10 to get a location reading every 100ms.
>
> These preprocessing steps can greatly imporve the speed processing. You can perform these pre-processing steps in SimBA. See the SimBA video pre-processing documnetation [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md) or [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md)
> for how to change the resolution and frame-rate of your videos.
>
> #### GOOD BACKGROUND VIDEOS
>
> It is possible to select the same video as thos beeing analysed as the background reference. However, this may (i) slow down processing, and (ii) fail in situations where the animal is freezing/staying still in a single location throughout the video which will cause the code to mistake the animal as beckground. Best is to use a shorter snippets of the original videos,
> where the light levels are the same as during the actual video, as the beckground video. These snippets should have the same resolution as the original video, but is ideally much shorter, and can have reduced FPS.   


#### QUICK SETTINGS

You can use these options to set all the videos to the same values. To do this, select the values you want to use in the dropdown menu, and hit the appropriate <kbd>APPLY</kbd> button. This will update the relevant column in the `VIDEOS` table. 

**THRESHOLD**: How different the animal has to be from the background to be detected. Higher values will result in less likelihood to wrongly assume that parts the animal belong to the background, and increase the likelihood of wrongly assume that parts background belong to the animal.

[quick_set_threshold.webm](https://github.com/user-attachments/assets/b04b8f13-f0c7-489a-a3cd-a07b8a39dc68)

**SMOOTHING TIME (S)**: If set to None, no temporal smooth of the animal tracking points are performed. If not None, then SimBA performs Savitzky-Golay smoothing across the chosen sliding temporal window. 

[quicK-set_smoothing_time.webm](https://github.com/user-attachments/assets/3418e032-9dcc-4b81-8c56-d0daaae36c99)

**BUFFER SIZE (PIXELS)**: If set to None, the animals detected key-points will be placed right along the hull perimeter. We may want to "buffer the animals shape a little, to capture a larger area as the animal. Set how many pixels that you wish to buffer the animals geometry with. 
See further details below or [THIS](https://github.com/user-attachments/assets/a86a0d7b-35c6-4da7-b44c-4856d71fd861) visual example of expected results from different buffer sizes.

**CLOSING KERNEL SIZE (PX)**: Controls animal shape refinement. Larger values will merge seperate parts if the images detected as the foreground together into a single entity. The larger the value, the further apart the seperate parts of the foreground is allowed to be and still be merged into a single entity. This can be helpful if
(i) there are parts of the background/flooring/environment/arena that has the same color as the animal, or (ii) the animal color is in part same or similar as the background, or (iii) we want to make the animal more "blob" like. See the further detailes below or [THIS](https://github.com/user-attachments/assets/a86a0d7b-35c6-4da7-b44c-4856d71fd861) visual example on expected results from different closing kernel sizes.


#### RUN-TIME SETTINGS

**BACKGROUND DIRECTORY**: Choose a directory that contains background video examples of the videos. Once selected, hit the <kbd>APPLY</kbd> button. SimBA will automatically pair each background video with the videos in the table. Before clicking <kbd>APPLY</kbd>, make sure the chosen
directory contains videos with the same file names as the videos in the table. 

[bg_directory.webm](https://github.com/user-attachments/assets/2fcb2aae-0c27-472e-8d29-b80e7306a6ad)

> [!NOTE]
> The larger the background videos are ( higher resolution, higher FPS, longer time), the longer the processing will be. To speed up processing, it is best to have a short representative background video reference for each video to be processed. E.g., it could be videos representing the first 20-30s of the videos in the table, or copies of the videos in the table where the FPS has been much reduced.

**GPU**: Toggle "USE GPU" if available to accelerate some aspects of processing. This option is automatically disabled if an NVIDEA GPU is not available on your machine.

**CPU COUNT**: Choose the number of CPU cores you wish to use. Default is the maximum available on your machine. Higher values will result in faster processing but will require more RAM. If you are hitting memory related errors, you can try to decrease the `CPU COUNT` value.

**VERTICE COUNT**: Controls animal shape precission. For example, if set to `30`, then SimBA will output  a CSV file containing the position 30 body-part keypoints along the animals outer bounds for every frame in the video. In other words, every output tracking file will contain 30 `x` and 30 `y` columns representing the positions of 30 "body-parts" in every frame of the video. If you selected `30` in this dropdown, an example output CSV file will look like [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/example_vertice_cnt_30.csv). If you selected `60` in this dropdown, an example output CSV file will look like [THIS]().



**SAVE BACKGROUND VIDEOS**: If True, then black-and-white backgroiund subtracted videos (where the foreground is white and background is black), which SimBa uses to detect the animals location, whill be saved in your chosen output directory. If False, these temporaryly stored videos are removed. Default is True. 

**CLOSE ITERATIONS**: If `CLOSING KERNEL SIZE (PX)` isn't `None`, SimBA will smooth out the animals geometry N times. Higher iteration number will result in smoother, more "blob" like animals. Default: 3. 

**DUPLICATE INCLUSION ZONES**: If we have drawn `INCLUSION ZONES` on a video, we can duplicate those inclusion zones to all othe rvideos in the project. To duplicate a videos inclusion zones to all videos in the project, choose the video in the dropdown and hit the <kbd>APPLY</kbd> button. 



#### VIDEO TABLE

This table lists all video files found inside your defined video directory, with one row per video. For each row, there is a bunch of settings, allowing you control over the precis methods for how the animals location are detected in each video. You can also use the `QUICK SETTINGS` and `RUN-TIME SETTINGS` windows above to batch set these values on all videos. 

> [!NOTE]
> If all videos have been recorded in a standardized way, you will likely get away with using the `QUICK SETTINGS` frame above to bulk set the methods for all videos at once.

* **BACKGROUND REFERENCE**: Select the path to a video file serving as the background reference video for the specific file.

> [!NOTE]
> Filling out the background videos indivisually row-by-row can be tedious, and it is much recommended to use the `BACKGROUND DIRECTORY` in the `RUN-TIME SETTINGS` frame above to set all of the reference video paths at once.

* **THRESHOLD**: How different the animal has to be from the background to be detected. Higher values will result in fewer pixels being detected as the animal while more pixels beeing assigned to the background, while lower values will result in more pixels beeing assigned to the animal and fewer pisels beeing assigned to the background.

* **INCLUSION ZONES**: If we are having movement in the videos that are **not** performed by the animal (e.g., experimenters moving around along the perimeneters of the video, light intensities changes outside of the areana), we can tell the code about which parts of the arena the animal can be detected, and remove any detection outside of these defined zones.
To do this, click the <kbd>SET INCLUSION ZONES</kbd> button for a video, and use the ROI drawing interface to specify which areas of the image te animals can be detected. For a full tutorial for how to use this interface, see [THIS](https://github.com/sgoldenlab/simba/blob/master/docs/roi_tutorial_new_2025.md) documentation. As an example, in the video below,
I define a polygon called `MAZE` for the first video, and save it, making sure that the animal will only be detected inside the `MAZE` region of interest.

[inclusion_zones_define_downsampled.webm](https://github.com/user-attachments/assets/9e498c6d-cf2c-4084-ba42-2582305421bd)

> [!NOTE]
> 1) To save time, consider specifying the inclusion zones on only one video, and duplicating these inclusion zones on the rest of the videos using the `DUPLICATE INCLUSION ZONES` dropdown menu in the `RUN-TIME SETTING` menu above.
> 2) Only use inclusion zones if you see problems with the tracking. If you have confirmed visually that there are no problems with the tracking (which for me has been overwhelmingly the case) then no inclusion zones are required.

**SMOOTHING TIME**:  If set to None, no temporal smooth of the animal tracking points are performed. If not None, then SimBA performs Savitzky-Golay smoothing of the detected animal geometry using the selected smoothing time.

**BUFFER SIZE**: If set to None, the animals detected key-points will be placed right along the hull perimeter. We may want to "buffer the animals shape a little, to capture a larger area as the animal. Set how many pixels that you wish to buffer the animals geometry with. For a visual example about what "buffering" means, see below video. 

[buffer_ex.webm](https://github.com/user-attachments/assets/62cca260-5ab9-41ca-9cae-5369b3dc194c)


**CLOSE KERNAL SIZE**: Controls animal shape refinement. Larger values will merge seperate parts if the images detected as the foreground together into a single entity. The larger the value, the further apart the seperate parts of the foreground is allowed to be and still be merged into a single entity. See below video of an animal when increasingly larger kernal sizes are choosen. 

[close_kernal_example.webm](https://github.com/user-attachments/assets/a86a0d7b-35c6-4da7-b44c-4856d71fd861)

> [!NOTE]
> In the example above, from a elevated plus maze, the borders of teh open arms are black (same color as the animal) causing the tracking to fail and split the animal into two parts of the animal gets mistaken for the background as the animal performs a head dip. YThis is solved by smoothing (or "closing") the animals geometry.

**QUICK CHECK**: We can do a "quick check" if the animal can be discerned from the background reliably, using the chosed settings for each video. Hitting the <kbd>QUICK CHECK</kbd> button will remove the background and display the video frame by frame. If you can reliably see the animal in white, and teh background in black, you are good to go!

[quick_check.webm](https://github.com/user-attachments/assets/b83dfef6-746e-4fb7-ab50-fec64ad803be)

#### EXECUTE

The `EXECUTE` frame is duplicated and located both at the top right of the blob tracking interface, and at the top right. This frame conmtains two buttons. 

**REMOVE INCLUSION ZONES**: Hitting this button will remove all inclusion zones drawn on all videos. 

**RUN**: Once all has been set, click the <kbd>RUN</kbd> button. You can follow the progress in the main SImBA terminal and in the OS terminal use dto boot up SimBA. 

#### EXECTED OUTPUT

In the selected output directory, there will be s single CSV file for each of the input video. Each of these CSV files will contain one row for every frame in teh input video, and one column for each of the chosen number N of vertices (named `vertice_0_x, `vertice_0_y` ... `vertice_N_x`, `vertice_N_y`). This file will also contain 
six columns representing the anterior, posterior, center, left and right positions of the animal geometry. These columns are named `center_x`, `center_y`,	`nose_x`, `nose_y`, `tail_x`, `tail_y`, `left_x`, `left_y`,	`right_x`, `right_y`. For an example of the expected output CSV , see [THIS]() file. 

Moreover, if you set `SAVE BACKGROUND VIDEOS` to True above, this output directory will also contain the background subtracted mp4 file for each processed videos. 

Lastly, this directory will contain a `.json` file, named `blob_definitions.json`, which contains the settings which you used to process each of the video files, that will look something like [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/blob_definitions_example.json)




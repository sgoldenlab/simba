


> [!IMPORTANT]
> The speed of tracking is not only determined by your available hardware (i.e., you have a GPU, how many CPU cores you have, and how much RAM memory you have available). It is also determined by the resolution of your videos and the frame-rate of the videos.
> 
> For example, if your video resolutions are large (1000 x 1000), but you can visually detect the animals even at smaller resolutions (640 x 640), then consider downsampling your videos before performing blob detection.
> 
> Moreover, if you have collected your videos at a higher frame-rate (say 30 fps), but you really don't need to know where the animal is every 33ms of the video, then reduce the FPS of your videos to say 10 to get a location reading every 100ms.
>
> These preprocessing steps can greatly imporve the speed processing. You can perform these pre-processing steps in SimBA. See the SimBA video pre-processing documnetation [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md) or [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md)
> for how to change the resolution and frame-rate of your videos.


#### QUICK SETTINGS

**THRESHOLD**: How different the animal has to be from the background to be detected. Higher values will result in less likelihood to wrongly assume that parts the animal belong to the background, and increase the likelihood of wrongly assume that parts background belong to the animal. 

**CPU COUNT**: Choose the number of CPU cores you wish to use. Default is the maximum available on your machine. Higher values will result in faster processing but will require more RAM. If you are hitting memory related errors, try and decrease this value.

**VERTEX COUNT**: Controls animal shape precission. For example, if set to `30`, then SimBA will output the position 30 body-part keypoints along the animals outer bounds for every frame in the video. In other words, every output tracking file will contain 30 `x` and 30 `y` columns representing the positions of 30 "body-parts" in every frame of the video.

**GPU**: Toggle "USE GPU" if available to accelerate some aspects of processing. This option is automatically disabled if NVIDEA GPU is unavailable on your machine.

**BUFFER SIZE**: If set to None, the animals detected key-points will be placed right along the hull perimeter. We may want to "buffer the animals shape a little, to capture a larger area as the animal. Set how many pixels that you wish to buffer the animals geometry with.

**SMOOTHING TIME**: If set to None, no temporal smooth of the animal tracking points are performed. If not None, then SimBa performs Savitzky-Golay smoothing across the chosen sliding temporal window. 

**BACKGROUND DIRECTORY**: Choose a directory that contains background video examples of the videos. Once selected, hit the <kbd>APPLY</kbd> button. SimBA will automatically pair each background video with the videos in the table. 

> [!NOTE]
> The larger the background videos are ( higher resolution, higher FPS, longer time), the longer the processing will be. To speed up processing, it is best to have a short representative background video reference for each video to be processed. E.g., it could be videos representing the first 20-30s of the videos in the table, or copies of the videos in
> table where the FPS has been much reduced.

**INCLUSION ZONES**: We can provide regions-of-interest (ROI) drawings if where in the videos we want SimBA to look for the animals. If drawn, then detections that do not intersect the drawn regions will be discarded. 

**DUPLICATE INCLUSION ZONES**: We can provide regions-of-interest (ROI) drawings if where in the videos we want SimBA to look for the animals. If drawn, then detections that do not intersect the drawn regions will be discarded. 

**CLOSING KERNEL SIZE**: Controls animal shape refinement. Larger values will merge seperate parts if the images detected as the foreground together into a single entity. The larger the value, the further apart the seperate parts of the foreground is allowed to be and still be merged into a single entity. This can be particlarly helpful if
(i) there are parts of the background/flooring/environment/areana that has the same color as the animal, or (ii) the animal color is in part same or similar as the background, or (iii) we want to make the animal more "blob" like. 

**CLOSING ITERATIONS**: The number of times the closing kernel (if it isn't `None`) is applied. Higher values will results in the animals geometric shape becoming more round. 


#### VIDEO TABLE

This table lists all video files found inside your defined video directory, with one row per video. For each row, there is a bunch of settings, allowing you control over the methods for how the animals location are detected in each video. You can also use the `QUICK SETTINGS` and `RUN-TIME SETTINGS` windows above to batch set these values on
all videos. 

















> [!NOTE]
> This could also be done manually, video by video, by using the table.







### Background Directory
1. Click "BACKGROUND DIRECTORY"
2. Select folder containing reference videos
3. Click "APPLY" to set references




### Smoothing Time
- Range: 0.01-2.1 seconds
- Controls temporal smoothing
- "None" disables smoothing
- Affects tracking stability


### Buffer Size
- Range: 1-100 pixels
- Defines detection boundary
- Larger values capture wider movements
- "None" disables buffering


### GPU Settings
- Toggle "USE GPU" if available
- Accelerates processing
- 

Vertex Count: 10-500 (controls shape precision)


*
* Balance between speed and system load 

### CPU Core Count
- Set number of CPU cores to use
- 
- 



* A value between 0 and 100. 

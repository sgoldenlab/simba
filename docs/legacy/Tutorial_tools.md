
# How to use tools to process videos in SimBA

We have developed video and image processing tools and incorporated them into the overall SimBA pipeline. However, many of the tools are useful on their own or in "one-off" situations. To make these easily accessible, the tools have been incorporated into their own space within the GUI, and are described below.

>Note: As of 07/2023, many of the video processing tool windows in SimBA includes a `GPU` checkbox. If you have an NVIDEA GPU on your computer, you can tick this checkbox to perform the video processing function on the GPU, rather than CPU, and potentially save a significant amount of time. For an indication of the potential time-savings by function, see [THIS](https://github.com/sgoldenlab/simba/blob/master/docs/gpu_vs_cpu_video_processing_runtimes.md) table.


<p align="center">
  <img src="https://github.com/sgoldenlab/simba/blob/master/images/tools_img_2024.png" />
</p>

- [Clip videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#shorten-videos)
- [Clip multiple videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#clip-multiple-videos)
- [Clip multiple videos by frame numbers](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#clip-trim-multiple-videos-by-frame-numbers)
- [Clip (trim) single video by frame numbers](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#clip-trim-single-video-by-frame-numbers)
- [Clip video into multiple videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#clip-video-into-multiple-videos)
- [Crop videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#crop-video)
- [Circle crop videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#circle-crop)
- [Fixed Crop Videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#fixed-crop-videos)
- [Multi-crop videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#multi-crop-videos)
- [Downsample videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#downsample-video)
- [Get mm/ppx](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#get-coordinates)
- [Change formats](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#change-formats)
- [Change fps](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#change-fps-frame-rate-of-videos)
- [Create path plots](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#create-path-plots)
- [CLAHE enhance video](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#clahe)
- [Superimpose frame numbers on video](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#add-frame-numbers)
- [Convert to greyscale](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#convert-to-grayscale)
- [Merge frames to video](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#merge-images-to-video)
- [Generate gifs](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#generate-gifs)
- [Extract frames](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#extract-frames)
- [Print model info](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#print-mode-info)
- [Convert .seq to .mp4](https://github.com/sgoldenlab/simba/blob/simba_JJ_branch/docs/Tutorial_tools.md#convert--seq-files-to-mp4-files)
- [Re-order pose-estimation tracking data](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#re-organize-tracking-data)
- [Remove body-parts from tracking data](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#re-organize-tracking-data)
- [Visualize pose estimation in folder](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#visualize-pose-estimation-in-folder)
- [Concatenate two videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#concatenate-two-videos)
- [Concatenate multiple videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#concatenate-multiple-videos)
- [Extract ROI definitions to human-readable format](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#extract-roi-definitions-to-human-readable-format)
- [Temporally join videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#temporal-join-videos)
- [Extract project annotation counts](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#extract-project-annotation-counts)
- [Remove video backgrounds](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#remove-background-from-all-videos-in-a-directory)

## Shorten Videos
This is a tool used to trim video lengths. The tool contains two different methods: 
**Method 1** and **Method 2**. 

<img src="https://github.com/sgoldenlab/simba/blob/master/images/shortenvideos.PNG" width="330" height="287" />

> Note: To trim multiple videos at specified time stamps, you can use the [batch pre-processing](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md) tools in SimBA or
> the [clip multiple videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#clip-multiple-videos) function.

### Method 1 
Use Method 1 to trim both the beginning and the end of the video. 

Let's say we have a 2-minute-long video, and we want to get rid of the first 10 seconds, and the last 5 seconds.
We would start our video on at `00:00:10`, and end the video at `00:01:55`.

1. First, click on `Browse File` to select the video to trim.

2. Enter the start time in the `Start at:` entry box, in this case it will be *00:00:10*

3. Enter the end time in `End at:` entry box, in this case it will be *00:01:55*. The settings should look like the image below.

![alt-text-1](/images/shortenvideos1.PNG "shortenvideos1")

4. Click `Cut Video` to trim the video and a new shorter video will be generated. The new shorter video will have a name of *Name of original video* + *_shorten* and will be located in the same folder as the original video.

![alt-text-1](/images/shortenvideos2.PNG "shortenvideos2")


### Method 2
Method 2 cuts of the beginning of the video. 

Let's say we have a 2 minute long video and we want to get rid of the first 20 seconds from the start of the video.

1. Enter the amount of time that needs to be trimmed from the start of the video in `Seconds:`, in this case it will be *20*

![alt-text-1](/images/shortenvideos3.PNG "shortenvideos3")

2. Click `Cut Video` to trim video and a new shorten video will be generated, the new video will have a name of *Name of original video* + *_shorten* and will be located in the same folder as the original video.

## Clip video into multiple videos
This tool can help users to cut the videos into multiple clips/section.

1. Click on the `Clip video into multiple videos` from the `Tools` section.

![](/images/clipmulti1.png)

2. Select the video that you want to split/clip by clicking `Browse File`

![](/images/clipmulti2.png)

3. Put in the number of output clips/sections that you want in `# of clips`, and click `Confirm`. Please note that if you put in the wrong number the first time, you can re-enter the number and click `Confirm` to change the table.

![](/images/clipmulti3.png)

4. Then enter the `Start Time` and `Stop Time` in the following format HH:MM:SS. For example, for a minute and 20 seconds it will be 00:01:20.

5. Once the table has been filled, click `Clip video` and the video will be output on the same folder path/ directory of your original video.

## Crop Video
This is a tool to crop videos.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/cropvideo.PNG" width="232" height="232" />

1. First, click on `Browse File` to select a video to crop.

2. Click on `Crop Video` and the following window `Select ROI` will pop up.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/cropvideo2.PNG" width="363" height="546" />

3. Use your mouse to *Left Click* on the video and drag the rectangle bounding box to contain the area of the video you wish to keep. *You can always left click and drag on the video to recrop your video*

<img src="https://github.com/sgoldenlab/simba/blob/master/images/cropvideo3.PNG" width="363" height="546" />

4. Hit `Enter` **twice** and SimBA will crop the video. The new video will have a name of *Name of original video* + *_cropped* and will be located in the same folder as the original video.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/cropvideo4.PNG" width="900" height="100" />

## Fixed Crop Videos
This tool allows the user to crop once and apply the same dimension to the rest of the videos in the folder.

![](/images/fixcropvid.PNG)

1. Under `Video directory`, select the input video folder by clicking `Browse Folder`.

2. Then, select an output folder.

3. Click on `Confirm` and an image will pop up, use your mouse to *Left Click* on the video and drag the rectangle bounding box to contain the area of the video you wish to keep. *You can always left click and drag on the video to recrop your video*

4. Hit `Enter` **twice** and SimBA will crop all the videos in the folder with the same coordinate. The new videos will have a name of *Name of original video* + *_cropped* and will be located in the output folder.

## Multi-crop videos

This is a tool to used to multi-crop videos. For example, if you recorded four different environments with a single camera, you can use this tool to split single recordings into 4 different videos. This tool operates on all videos in a folder that is defined by the user. The user is required to draw the number of defined rectangles on **each** of the videos in the specified folder.  

1. First, click on Multi-crop and the following menu will pop-open:

![alt-text-1](https://github.com/sgoldenlab/simba/blob/master/images/Multi_crop_1.JPG "Multi_crop_1")

2. Next to *Video Folder*, click on `Browse Folder` and select a folder containing the videos you which to multi-crop. 

3. Next to *Output folder*, click on `Browse Folder` and select a folder that should house the cropped output videos. 

4. Next to *Video type*, type the file format of yout input videos (e.g., mp4, avi etc). 

5. Next to *# of crop*, type in the number of cropped videos you wich to generate from each single input video (e.g., 4). Click on **Crop** to proceed. When you click on **Crop**, the first frame of the first video in the specified folder will be displayed, and the name of the video and rectangle number is printed overlaid: 

![alt-text-1](https://github.com/sgoldenlab/simba/blob/master/images/Multi_crop_example.gif "Multi_crop_1")
6. Left click the mouse and drag from the top left corner to the bottom right corner of the first video you wish to generate. When finished with the first video, press `Enter`. Repeat this step for the next videos you wish to generate from Video 1. Once Video1 is complete, repeat these steps for all the videos in the user-specified *Video Folder*.

7. The cropped output videos will be located in the user-defined *Output folder* as defined in Step 3. 


## Downsample video
This is a tool to downsample a video into smaller size and reduce the resolution of the video.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/downsamplevid.PNG" width="400" height="300" />

The downsample video tool has two options: **Customize Resolution** and **Default Resolution**.

### Customize Resolution
Use this tool to downsample video into any height and width.

1. First, click on `Browse File` to select a video to downsample.

2. Then, enter any values in the `Height` and `Width` entry boxes.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/downsamplevid2.PNG" width="400" height="300" />

3. Click on `Downsample to custom resolution` to downsample the video. The new video will have a name of *Name of original video* + *_downsampled* and will be located in the same folder as the original video.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/downsamplevid3.PNG" width="900" height="100" />

### Default resolution 
This tool allows the user to downsample a video quickly.

1. First, click on `Browse File` to select a video to downsample.

2. Tick on one of the resoltion options. 

3. Click on `Downsample to default resolution` and the video will downsample into the selected resolution. The video will be located in the same folder as the original video.


## Get Coordinates (calibrate distance)
This tool is to get the length (millimeter) per pixel of a video.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/getcoordtool.PNG" width="300" height="200" />

Let's say we want to find out the metric length per pixel of a video of a mouse in a cage, and we know the width of cage is 130 millimeters (it's a tight one).

1. First, click on `Browse File` to select a video.

2. Enter *130* in the `Known length in real life(mm)` entry box.

3. Click on `Get Distance`, and the following window will pop up.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/getcoordtool2.PNG" width="500" height="500" />

>*Note*: When the frame is displayed, it may not be shown at the correct aspect ratio. To fix this, drag the window corner to the correct aspect ratio. 

4. Use your mouse to double *Left click* at the left side of the cage and double *Left click* again on the right side of the cage. These are the known distance of 130 mm.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/getcoordtool3.PNG" width="500" height="500" />

>**Note:** You can double click any point again to change the location of the point.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/getcoord.gif" width="500" height="500" />

5. Once two points are selected, hit `Esc` button. The millimeter per pixel value is printed in the main SimBA interface.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/getcoordtool4.PNG" width="400" height="250" />

## Change formats
This menu includes **Change image formats** and **Change video formats**

### Change image formats
This tool allows the user to select a folder containing multiple images and convert the formats.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/changeimageformat.PNG" width="200" height="200" />

1. Click on `Browse Folder` to select a folder that contains multiple images.

2. Choose the original format of the images in the selected folder.

3. Choose the desired output image format.

4. Click on `Convert image file format`.

### Change video format
This tool allows the user to convert the file format of a single or multiple videos.  

<img src="https://github.com/sgoldenlab/simba/blob/master/images/changevideoformat.PNG" width="400" height="400" />

#### Convert multiple video file formats

1. Click on `Browse Folder` to select the directory that contains the videos that you want to convert.

2. Enter the original file format (eg: mp4, flv, avi etc.) in the `Input format` entry box. **Note: do not put dots ('.') in the file format name (eg: mp4 or flv, etc)**.

3. Enter the desired output format in the `Output format` entry box .

4. Click on `Convert multiple videos`.

#### Convert single video file format

1. Click on `Browse File` to select a video to convert.

2. Choose one of the following `Convert .avi to .mp4` or `Convert mp4 into Powerpoint supported format`

3. Click on `Convert video format`.


## [CLAHE enhance video](https://docs.opencv.org/3.4/d5/daf/tutorial_py_histogram_equalization.html)

1. Click on `Browse File` and select a video file.

2. Click `Apply CLAHE`. The new video will have a name of *CLAHE_* *Name of original video*. The new file will be in a **.avi** format will be located in the same folder as the original video.

## Superimpose frame numbers on video
This tool creates a video with the frame numbers printed on top of the video. 

1. Click on `Superimpose frame numbers on video` and a new window will pop up.

2. Select a video and click on `Open`.

3. The new version of the video will be created with the name *Name of original video* + *_frame_no* and will be located in the same folder as the original video.

## Convert to grayscale

1. Click on `Convert to grayscale` and a new window will pop up.

2. Select video and click `Open`.

3. The new greyscale version of the video will be created and have the name *Name of original video* + *_grayscale*. The new video will be located in the same folder as the original video.

## Change FPS (frame-rate) of videos

### Change FPS of a single video

1. Click on `Tools` -> `Change fps` -> `Change fps for single video` and a new pop up window is displayed. 

2. Click on `Browse File` and select the video you want to change the FPS for. 

3. In the `Output FPS` entry-box, enter the FPS of the new video as a number (e.g., `15`). 

4. Click the `Convert` button. A new video is saved in the directory of the input video with the `fps_15` filename suffix.

### Change FPS of multiple videos

1. Click on `Tools` -> `Change fps` -> `Change fps for multiple videos` and a new pop up window is displayed. 

2. Click on `Browse Folder` and select the directory containing the videos you want to change the FPS for. 

3. In the `Output FPS` entry-box, enter the FPS of the new videos as a number (e.g., `15`). 

4. Click the `Convert` button. New videos are saved in the directory of the input videos with the `fps_15` filename suffix.

## Create path plots

Use this tool to create a path plot videos from raw pose-estimation data files. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/raw_line_plot.png" />
</p>

1. Click on `Browse File` next to `VIDEO PATH` and select the video file used to create the pose-estimation data (this is needed get and apply the correct fps and resolution of the output video).

2. Click on `Browse File` next to `DATA PATH` and select the raw output pose-estimation data from the video file (e.g., a DLC-generated H5 or CSV file). 

3. In the `BODY-PART` entry-box, enter the body-part (e.g., Nose) you which to represent the location of the animal in the path plot video. 

4. In the `BACKGROUND COLOR` drop-down, select the color you which to represent the background in the path plot video. 

5. In the `LINE COLOR` drop-down, select the color you which to represent the path of your animal in the path plot video (NOTE: make sure background and line colors are not identical). 

6. In the `LINE THICKNESS` drop-down, select the thickness of the lines you which to represent the path of your animal in the path plot video.

7. In the `CIRCLE SIZE` drop-down, select the size of the circle which represent the current location of the animal in the path plot video. 

8. Click on `CREATE PATH PLOT VIDEO`. A new path plot video is created in the same directory as the input video with the `line_plot` file name suffix.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/ez_line_plot_gif.gif" />
</p>


## Merge images to video

1. Click on `Browse Folder` to select a folder containing multiple frames.

2. Enter the input image format of the frames. Eg: if the image name is *Image001.PNG*, enter *PNG* in the `Image format` entry box.

3. Enter the desire output video format. Eg: if the video should be in *.mp4* file format, enter *mp4* in the `Video format` entry box.

4. Enter the desire frames per second in the `fps` entry box.

5. Enter the desired [bitrate](https://help.encoding.com/knowledge-base/article/understanding-bitrates-in-video-files/) for the video in the `Bitrate` entry box.

6. Click on `Merge Images` to create the video. 

## Generate gifs

1. Click on `Browse File` and select a video to convert to a GIF.

2. Enter the starting time of the GIF from the video in the `Start times(s)` entry box.

3. Enter the duration of the GIF in the `Duration(s)` entry box.

4. Enter the size of the GIF in the `Width` entry box. The output GIF will be scale automatically.

5. Click on `Generate gif` to create the gif. 

## Extract Frames
The Extract frames menu has two options: **Extract defined frames**, **Extract frames**, and **Extract frames from seq files**.

### Extract defined Frames
This tool allows users to extract frames from a video by inputting specific start- and end-frame numbers. This is useful if you want to extract a subset of frames from a larger video, without first needing to generate a new video of the desired length.

1. Click `Browse File` to select a video.

2. Enter the starting frame number in the `Start Frame` entry box.

3. Enter the ending frame number in the `End Frame` entry box.

4. Click on `Extract Frames` to extract the frames from the `Start Frame` to the `End Frame`.

5. A folder with the video name will be generated and the all the extracted frames will be located in the folder. The frames will be in *.png* file format. 

### Extract frames
Use this tool to extract every frame from a single video or multiple videos.

#### Single video

1. Click on `Browse File` to select a video file. 

2. Click on `Extract Frames(Single video)` to extract every frame from the video.

3. A folder with the video name will be generated and the all the extracted frames will be located in the folder. The frames will be in *.png* file format. 

#### Multiple videos

1. Click on `Browse Folder` to select the folder with videos. 

2. Click on `Extract Frames(Multiple videos)` to extract every frame from the video.

3. Folders with the video name will be generated and the all the extracted frames will be located in the folders. The frames will be in *.png* file format. 

### Extract frames from seq files
Use this tool to extract all the frames from a video in **seq** file format.

1. Click on `Browse File` to select a video.

2. Click on `Extract All Frames` to extract all the frames from the video.

3. A folder with the video name will be generated and the all the extracted frames will be located in the folder. The frames will be in *.png* file format.

### Convert . seq files to .mp4 files
Use this tool to convert .seq files to .mp4 files.

1. Click on `Tools`, then `Change formats`, and click on `Change .seq to .mp4`.

2. A window will pop up and you can then navigate and select the video folder that contains the mp4's. 

3. The conversion progress can be followed through the progress bar printed in the terminal window.


### Re-organize tracking data
Use this tool to re-order the pose-estimation tracking data of multiple files in a folder. For example, you may have some pose-estimation body-part tracking files where the `Animal_1_nose` is body-part number 1, and some other pose-estimation body-part tracking files where `Animal_1_tail_base` is body-part number 1. Now you want to re-order the  data so that all files contains the same order of tracked body-parts.

1. Click on `Tools`, then `Re-organize Tracking Data`. 
2. In the entry box `Data Folder`, select the directory containing your pose-estimation tracking files. 
3. Select your pose-estimation tool in the `Tracking Tool` drop-down menu. 
4. Select the file-format of your tracking data in the `File Type` drop-down menu. 
5. Click `Confirm` and the following menu below will pop open. If you have one animal in your tracking data, the menu on the left will show. If you have multiple animals in your tracking data, the menu on the right will show. The image on the right contains an extra column compared to the image on the left, representing the Animal name. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/reorg_1.png" />
</p>

6. The `Current Order` sub-menu on represents the column order of the input files. The `New Order` sub-menu on represents the order which the body-parts should be re-ordered to. Use the drop-down menus to select the new order of the body-parts, then click on `Run re-organization`. 

7. Your new, re-organized files will be saved in a date-time stamped folder inside the `Data Folder` selected in Step 2. The new folder will be named something like `Reorganized_bp_20210726111127`. 


### Remove body-parts from tracking data
Use this tool to delete user-specified body-parts from pose-estimation tracking data for all files in a folder. For example, you may have pose-estimation body-part tracking files for 16 body-parts, but now you want to get rid of the data for 2 body-parts and keep the other 14. 

1. Click on `Tools`, then `Remove body-parts from tracking data`.
2. In the entry box `Data Folder`, select the directory containing your pose-estimation tracking files. 
3. Select your pose-estimation tool in the `Tracking Tool` drop-down menu. 
4. Select the file-format of your tracking data in the `File Type` drop-down menu. 
5. Select how many body-parts you like to remove from the pose-estimation tracking data files.
6. Click `Confirm` and the following menu below will pop open (Note: if you have multiple animals in your pose-estimation files, you will also see a dropdown menu named `Animal`.) Select the body-parts you would like to remove, and click `Run Removal`. 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/remove_bps.png" />
</p>

7. Your new files (with the removed body-parts) will be saved in a date-time stamped folder inside the `Data Folder` selected in Step 2. The new folder will be named something like `Reorganized_bp_20210726111127`. You can now go ahead and import the files into your SimBA project. 

### Visualize pose estimation in folder
Use this tool to visualize the pose-estimation of all the files inside a SimBA project directory. This tool can be useful when you have [interpolated  and/or smoothened](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-import-dlc-tracking-data) and or [performed outlier correction](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-4-outlier-correction) on your pose-estimation data at import, and now you want to visualize the results of that interpolation and smoothing inside the . 

1. Click on `Tools`, then `Visualize pose-estimation in folder...` which brings up the following pop-up:

<img width="632" height="547" alt="image" src="https://github.com/user-attachments/assets/82bd4876-c175-4b3a-af8b-a1a0fb930a85" />

With the following OPTIONS:

* **KEY-POINT SIZES** – Options: AUTO, 1 through 100. Sets the pose-estimated key-pointradius in the visualization; AUTO derives suitable size automatically from video resolution.

* **VIDEO SECTION (SECONDS)** – Options: ENTIRE VIDEO(S), plus 10, 20, …, 200. Restricts rendering to a clip length of each video in seconds, or produces the full video.

* **CPU COUNT** – Options: integers 1 up to detected cores minus one. Controls single-process vs. multiprocessing plotters. The more cores the faster the video is created but with taxing GPU RAM more.

*  **USE GPU** – Options: TRUE, FALSE. Enables GPU-assisted concatenation when multiprocessing.

*  **INCLUDE BOUNDING BOX** – Options: TRUE, FALSE. Adds rectangular bounding boxes encompassing each animals body-parts to the rendered pose output.

*  **SAVE DIRECTORY** – Folder chooser (defaults to the desktop). Destination for annotated videos.

*  **NUMBER OF ANIMALS** – Tells SimBA how many tracked individuals to expect.

*  **ANIMAL n COLOR PALETTE** – Options: Assigns color schemes per animal. AUTO and SImBA will pick a random categorical palette. 

*  **SELECT DATA FILE (CSV/PARQUET)** – File picker limited to CSV/Parquet. Runs pose visualization on a single file when you click RUN.

*  **SELECT DATA DIRECTORY (CSV/PARQUET)** – Folder picker. Batch-runs pose visualization across all supported files in the selected directory when you click RUN.

2. Click on the `RUN`. You can follow the progress in the main SimBA terminal window. The videos are stored in the selected **SAVE DIRECTORY** folder. 

### Temporal join videos
Use this tool to join a single recording represented by multiple video files into a video file. 

1. Click on `Tools`, then `Temporal join videos`.
2. In the `INPUT DIRECTORY`, click browse and select the directory holding your videos. 
>NOTE: The video files in the directory have to be sequantuall and numerically named in order for SimBA to know the order. E.g., `1.mp4`, `2.mp4`... 
3. In the `INPUT FORMAT` dropdown, select the file format of the input videos. 
4. Click run. A new video file will be created inside the directory selected in Step 2 called `concatenated.mp4`. 

### Rotate videos

At times, the camera made have been tilted during a recording which prevents accurate bounding-box cropping and/or messes with our classifiers and now we want to fix this. 

1. To rotate videos, click `Tools`, then `Rotate videos`.
2. In `Save directory`, select the directory where you want to store your rotated videos. 
3. To rotate several videos, select the directory where the videos live in `Rotate videos in directory`. Alternatively, to rotate a single video, select the path to the video file in `Video path`. 
4. Once selected, click the `Run` button and the interface in the video below pops up. Use the left and right keyboard buttons to rotate the video to the left and right. Once happy, use the `ESC` button to rotate and save the video in the directory chosen in Step 2. 


https://user-images.githubusercontent.com/34761092/234957017-6b3de496-f485-47ce-b66c-00b3d841d7ab.mp4

### Concatenate two videos
Use this tool to concatenate two videos into a single video. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/concat_1.png" />
</p>

1. In the `First video path`, click on `Browse` and select the first video. 
2. In the `Second video path`, click on `Browse` and select the second video. 
3. In the `Resolution` drop-down menu, select the resolution of the output video.
> Note 1: Select `Video 1` in the `Resolution` drop-down menu to use the resolution of Video 1 as the resolution of the output video.
> Note 2: Select `Video 2` in the `Resolution` drop-down menu to use the resolution of Video 2 as the resolution of the output video.
4. To vertically concatenate the videos, click the `Vertical concatenation` radio-button. To horizontally concatenate the videos, click the `Horizontal concatenation` radio-button. The resolution choosen in Step 3 refers to the height or width depending on which type of concatenation (vertical vs horizontal).
5. Click `Run` to perform the concatenation. The file-path of the output video will be printed in the main SimBA terminal window (the file will be located in the same directory as the `First video path` with the `_concat.mp4` suffix. 

### Concatenate multiple videos
Use this tool to concatenate a user-defined number of videos into a single video mosaic of videos. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/frm_merge.png" />
</p> 

Begin by selecting how many videos you want to concatenate together in the `VIDEOS #` drop-down menu and click `SELECT`. A table, with one row representing each of the videos, will show up titled `VIDEO PATHS`. Here, click the `BROWSE FILE` button and select the videos that you want to merge into a single video. 

Next, in the `JOIN TYPE` sub-menu, we need to select how to join the videos together, and we have 4 options:

* MOSAIC: Creates two rows with half of your choosen videos in each row. If you have an unequal number of videos you want to concatenate, then the bottom row will get an additional blank space. 
* VERTICAL: Creates a single column concatenation with the selected videos. 
* HORIZONTAL: Creates a single row concatenation with the selected videos. 
* MIXED MOSAIC: First creates two rows with half of your choosen videos in each row. The video selected in the `Video 1` path is concatenated to the left of the two rows. 

Finally, we need to choose the resolution of the videos in the `Resolution width` and the `Resolution height` drop-down videos. **If choosing the MOSAIC, , VERTICAL, or horizontal join type, this is the resolution of each panel video in the output video. If choosing MIXED MOSAIC, then this is the resolution of the smaller videos in the panel (to the right)**. 

After clicking `RUN`, you can follow the progress in the main SimBA terminal and the OS terminal. Once complete, a new output video with a date-time stamp in the filename is saved in the the same directory as the directory of the video selected as `Video 1`.

### Print mode info
Use this tool to get the model information (e.g., number of features, number of trees used to create a model) from a model `.sav` file in SimBA. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/model_info_1.png" />
</p>

1. Next to the `Model path` entry box, click `Browse` and navigate to a model classifier that you have created in SimBA (e.g., select the `MyClassifier.sav` file. 

2. Next, click `PRINT MODEL INFO` and you should see the following information printed in the main SimBA terminal. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/model_info_2.png" />
</p>

This information tells you how many trees the classifier uses, how many features the classifier expects, further model hyperparameter settings, and when the file was created etc. 

### Extract ROI definitions to human-readable format

Your drawn ROI definitions (their locations, sizes, vertices, colors etc.) are saved inside the `project_folder/logs/measures/ROI_definitions.h5` file in your SimBA project. Sometimes, we may want to extract this data into human-readable format. To do this, click the `Convert ROI definitions` in the tools menu:

<img src="https://github.com/sgoldenlab/simba/blob/master/images/roi_definitions_tools.png" width="900" />

* In the `ROI DEFINITIONS PATH (H5)` file-selection box, click `Browse` and select the path to the `project_folder/logs/measures/ROI_definitions.h5` for which you want to extract the.
* In the `SAVE DIRECTORY` folder-select box, select the directory where you want to store the output files.
* Once filled in, click `RUN`.

Once complete, check the `SAVE DIRECTORY` for files with the names formatted as `polygons_datetimestamp.csv`, `rectangles_datetimestamp.csv`, and `circles_datetimestamp.csv`. If no circles, rectangles, or polygons exist in the `ROI_definitions.h5`, the respective CSV file will not be created. 

For some examples of expected outputs, check [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/polygons_20230809094913.csv) file for polygons, [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/rectangles_20230809094913.csv) file for rectangles, and [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/circles_20230809102338.csv) file for circles.

### Extract project annotation counts

If you have annotated (or appeneded annotations to) a bunch of videos in a SimBA project, you may want a count of how many frames, and how many seconds, you have annotated each behavior as present and absent in each video in the entire project. To get this, 

1) Click the `Count labels in project` button in the Tools menu.
2) Next, select the `project_config.ini` file belonging to the project in which you want to count the annotations within.
3) Click the RUN button.

You can follow the progress in the main SimBA window. Once complete, the annotation counts are saved in a CSV within the `project_folder/logs` directory in with the annotation_counts_20240110185525.csv filename format. For an example of the expected output, see [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/print_project_annotation_counts_example.csv) file. 

>Note: Each of the videos inside the `project_folder/csv/targets_inserted` directory also **have to be&& represented in the `project_folder/logs/video_info.csv` for this to work. This is in order for SimBA to know the FPS of the videos and accurately calculate the number of seconds that each behavior has been annotated as present and absent in each of the videos. 

### Circle crop

Sometimes, we may want to crop videos according to user-defined circles rather than [user-defined rectangles](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#crop-video). To do this, 

1) Click the 'Crop videos (circles)' button in the Tools menu, and you should see the pop-up below. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/crop_circles_3.png" />
</p>

2). To circle crop a single video, select the video path in the top menu and click the <kbd>Crop</kbd> button. 

3). The first frame of the video should pop open. Click and hold-down the left mouse-button at the center of your to-be cropped region. The, drag the mouse towards teh outer boundary of the circle regions. Finally, **without letting go of the left mouse button**, hit the ESC, SPACE or Q button on your keyboard.  See the videos below for expected input and output. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/circle_crop_1.gif" />
</p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/circle_crop_2.gif" />
</p>

4). If you want to circle crop multiple videos, and your camera location / arena location is static across multiple recordings,
then you may just want to define the cropped circle location in one video and crop all videos in a directory using the circle defined one. For this, 
use the ``Fixed circle coordinates crop for multiple videos`` submenu. 

5) Define the directory where your videos are stored using the ``Video directory`` browse button. 
6) Use the ``Output directory`` browse button to specify the location directory where the cropped videos should be stored. 
7) Hit the ``Crop Videos`` button. You can follow the progress in the main SimBA terminal window. 

### Clip (trim) single video by frame numbers

Sometimes, we may want to clip videos between specified frame numbers instead of specified [time-stamps](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#shorten-videos)
To do this, click the **Clip single video by frame number** button in the Tools menu, and you should see the follow po-up menu:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/frame_id_trim_single_video.png" />
</p>

i) Browse and select for the video that you want to clip by frame number.

ii) In the `START FRAME` entry-box, select the frame where the trimmed video should start.  In the `END FRAME` entry-box, select the frame where the trimmed video should end.

iii) Click the `RUN` button. You can follow the progress in the main SimBA terminal and the opertaing system terminal. 

iv) Once complete, a new video will be stored in the same directory as in the input video, with a suffix in the file-name representing the clipped start and end point frame numbers. 


### Clip (trim) multiple videos by frame numbers

Sometimes, we may want to clip multiple videos between specified frame numbers instead of specified [time-stamps](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#shorten-videos)
To do this, click the **Clip multiple videos by frame number** button in the Tools menu. 

i) First, you will be asked to select a directory containing input videos, as well as a directory where the clipped videos should be stored. Select the directories and click ``RUN``, and you should see the following pop-up:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/frame_id_trim_multiple_videos.png" />
</p>

ii) The table will contain one row per video in the input directory. For reference, under the `TOTAL FRAMES` heading, is the total number of frames contained in each video file.

iii) In the `START FRAME` entry-box, select the frame where the trimmed video should start.  In the `END FRAME` entry-box, select the frame where the trimmed video should end.

iv) Click the `RUN` button. You can follow the progress in the main SimBA terminal and the operating system terminal. 

v) Once complete, new videos will be stored in your selected output directory.

### Clip multiple videos

Use this menu to clip multiple videos between specified time-stamps.

>Note I: If you instead want to clip videos between specific frame numbers rather than time-stamps, use the [Clip (trim) multiple videos by frame numbers](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#clip-trim-multiple-videos-by-frame-numbers) or [clip (trim) single video by frame numbers](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#clip-trim-single-video-by-frame-numbers) functions. 

>Note II: To clip multiple videos by specified time-stamps, you can also use the [batch pre-processing](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md) tools in SimBA.

i) First, you will be asked to select a directory containing input videos, as well as a directory where the clipped videos should be stored. Select the directories and click ``RUN``, and you should see the following pop-up:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/clip_multiple_videos_tools.png" />
</p>

ii) The table will contain one row per video in the input directory. For reference, under the `VIDEO LENGTH` heading, is the length of each video in HH:MM:SS format.

iii) In the `START TIME` entry-box, enter the time when the new video should start. E.g., enter `00:02:01` to make the new video start after two minutes and one second of the input video.  In the `END TIME` entry-box, enter the time when the new video should end.  E.g., enter `00:04:21` to make the new video end after four minutes and twenty-one second of the input video.

iv) Click the `RUN` button. You can follow the progress in the main SimBA terminal and the operating system terminal. 

v) Once complete, new videos will be stored in your selected output directory.


### REMOVE BACKGROUND FROM A SINGLE VIDEO 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/bg_removal_single.webp" />
</p>

* In `VIDEO PATH`, click <kbd>BROWSE</kbd> and select the file you which to remove the background from.
* OPTIONAL: In `BACKGROUND REFERENCE VIDEO PATH`, click <kbd>BROWSE</kbd> and select a video file that represents the background (has no animals visible but otherwise representative / near identical copy of video selected in `VIDEO PATH`)
> [!NOTE]
> `BACKGROUND REFERENCE VIDEO PATH` can often be left blank (do not select any video). If no video is selected, the background with be computed from the `VIDEO PATH`.
* `BACKGROUND COLOR`: The color of the pixels determined to be part of the background. Default: White.
* `BACKGROUND COLOR`: The color of the pixels determined to be part of the background. Default: Original (the colors of the the original video are retained)
* `BACKGROUND THRESHOLD`: Percent deviation from the average image required to select pixel as foreground. Default: 30.  
* `COMPUTE BACKGROUND FROM ENTIRE VIDEO`: The background can be computed from the entire video, of a segment of the video. If you want to compute it from teh entire video, check the `COMPUTE BACKGROUND FROM ENTIRE VIDEO` checkbox.
   - `BACKGROUND VIDEO START (FRAME # OR TIME)`: If **un-checking** the `COMPUTE BACKGROUND FROM ENTIRE VIDEO`, we need to select a segment of the background video which we will use to compute the background. Enter the **START** time (e.g., `00:00:00`) or or start frame (e.g., `0`) of the segment here.
   - `BACKGROUND VIDEO END (FRAME # OR TIME)`: If **un-checking** the `COMPUTE BACKGROUND FROM ENTIRE VIDEO`, we need to select a segment of the background video which we will use to compute the background. Enter the **END** time (e.g., `00:00:20`) or or start frame (e.g., `1200`) of the segment here.
* `MULTIPROCESS VIDEO (FASTER): To use multiprocessing, and process each image of the video in parallel on each of your available CPU cores, check this box and select how many cores you want to use in the `CPU cores` dropdown.

Once complete, hit the <kbd>RUN</kbd> button. You can follow the progress in the main SimBA terminal window and the operating system terminal window. The background removed video will be saved in the same directory as the `VIDEO PATH` with the `_bg_subtracted` suffix. 

### REMOVE BACKGROUND FROM ALL VIDEOS IN A DIRECTORY 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/bg_removal_directory.webp" />
</p>

* In `VIDEO DIRECTORY`, click <kbd>BROWSE</kbd> and select the directory containing videos you want to remove the background from. 
* OPTIONAL: In `BACKGROUND VIDEO DIRECTORY`, click <kbd>BROWSE</kbd> and select a directory containing videos mirring the filenames of the videos in the `VIDEO DIRECTORY` but have animals visible but otherwise representative / near identical copy of video selected in `VIDEO DIRECTORY`.
> [!NOTE]
> `BACKGROUND VIDEO DIRECTORY` can most often be left blank (do not select any directory). If no directory is selected, the background with be computed from the  videos in `VIDEO DIRECTORY` themselves.
* `BACKGROUND COLOR`: The color of the pixels determined to be part of the background. Default: White.
* `BACKGROUND COLOR`: The color of the pixels determined to be part of the background. Default: Original (the colors of the the original video are retained)
* `BACKGROUND THRESHOLD`: Percent deviation from the average image required to select pixel as foreground. Default: 30.  
* `COMPUTE BACKGROUND FROM ENTIRE VIDEO`: The background can be computed from the entire videos, of a segment of the videos. If you want to compute it from the entire videos, check the `COMPUTE BACKGROUND FROM ENTIRE VIDEO` checkbox.
   - `BACKGROUND VIDEO START (FRAME # OR TIME)`: If **un-checking** the `COMPUTE BACKGROUND FROM ENTIRE VIDEO`, we need to select a segments of the background videos which we will use to compute the background. Enter the **START** time (e.g., `00:00:00`) or or start frame (e.g., `0`) of the segment here.
   - `BACKGROUND VIDEO END (FRAME # OR TIME)`: If **un-checking** the `COMPUTE BACKGROUND FROM ENTIRE VIDEO`, we need to select a segments of the background videos which we will use to compute the background. Enter the **END** time (e.g., `00:00:20`) or or start frame (e.g., `1200`) of the segment here.
* `MULTIPROCESS VIDEO (FASTER): To use multiprocessing, and process each image of the video in parallel on each of your available CPU cores, check this box and select how many cores you want to use in the `CPU cores` dropdown.

Once complete, hit the <kbd>RUN</kbd> button. You can follow the progress in the main SimBA terminal window and the operating system terminal window. The background removed video will be saved in the same directory as the `VIDEO PATH` with the `_bg_subtracted` suffix. 
   
##
Author [Simon N](https://github.com/sronilsson), [JJ Choong](https://github.com/inoejj)


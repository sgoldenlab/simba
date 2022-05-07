
# How to use tools to process videos in SimBA

We have developed video and image processing tools and incorporated them into the overall SimBA pipeline. However, many of the tools are useful on their own or in "one-off" situations. To make these easily accessible, the tools have been incorporated into their own space within the GUI, and are described below.

![alt-text-1](/images/toolsmenu.png)

- [Clip videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#shorten-videos)
- [Clip video into multiple videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#clip-video-into-multiple-videos)
- [Crop videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#crop-video)
- [Fixed Crop Videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#fixed-crop-videos)
- [Multi-crop](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#multi-crop-videos)
- [Downsample videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#downsample-video)
- [Get mm/ppx](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#get-coordinates)
- [Change formats](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#change-formats)
- [CLAHE enhance video](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#clahe)
- [Superimpose frame numbers on video](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#add-frame-numbers)
- [Convert to greyscale](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#convert-to-grayscale)
- [Merge frames to video](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#merge-images-to-video)
- [Generate gifs](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#generate-gifs)
- [Extract frames](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#extract-frames)
- [Convert .seq to .mp4](https://github.com/sgoldenlab/simba/blob/simba_JJ_branch/docs/Tutorial_tools.md#convert--seq-files-to-mp4-files)
- [Re-order pose-estimation tracking data](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#re-organize-tracking-data)
- [Remove body-parts from tracking data](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#re-organize-tracking-data)
- [Visualize pose estimation in folder](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#visualize-pose-estimation-in-folder)

## Shorten Videos
This is a tool used to trim video lengths. The tool contains two different methods: 
**Method 1** and **Method 2**. 

<img src="https://github.com/sgoldenlab/simba/blob/master/images/shortenvideos.PNG" width="330" height="287" />

### Method 1 
Use Method 1 to trim both the beginning and the end of the video. 

Let's say we have a 2 minute long video and we want to get rid of the first 10 seconds, and the last 5 seconds.
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

#### Convert multiple videos

1. Click on `Browse Folder` to select the directory that contains the videos that you want to convert.

2. Enter the original file format (eg: mp4, flv, avi etc.) in the `Input format` entry box. **Note: do not put dots ('.') in the file format name (eg: mp4 or flv, etc)**.

3. Enter the desired output format in the `Output format` entry box .

4. Click on `Convert multiple videos`.

#### Convert single video

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
Use this tool to visualize the pose-estimation of all the files inside a SimBA project directory. This tool can be useful when you have [interpolated  and/or smoothened](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-import-dlc-tracking-data) your pose-estimation data at import, and now you want to visualize the results of that interpolation and smoothing. 

1. Click on `Tools`, then `Visualize pose-estimation in folder...`.
2. In the `Input directory (with csv/parquet files)` menu, click bowse and select a folder that contain CSV or parquet files (e.g., the `project_folder/csv/input_csv` directory) 
3. In the `Output directory (where your videos will be saved)` menu, click bowse and select a folder where your videos should be saved (I recommend to choose an empty folder or create a new folder).
4. In the `Circle size` entry box, choose the size of the circles denoting the location of your body-parts (e.g., `5`)
5. Click on the `Visualize pose`. You can follow the progress in the main SimBA terminal window. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/viz_pose_folder.png" />
</p>

# How to use tools in GUI

<img src=https://github.com/sgoldenlab/simba/blob/master/images/tools.png width="375" height="322" />

- [Shorten Videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#shorten-videos)
- [Crop Videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#crop-video)
- [Downsample Videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#downsample-video)
- [Get Coordinates](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#get-coordinates)
- [Change formats](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#change-formats)
- [CLAHE](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#clahe)
- [Add Frame Numbers](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#add-frame-numbers)
- [Grayscale](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#convert-to-grayscale)
- [Merge Frames to video](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#merge-images-to-video)
- [Generate Gifs](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#generate-gifs)
- [Extract Frames](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#extract-frames)

## Shorten Videos
This is a tool to trim videos using ffmpeg. It contains **Method 1** and **Method 2**

<img src="https://github.com/sgoldenlab/simba/blob/master/images/shortenvideos.PNG" width="330" height="287" />

### Method 1 
Method 1 has the freedom to cut from the beginning of the video and from the end of the video.

Let's say we have a 2 minute long video and we want to get rid of the first 10 seconds and the last 5 seconds of the video.
We would start our video on the 10th second `00:00:10` and end at `00:01:55`.

1. First, click on `Browse File` to select the video to trim.

2. Enter the time frame in `Start at:` entry box, in this case it will be *00:00:10*

3. Enter the time frame in `End at:` entry box, in this case it will be *00:01:55*. The settings should look like the image below.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/shortenvideos1.PNG" width="435" height="287" />

4. Click `Cut Video` to trim video and a new shorten video will be generated, the new video will have a name of *Name of original video* + *_shorten* .

<img src="https://github.com/sgoldenlab/simba/blob/master/images/shortenvideos2.PNG" width="1014" height="114" />


### Method 2
Method 2 cuts of the beginning of the video. In order to use method 2, please check on the check box.

Let's say we have a 2 minute long video and we want to get rid of the first 20 seconds from the start of the video.

1. First, check on the check box `Check this box to use Method 2`

>**Note**: If the checkbox is not checked, method 1 will be used instead of method 2.

2. Enter the amount of time that needs to be trimmed from the start of the video in `Seconds:`, in this case it will be *20*

<img src="https://github.com/sgoldenlab/simba/blob/master/images/shortenvideos3.PNG" width="435" height="287" />

3. Click `Cut Video` to trim video and a new shorten video will be generated, the new video will have a name of *Name of original video* + *_shorten* .


## Crop Video
This is a tool to crop videos.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/cropvideo.PNG" width="232" height="232" />

1. First, click on `Browse File` to select a video to crop.

2. Then click on `Crop Video` and the following window `Select ROI` will pop up.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/cropvideo2.PNG" width="363" height="546" />

3. Use your mouse to *Left Click* on the video and drag to crop the video. *You can always left click and drag on the video to recrop your video*

<img src="https://github.com/sgoldenlab/simba/blob/master/images/cropvideo3.PNG" width="363" height="546" />

4. Hit `Enter` **twice** and it will crop the video, the new video will have a name of *Name of original video* + *_cropped*.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/cropvideo4.PNG" width="900" height="100" />


## Downsample Video
This is a tool to downsample video into smaller size and reduce the resolution of the video.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/downsamplevid.PNG" width="400" height="300" />

The downsample video tool has **Customize Resolution** and **Default Resolution**

### Customize Resolution
This allows you to downsample video into any height and width.

1. First, click on `Browse File` to select a video to downsample.

2. Then, enter any values in the `Height` and `Width` entry boxes.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/downsamplevid2.PNG" width="400" height="300" />

3. Click on `Downsample Custom Resolution Video` to downsample the video, the new video will have a name of *Name of original video* + *_downsampled*.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/downsamplevid3.PNG" width="900" height="100" />

### Default Resolution 
This allows you to downsample video quickly within two mouse clicks. There will be more options added into the **Default Resolution** in the future.

1. First, click on `Browse File` to select a video to downsample.

2. Click on one of the option `1980 x 1080` or `1280 x 1024`.

3. Click on `Downsample Default Resolution Video` and the video will downsample into the selected resolution.


## Get Coordinates
This tool is to get the length(millimeter) per pixel of a video.


<img src="https://github.com/sgoldenlab/simba/blob/master/images/getcoordtool.PNG" width="300" height="200" />

Let's say we want to find out the length(mm) per pixel of a video of a mouse in a cage, we know the width of cage is 130mm.

1. First, click on `Browse File` to select a video.

2. Enter *130* in the `Known length in real life(mm)` entry box.

3. Click `Get Distance`, and the following window should pop up.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/getcoordtool2.PNG" width="500" height="500" />

4. Use your mouse to double *Left click* at the left side of the cage and double *Left click* again on the right side of the cage.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/getcoordtool3.PNG" width="500" height="500" />

>**Note:** Double click the point again to change the location of the point.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/getcoord.gif" width="500" height="500" />

5. Once two points are selected, hit `Esc` button and it should print out the mm per pixel on the main GUI interface.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/getcoordtool4.PNG" width="400" height="250" />

## Change formats
Change formats includes **Change image format** and **Change video format**

### Change image format
This tool allows the user to select a folder with multiple images and convert the image format.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/changeimageformat.PNG" width="200" height="200" />

1. Click on `Browse Folder` to select a folder that contains multiple images

2. Choose the original image format that is in the folder

3. Choose the desire output image format

4. Click `Convert Image Format` .

### Change video format
This tool allows the user to convert videos into desired format.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/changevideoformat.PNG" width="400" height="400" />

#### Convert multiple videos

1. Click on `Browse Folder` to select the directory that contains all the videos that you wished to convert.

2. Enter the original format (eg: mp4,flv,avi...etc.) in the `Orignal Format` entry box. **Note: do not put in the dots " . " (eg: .mp4 or .flv etc)**

3. Enter the format to be converted into in the `Final Format` entry box 

4. Click `Convert multiple videos`.

#### Convert single video

1. Click on `Browse File` to select a video to convert.

2. Choose one of the following `Convert .avi to .mp4` or `Convert mp4 into Powerpoint supported format`

3. Click `Convert video format`.


## [CLAHE](https://docs.opencv.org/3.4/d5/daf/tutorial_py_histogram_equalization.html)

1. Click on the `Browse File` to select a video.

2. Click `Apply CLAHE`, and the new video will have a name of *CLAHE_* *Name of original video* and it will be in a **.avi** format.

## Add Frame Numbers

1. Click on `Add Frame Numbers`, and a new window will pop up.

2. Select video and click `Open` and frame numbers wil be superimposed to the video.

3. The new video will have a name of *Name of original video* + *_frame_no*.

## Convert to grayscale

1. Click on `Convert to grayscale`, and a new window will pop up.

2. Select video and click `Open` and the video will be converted to grayscale.

3. The new video will have a name of *Name of original video* + *_grayscale*.


## Merge images to video

1. Click on `Browse Folder` to select a folder with frames 

2. Enter the image format of the frames. Eg: if the image name is *Image001.PNG*, enter *PNG* in the `Image Format` entry box.

3. Enter the desire output video format. Eg: if the video should be a *.mp4*, enter *mp4* in the `Video Format` entry box.

4. Enter the desire frames per second in the `fps` entry box.

5. Enter the desire [bitrate](https://help.encoding.com/knowledge-base/article/understanding-bitrates-in-video-files/) for the video in the `Bitrate` entry box.

6. Click `Merge Images`

## Generate Gifs

1. Click on `Browse File` to select a video to convert to a GIF

2. Enter the starting time of the GIF from the video in the `From time(s)` entry box.

3. Enter the duration of the GIF in the `Duration(s)` entry box.

4. Enter the size of the GIF in the `Width` entry box. The output GIF will be scale automatically.

## Extract Frames
Extract frames consist of **Extract Defined Frames**, **Extract Frames**, and **Extract Frames from seq files**

### Extract Specific Frames
Extract specific frames allows the user to extract from certain frames to certain frames in a video.

1. Click `Browse File` to select a video.

2. Enter the starting frame number in the `Start Frame` entry box.

3. Enter the ending frame number in the `End Frame` entry box.

4. Click `Extract Frames` to extract the frames from `Start Frame` to `End Frame`.

5. A folder with the video name will be generated and the all the frames will be in the folder. The image will be in *.png* format

### Extract Frames
Extract frames allows the user to extract every frame from a video.

1. Click `Browse File` to select a video.

2. Click `Extract All Frames` to extract every frames from the video.

3. A folder with the video name will be generated and the all the frames will be in the folder. The image will be in *.png* format

### Extract Frames from seq files
Extract frames from seq files allows the user to extract every frame from a .seq file.

1. Click `Browse File` to select a video.

2. Click `Extract All Frames` to extract every frames from the video.

3. A folder with the video name will be generated and the all the frames will be in the folder. The image will be in *.png* format


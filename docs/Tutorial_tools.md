# How to use tools in GUI

<img src=https://github.com/sgoldenlab/tkinter_test/blob/master/images/tools.png width="375" height="322" />

- [Shorten Videos](https://github.com/sgoldenlab/tkinter_test/blob/master/docs/Tutorial_tools.md#shorten-videos)
- [Crop Videos](https://github.com/sgoldenlab/tkinter_test/blob/master/docs/Tutorial_tools.md#crop-video)
- [Downsample Videos](https://github.com/sgoldenlab/tkinter_test/blob/master/docs/Tutorial_tools.md#downsample-video)
- Get Coordinates
- CLAHE
- Add Frame Numbers
- Greyscale
- Merge Frames to video
- Generate Gifs
- Extract Frames

## Shorten Videos
This is a tool to trim videos using ffmpeg. It contains **Method 1** and **Method 2**

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/shortenvideos.PNG" width="330" height="287" />

### Method 1 
Method 1 has the freedom to cut from the beginning of the video and from the end of the video.

Let's say we have a 2 minute long video and we want to get rid of the first 10 seconds and the last 5 seconds of the video.
We would start our video on the 10th second `00:00:10` and end at `00:01:55`.

1. First, click on `Browse File` to select the video to trim.

2. Enter the time frame in `Start at:` entry box, in this case it will be *00:00:10*

3. Enter the time frame in `End at:` entry box, in this case it will be *00:01:55*. The settings should look like the image below.

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/shortenvideos1.PNG" width="435" height="287" />

4. Click `Cut Video` to trim video and a new shorten video will be generated, the new video will have a name of *Name of original video* + *_shorten* .

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/shortenvideos2.PNG" width="1014" height="114" />


### Method 2
Method 2 cuts of the beginning of the video. In order to use method 2, please check on the check box.

Let's say we have a 2 minute long video and we want to get rid of the first 20 seconds from the start of the video.

1. First, check on the check box `Check this box to use Method 2`

>**Note**: If the checkbox is not checked, method 1 will be used instead of method 2.

2. Enter the amount of time that needs to be trimmed from the start of the video in `Seconds:`, in this case it will be *20*

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/shortenvideos3.PNG" width="435" height="287" />

3. Click `Cut Video` to trim video and a new shorten video will be generated, the new video will have a name of *Name of original video* + *_shorten* .


## Crop Video
This is a tool to crop videos.

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/cropvideo.PNG" width="232" height="232" />

1. First, click on `Browse File` to select a video to crop.

2. Then click on `Crop Video` and the following window `Select ROI` will pop up.

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/cropvideo2.PNG" width="363" height="546" />

3. Use your mouse to *Left Click* on the video and drag to crop the video. *You can always left click and drag on the video to recrop your video*

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/cropvideo3.PNG" width="363" height="546" />

4. Hit `Enter` **twice** and it will crop the video, the new video will have a name of *Name of original video* + *_cropped*.

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/cropvideo4.PNG" width="900" height="100" />


## Downsample Video
This is a tool to downsample video into smaller size and reduce the resolution of the video.

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/downsamplevid.PNG" width="400" height="300" />

The downsample video tool has **Customize Resolution** and **Default Resolution**

### Customize Resolution
This allows you to downsample video into any height and width.

1. First, click on `Browse File` to select a video to downsample.

2. Then, enter any values in the `Height` and `Width` entry boxes.

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/downsamplevid2.PNG" width="400" height="300" />

3. Click on `Downsample Custom Resolution Video` to downsample the video, the new video will have a name of *Name of original video* + *_downsampled*.

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/downsamplevid3.PNG" width="900" height="100" />

### Default Resolution 
This allows you to downsample video quickly within two mouse clicks. There will be more options added into the **Default Resolution** in the future.

1. First, click on `Browse File` to select a video to downsample.

2. Click on one of the option `1980 x 1080` or `1280 x 1024`.

3. Click on `Downsample Default Resolution Video` and the video will downsample into the selected resolution.


## Get Coordinates
This tool is to get the length(millimeter) per pixel of a video.


<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/getcoordtool.PNG" width="300" height="200" />

Let's say we want to find out the length(mm) per pixel of a video of a mouse in a cage, we know the width of cage is 130mm.

1. First, click on `Browse File` to select a video.

2. Enter *130* in the `Known length in real life(mm)` entry box.

3. Click `Get Distance`, and the following window should pop up.

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/getcoordtool2.PNG" width="500" height="500" />

4. Use your mouse to double *Left click* at the left side of the cage and double *Left click* again on the right side of the cage.

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/getcoordtool3.PNG" width="500" height="500" />

>**Note:** Double click the point again to change the location of the point.

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/getcoord.gif" width="500" height="500" />

5. Once two points are selected, hit `Esc` button and it should print out the mm per pixel on the main GUI interface.

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/getcoordtool4.PNG" width="400" height="250" />



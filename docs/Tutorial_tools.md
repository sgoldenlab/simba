# How to use tools in GUI

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/tools.PNG" width="375" height="322" />

## Shorten Videos
This is a tool to trim videos using ffmpeg. It contains **Method 1** and **Method 2**

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/shortenvideos.PNG" width="330" height="287" />

### Method 1 
Method 1 has the freedom to cut from the beginning of the video and from the end of the video.

Let's say we have a 2 minute long video and we want to get rid of the first 10 seconds and the last 5 seconds of the video.
We would start our video on the 10th second `00:00:10` and end at `00:01:55`.

1. First, select the video to trim, click on `Browse File`

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

1. First, select a video to crop, click on `Browse File`

2. Then click on `Crop Video` and the following window `Select ROI` will pop up.

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/cropvideo2.PNG" width="363" height="546" />

3. Use your mouse to *Left Click* on the video and drag to crop the video. *You can always left click and drag on the video to recrop your video*

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/cropvideo3.PNG" width="363" height="546" />

4. Hit `Enter` **twice** and it will crop the video, the new video will have a name of *Name of original video* + *_cropped*.

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/cropvideo4.PNG" width="900" height="100" />


## Downsample Video
This is a tool to downsample video into smaller size and reduce the resolution of the video.

The downsample video tool has **Customize Resolution** and **Default Resolution**

### Customize Resolution
This allows you to downsample video into any height and width.

1. First, click 



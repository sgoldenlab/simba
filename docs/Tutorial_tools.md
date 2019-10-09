# How to use tools in GUI

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

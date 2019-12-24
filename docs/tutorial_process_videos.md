

# Tutorial for processing videos in batch
This video pre-processing tool allows users to change multiple video parameters (clip/trim/crop etc.) for many videos independently in a batch. Once the required parameters has been set for all the videos, the user press `Execute` and the new videos will be generated according to the user input. Videos are processed using **FFmpeg**. Click [here](https://m.wikihow.com/Install-FFmpeg-on-Windows) to learn how to install FFmpeg on your computer. 

## Pipeline

The video parameters specified by the user will be processed in the following sequence. If the user leaves certain parameters unchanged, then they are ignored in the pipeline.   

![alt-text-1](/images/processvideo_flowdiagram.png "processvideo_flowdiagram")

## Step 1: Folder Selection

![alt-text-1](/images/processvideo.PNG "processvideo")

1. To get started with the batch pre-process, in the main SimBA window, click on `Process Videos` --> `Batch pre-process videos`. The window shown below will pop open. 

![](/images/batchprocessvideo1.PNG)

2. Under **Folder Selection** heading and next to `Video directory`, click on `Browse Folder` and navigate to a folder that contains the videos that should be batch processed and click on 'Select Folder`.

![](/images/selectfolderwithvideos.PNG)

3. Next to `Output Directory`, click on `Browse Folder` and navigate to a folder *(usually a new, empty, folder)* that should store the processed videos and click on 'Select Folder`.

4. Click to `Confirm` the two selected directories.

>**Note**: Please make sure there is no spaces in your folder names or video names.

![](/images/processvideo2.PNG)

## Step 2: Select parameters

1. Once the user clicks `Confirm`, a table listing the videos in the `Video directory` folder will pop up.

![](/images/batchprocessvideo.PNG)

2. If you wish to crop your videos, click on the `Crop` button. A frame from the video will pop up. Left click on the frame and drag the rectangular bounding box to mark the region of the video you wish to keep. Once the rectangle is marked, double tap "Enter" on your keyboard. *(The relevant crop button will turn red once you have selected to crop a video)*

![](/images/cropvideoroi.gif)

3. If you wish to trim a specific video, check the `Shorten` box and enter the **Start Time** and  **End Time** in the HH:MM:SS format.

4. If you wish to downsample a specific video, check the `Downsample` box and enter the **Width** and **Height** in pixels.

5. If you wish to change a specific video to grayscale, check the `Grayscale` box.

6. If you wish to superimpose the specific frame number onto all frames of a specific video, check on the `Add Frame #` box.

7. If you wish to apply CLAHE, check on the `CLAHE` box. For more information on CLAHE, click [here](https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html)

8. In the first row of the table, there are `Select All` checkboxes. Use these checkboxes to apply a manipulation to all of the videos in the folder. 

> **Note:** We know that the `Select All` checkbox might be slightly off position in the table. We are working on a fix. 

## Step 3: Execute

1. Once all the parameters are set, click on `Execute`. 

2. The final output videos will be saved in the `Output Directory` that you selected in *Step 1*.

![alt-text-1](/images/processvideo4.PNG "processvideo4.PNG")

3. A subfolder in the `Output Directory` called **tmp** will contain the step-by-step processsed videos.

4. The **process_archieve** folder contains a **.txt** file that lists the processes that were run. 

5. The **Output Directory** will contain all the final processed videos.


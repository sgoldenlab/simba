

# Tutorial for processing video in batch
The batch pre-processing tools allows users to change video parameters (clip/trim/crop etc.) for multiple videos independently. Once the required parameters has been set for all the videos, the user press `Execute` and the new videos will be generated. Videos are processed via **FFmpeg** Click [here](https://m.wikihow.com/Install-FFmpeg-on-Windows) to learn how to install FFmpeg on your computer. 

## Pipeline

The video parameters specified by the user will be processed in the following sequence. If the user leaves certain parameters unchanged in, they will be ignored in the pipeline.   

<img src=https://github.com/sgoldenlab/simba/blob/master/images/processvideo_flowdiagram.png width="800" height="310" />

## Step 1: Folder Selection

<img src=https://github.com/sgoldenlab/simba/blob/master/images/processvideo.PNG width="285" height="111" />

1. To get started with the batch pre-process, in the main SimBA window, click on `Process Videos` --> `Batch pre-process videos`. The windows in the image below will pop open. 

![](/images/batchprocessvideo1.PNG)

2. Under **Folder Selection** and `Video directory`, click on `Browse Folder` and navigate to a folder that contains the videos that should be processed and click on 'Select Folder`.

![](/images/selectfolderwithvideos.PNG)

3. Under `Output Directory`, lick on `Browse Folder` and navigate to a folder *(usually a new, empty, folder)* that should store the processed videos and click on 'Select Folder`.

4. Click `Confirm` to confirm the two selected directories.

>**Note**: Please make sure there is no spaces in your folder names or video names.

<img src=https://github.com/sgoldenlab/simba/blob/master/images/processvideo2.PNG width="450" height="200" />

## Step 2: Select parameters

1. Once the user clicks `Confirm`, a table listing the videos in the `Video directory` folder will pop up.

<img src=https://github.com/sgoldenlab/simba/blob/master/images/batchprocessvideo.PNG width="1102" height="200" />

2. If you wish to crop your videos, click on the `Crop` button. A frame from the video will pop up. Left click on the frame and drag the rectangle to mark the region of the video you wish to keep. Once it is done, double tap "Enter" on your keyboard. *(The crop button will turn red once you have cropped the selected video)*

![](/images/cropvideoroi.gif)

3. If you wish to trim a specific video, check on the `Shorten` box and enter the **Start Time** and  **End Time** in the HH:MM:SS format.

4. If you wish to downsample a specific video, check on the `Downsample` box and enter the **Width** and **Height** in pixels.

5. If you wish to change a specific video to grayscale, check on the `Grayscale` box.

6. If you wish to superimpose a print of the specific frame number onto a specific video, check on the `Add Frame #` box.

7. If you wish to apply CLAHE, check on the `CLAHE` box. For more information on CLAHE, click [here](https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html)

8. In the first row of the table, there are `Select All` checkboxes. Use these checkboxes to apply a manipulation to all of the videos in the folder. 

> **Note:** The `Select All` checkbox might be off position. We are working on a fix. 

## Step 3: Execute

1. Once all the parameters are set, click on `Execute`. 

2. The final output videos will be saved in the `Output Directory` that you selected in *Step 1*.

<img src=https://github.com/sgoldenlab/simba/blob/master/images/processvideo4.PNG width="500" height="200" />

3. A subfolder in the `Output Directory` called **tmp** will contain the step-by-step processsed videos.

4. The **process_archieve** folder contains a **.txt** file that lists the processes that were run. 

5. The **Output Directory** will have all the final processed videos.


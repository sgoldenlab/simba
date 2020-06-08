# 'Pseudo-labelling' behaviors in SimBA

Annotating videos and frames can often be the most time-consum aspect of supervised machine learning, and computational tools are developed at pace to address this time-sink. For example, pose-estimation tools such as [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) and [SLEAP](https://sleap.ai/) has tools and workflows that significantly decrease the required body-part labelling time; DeepLabCut asks users to [extract/refine/merge* outliers](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/UseOverviewGuide.md#optional-active-learning----network-refinement---extract-outlier-frames-from-a-video), and SLEAP asks users to take a computer-assicted labelling approach, where users run [non-optimized modelles and correct their prediction results until they are perfected](https://sleap.ai/tutorials/initial-training.html). This way we feed the computer-generated predictions, initially with human oversight, back into the model as training examples and thus improving its predictions through more data all while spending significantly less time annotating images.


*>Note:* If you want to label behaviors in SimBA, you **must** extract the frames for the videos you wish to label. The frame extraction ensures that the frame number, the behavioral label, and the displayed frame during annotations all align. For more information on how to extract frames in SimBA, see the [Scenario 1 Part 2 - Import more DLC tracking data into project](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-2-optional--import-more-dlc-tracking-data-or-videos), or the generic [Tools menu] tutorials(https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#extract-frames).

## Step 1: Generating initial machine predictions in SimBA

Before using the 'pseudo-labelling' tool in SimBA, we need to generate some machine learning classification predictions. Detailed information on how to generate such machine classifications can be found in the [Scenario 1 - building classifiers from scratch](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md) tutorial. More  documentation on how to generate machine predictions can also be found in the [generic SimBA documentation](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md). We recommend reading the one of these tutorials up to and including [Step 8 - Run Machine Model](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-8-run-machine-model) 

When you have successfully generated your machine predictions, you will have CSV files containing your machine classifications, with one CSV file for each video in your project, in your `project_folder/csv/machine_results` directory. We will now use the 'Pseudo-labbeling` tool in SimBA to look at these machine classifications and correct them when necessery.

## Step 2: Run pseudo-labelling 

After [loading your project in SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-load-project-config), navigate to the `Label behavior` tab and uou should see the following menu:

![](/images/Pseudo_1.PNG)

1. In the sub-menu titled `Pseudo Labelling`, there is a entry-box called `Frame folder`. Click on `Browse` and select to the folder containing the frames for the video you wish to correct the machine predictions for.  

2. Beneath the `Frame folder` entry-box, there is a sub-menu titled 'Threshold'. Within this sub-menu there are entry-boxes, with one entry-box for each classifier in the project. In these entry-boxes, insert a value between 0.00 and 1.00 which represents the classifcation threshold for each classifier. If you are un-familiar with classification thresholds, hyou can read more about them in the tutorial for [Scenario 1 - Step 8](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-8-run-machine-model), or [Scenario 2 - Part3](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data). In brief, the classification threshold represents how sure the computer has to be before it classifies a frame as containing the behavior - a value of 0.80, for example, would mean that the computer has to be 80% sure that it contains the behavior to classify it as containing the behaviour. When the thresholds have been entered, click on `

3. 











running pseudo label, the video has to go through [**Run machine model**](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-8-run-machine-model) and has .csv file in the `/project_folder/csv/output/machine_results` folder.
With this tool, users are able to look at the machine annotation and correct them.

![](/images/pseudolabel.PNG)

1. Load the *project_config.ini* file.

2. Under `[ Label behavior ]`, **Pseudo Labelling**, select the video folder.

3. Set the threshold for the behavior.

4. Click `Correct label`.

5. Note that the checkboxes will autopopulate base on the computer's prediction on the threshold set by the user.

6. Once it is completed, click on `Save csv` and the .csv file will be saved in `/project_folder/csv/output/target_inserted`.

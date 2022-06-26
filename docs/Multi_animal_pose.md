# Using multi-animal pose-estimation data in SimBA

New developments in pose-estimation packages such as [maDeepLabCut (version >=2.2b5)](https://github.com/DeepLabCut/DeepLabCut/releases/tag/v2.2b5) and [SLEAP](https://sleap.ai/) allow users to track similar looking animals, for example multiple littermates of the same coat-color. In prior versions of these pose-estimation tools, tracking several similar looking animals could be difficult as a single body-part (i.e., the "Snout" or the "Tail-base") was intermittently attributed to either animal. 

In this tutorial we will import multi-animal tracking data from DeepLabCut into SimBA from videos of two black-coated C57BL/6J mice. SimBA is capable of working with more than two subjects, but has been optimized for two-subject pairings. Multiple animals can also be tracked with [*Animal Part Tracker* (APT)](http://kristinbranson.github.io/APT/index.html) and such data (.trk files) can also be imported into SimBA. 

**IMPORTANT NOTE:** Both maDLC and SLEAP present significant advances in pose-estimation. However, behaviors that contain elements where experimental subjects are fully occluded, by either environmental factors (such as enrichment igloos) or by other experimental subjects (such as during aggression or mating behaviors), may present tracking difficulties. The accuracy of predictive classifiers depends on tracking, so it is critical to troubleshoot any similar multiple animal pose-estimation issues prior to importing tracking data into SimBA.

### Step 1: Generate Project Config

If you are coming along from [Scenario 1 - Creating a classifier from scratch](https://github.com/sgoldenlab/simba/edit/SimBA_no_TF/docs/Scenario_1_new.md), then you have already completed *Step 1* of the current tutorial and you can skip this part and continue to Step 2. 

In this Step we will create the main project folder, which will then auto-populate with all the required sub-directories.

1. In the main SimBA window, click on `File` and and `Create a new project`. The following windows will pop up.

![](/images/Create_project_1.PNG "createproject")

2. Navigate to the `[ Generate project config ]` tab. Under **General Settings**, specify a `Project Path` which is the directory that will contain your main project folder.

3. `Project Name` is the name of your project. 
*Keep in mind that the project name cannot contain spaces. We suggest to instead use underscore "_"* 

4. In the `SML Settings` sub-menu, put in the number of predictive classifiers that you wish to create. For an example, in Scenario 1 we would like to create a single classifier. We will enter the number 1. Note that in the real world you would probably want to create multiple classifiers concurrently as this would decrease the number of times a video would need to be manually annotated. For simplicity, we will here create a single classifier.

5. Click <img src="https://github.com/sgoldenlab/simba/blob/master/images/addclassifier.PNG" width="153" height="27" /> a single time, and it creates a row as shown in the following image. In each entry box, fill in the name of the behavior (BtWGaNP) that you want to classify. If you click too many times, as long as you leave the extra boxes empty, all is well.

<p align="center">
  <img width="385" height="106" src="https://github.com/sgoldenlab/simba/blob/master/images/classifier1.PNG">
</p>

### Step 2: Define your Type of Tracking

1. In this current Scenario we are using multi-animal tracking. Click on the `Type of Tracking` drop-down menu in the `Animal Settings` sub-menu and chose `Multi tracking`.  

![](/images/Multi_1.png "createproject")

2. Next, click on the `# config` drop-down menu to specify the pose-estimation body-parts configuration you used to track your animals. If the body-parts you tracked on your animals are not listed, then chose the `Create pose config...` option and head to the tutorial on [how to use user-defined pose-configurations in SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/Pose_config.md) for intructions on how to import your data into SimBA. 

![](/images/Multi_animal2.jpg "createproject")

**IMPORTANT**. The images of the animals in the menu above shows two black mice. However, this is for illustrative purposes only and is used for hightlighting the expected body-parts and their order of how the body-parts were labeled in DeepLabCut or SLEAP (the numbers 1-8). If your data contains any other number of animals (e.g., you have 3 animals rather than 2), and your animals look different (e.g., you have several white coat-coated animals rather than several black-coated animals), but your data contains the pose-estimation tracks of the 8 body-parts in the image, then go ahead and select this `# config` setting. In other words, select the `Multi-tracking # config` setting based solely on the expected body-parts and their order of labelling and **NOT** the species / number of animals shown in the image.  

In this current scenario we will import the tracking data for 8 body-parts on each of the two black-coated mice (16 body-parts in total). In the `# config` drop-down menu we select `Multi-animals, 8bps`. Once selected, we click on `Generate Project Config`. 

### Step 3: Import your videos

1. Next, click on the `[Import videos into project folder]` tab in the `Project configuration` window. If you have multiple videos, use the `Import multiple videos` sub-menu to select a folder containing the *raw* MP4 or AVI video files you have analyzed in DeepLabCut. If you have a single video, use the `Import single video` sub-menu to select a single MP4 or AVI file to import into the project. For more details on importing video files into SimBA projects, see the following tutorial documentation: [1](https://github.com/sgoldenlab/simba/blob/SimBA_no_TF/docs/Scenario1.md#step-2-import-videos-into-project-folder)[2](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-2-import-videos-into-project-folder).

>*Note*: When importing multi-animal pose-estimation data into SimBA you **must** import your videos prior to importing your tracking data. The videos will be used to interactively re-organise your pose-estimation data to ensure  that the pose-estimation data for each specific individual falls in the same column, and same column order, in all your tracked files (more information on this follows below). 

### Step 4: Import your tracking data

1. After importing your video files, click on the `[Import tracking data]` tab in the `Project configuration` window. In the `Import tracking data` sub-menu, click on the `File type` drop-down menu to select your pose-estimation tracking data file type. 

![](/images/Import_data_create_project_new_4.png "createproject")

In this tutorial we have multi-animal tracking data in .H5 file format from DeepLabCut, and we select the `H5 (multi-animal DLC)` option. For more information on the generating .H5 multi-animal tracking files in DeepLabCut, consult the [DeepLabCut tutorials on YouTube](https://www.youtube.com/channel/UC2HEbWpC_1v6i9RnDMy-dfA), the [DeepLabCut GitHub documentation](https://github.com/DeepLabCut/DeepLabCut/releases/tag/v2.2b5), or the [DeepLabCut Gitter channel](https://gitter.im/DeepLabCut/community). SimBA will convert these files during the processes descibed below into organised and more readable CSV files that can be opened with, for example, Microsoft Excel or OpenOffice Calc. 

>*Note I*: If you have generated multi-animal tracking data using [SLEAP](https://sleap.ai/) you should select the `SLP (SLEAP)` option from the `File type` menu. This difference in `File type` selection is the **only user difference** between processing SLEAP and DLC data in SimBA. In fact, despite the different file type names (H5 vs. SLP) they are both H5 dataframes that will be converted into readily readable and organised CSV file formats in SimBA which are more accessable to neuroscientst unfamiliar with dataframe container file formats.

>*Note II*: If you have generated multi-animal tracking data using [*Animal Part Tracker* (APT)](http://kristinbranson.github.io/APT/index.html) and have .TRK tracking files, you should select the `TRK (multi-animal APT)` option from the `File type` menu.

>*Note III*: SimBA focus on use of CSV file formats due to its greater accessability, and SimBA converts alterative pose-estimation tracking formats (e.g., H5, JSON, SLP) behind-the-hood to CSV to enable a greater number of researchers to take advantage of SimBA. If you are tracking a large number of body-parts (e.g., more than 8 body-parts), on a larger number of animals (e.g., more than 3 animals), in longer videos (e.g., more than 20min / 30 fps) then CSV files may be non-optimal to work with due to their relatively slow read/write speed and less-than-ideal compression form (they can become gigabytes after calculating [feature sets](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features)). Although working with CSV files is fine, we are currently working on a tick-box option in SimBA that would allow users to save their data in file formats with better compression ratio and read-write speed. This may be preferred for more advanced users.

2. Below the `File type` drop-down menu, there are two option menus that can help correct and improve the incoming pose-estimation data (*interpolation*, and *smoothing*). Both of these menus default to `None`, meaning no corrections will be performed on the incoming data. If you are intrested in removing missing values (interpolation), and/or correcting "jitter" (smoothing), please see sections `2` and `3` in [TUTORIAL SCENARIO 1: IMPORT TRACKING DATA](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#to-import-multiple-dlc-csv-files) which details these functions, including video examples of expected results.

3. After selecting the File type, enter the number of animals you are tracking in the `No of animals` entry box. In the current scenario we are tracking two animals and enter the number `2`. After entering the number 2, two further rows will appear asking you to name your two animals. In this tutorial, we will name Animal 1 *Resident* and Animal 2 *Intruder*. 

![](/images/Multi_animal4.jpg "createproject")

>*Note*: Avoid spaces and commas `,` in your animal names. If you had planned to use spaces in your animal names, we recommend replacing them with underscore `_`. 

4. Multi animal DLC (DLC verson >=2.2b5) offers two tracking methods (skeleton vs. box). In the `Tracking type` dropdown, select the option you used to generate your tracking pose-estimation (this drop-down menu is only relevant for importing deepLabCut H5 files, and ths drop-down is not shown if importing SLEAP SLP data). 

5. Next to `Path to h5 files`, click on `Browse` to select the folder that contain you H5 files. The selected folder should contain one H5 file for all the videos imported into the project during [Step 3 of the current tutorial](https://github.com/sgoldenlab/simba/blob/SimBA_no_TF/docs/Multi_animal_pose.md#step-3-import-your-videos). Once you have selected a folder containing your H5 files, click on `Import h5`.

### Step 5: Assigning tracks the correct identities. 

When SLEAP and multi-animal DLC predicts the location of body-parts for multiple animals, the animals are assigned *tracks* or *tracklets* (for more information, see the SLEAP and maDLC documentation), with one track for each animal. In the current scenario, this means that DeepLabCut may have assigned *track 1* to be the Resident, and *track 2* to be the Intruder. It could also be the reverse, and *track 1* is the Intruder and *track 2* is the Resident. Further - if you have multiple videos, then the identity of the animal representing *track 1* and *track 2* will most likely shift across the different videos in your project. We need to organise the pose-estimation data such as *track 1* is Animal 1 (Resident) and *track 2*  is Animal 2 (Intruder) **in all of our tracking files** and SimBA does this through an interactive interface. 

>Note: If you used SLEAP to track only **one** animal, then SimBA will automatically skip over over this step.  

1. When you click on `Import h5`, you will see the following window pop open.

![](/images/Multi_animal5.jpg "createproject")

This represents the first frame of your video, with the pose-estimation body-part locations plotted on top of the animals. If you **can** tell which animal is the Resident and which animal is the Intruder in the diplayed frame, then proceed to press `c` on your keyboard. If you **cannot** tell which animal is the Resident and which animal is the Intruder, press `x` on your keyboard.

If you press `x` on your keyboard, then a new random frame will be shown and you will be asked again if you can tell which animal is the Resident, and which animal is the Intruder. Continoue to press `x` on your keyboard until SimBA displays a frame where you can tell which animal is which. 

When you press `c`, SimBA will shown this new message below asking you to double left mouse click on the Resident (Animal 1). After you double left mouse click on the Resident (Animal 1), SimBA will ask you to double left mouse click on the Intruder (Animal 2). 

![](/images/Multi_animal6.jpg "createproject")

Once you have clicked on all the animals in your project, you will be asked if you are happy with your assigned identities. If you are not happy with your assigned identities, press `x` on your keyboard to re-assign the animal identities. If you are happy with your assigned identities,  press `c` on your keyboard to save an organised (according to the animal identities) CSV pose-estimation file. SimBA will loop over all the videos and H5 files in your project asking you to assign the identities of the animals in each file. 

The process for each video should look like in the following gif:

![](/images/multiVid.gif "createproject")

>*Note I*: What happens behind-the-hood is that SimBA looks for the body-part closest to your left-mouse click and assigns the track that encompasses that body-part to the current Animal ID (e.g., the Resident). Thus, we recommend that your left-mouse click is at the centroid of the animal you wish to assign the current identity to prevent several tracks being assigned to the same identity. 

>*Note II*: The back-end process of converting and organising SLEAP H5 tracks into CSV is a little more time-consuming for SimBA than converting DLC tracks, so if you are shifting from DeepLabCut to SLEAP pose-estimation, and the process takes a little longer than what you are used to, then don't be alarmed - you can follow the progress through messages printed in the main SimBA terminal window. 

2. When you have attributed the animal IDs for all of the videos of your project, you can navigate to your project folder to check that all CSV files have been generated as expected. Navigate to your `project_folder/csv/input_csv` and you should see one CSV file for each of the videos in your project. 

Congratulation! You have now successfully imported your multi-animal pose-estimation tracking into SimBA and you are now ready to use this data to create predictive classifiers, analyze using already generated predictive classifiers, and/or analyze descriptive statistics.

Go to [Scenario 1 - Creating a classifier from scratch](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-2-load-project-1) for tutorials on how to use the imported data to create predictive classifiers. 

Go to [Scenario 2 - Using a classifier on new experimental data](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md) for tutorials on how to analyze the imported data using previously generated classifiers. 

Go to [Scenario 3 - Updating a classifier with further annotated data](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md) for tutorials on how to use the imported data to update previously created classifiers.











































 

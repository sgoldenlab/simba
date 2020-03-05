# Using DeepPoseKit in SimBa

DeepPoseKit projects can be created and managed in SimBa, and DeepPosekit tracking data can be imported into SimBA to [generate machine learning behavioral classifiers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md).

[DeepPosekit](https://github.com/jgraving/DeepPoseKit) is an extremely powerful and flexible pose-estimation tracking tool that has a wide-range of features that enables user to create robust tracking models. [DeepLabCut](https://github.com/AlexEMG/DeepLabCut) and [LEAP](https://github.com/talmo/leap) based model architectures can also be generated within the DeepPoseKit pipeline which may allow model comparisons - please read the [DeepPoseKit paper in ELife](https://elifesciences.org/articles/47994). The base DeepPoseKit package does not come with a GUI or standardized project folder structures, which users of other animal pose-estimation tools might have gotten used to. However, such folder structures are created when DeepPoseKit projects are managed through SimBA.  

**IMPORTANT I**: Although DeepPoseKit currently does not come with a GUI, it's commandline  functions offer much(!!) extended functionality over the SimBA menus accessing DeepPoseKit. We **strongly** urge  those interested in DeepPoseKit to access its functions through the available [notebooks](https://github.com/jgraving/DeepPoseKit#using-deepposekit-is-a-4-step-process) when possible to get the full tour-de-force DeepPoseKit experience. 

**IMPORTANT II**: For extensive details on the DeepPoseKit workflow, please see the documentation on the [DeepPoseKit Github page](https://github.com/jgraving/DeepPoseKit).

>*Note*: Depending on the specific version of the different pose estimation tools the dependencies may diverge. For example, note that DeepPoseKit require tensorflow >=1.13 and CUDA 10, while some versions of DeepLabCut run on earlier CUDA/tensorflow releases. 

# Part 1: Creating a DeepPoseKit project in SimBA

To start using DeepPoseKit in SimBA, begin by creating a SimBA project as decribed in-depth in [Part 1 - Create a new project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-1-create-a-new-project-1) of the Scenario 1 Tutorial. 

Next, close the Project Configuration window. In the main SimBA console window, click on Tracking. In the dropdown, chose DeepPoseKit, and click on Create DeepPoseKit project.

![](https://github.com/sgoldenlab/simba/blob/master/images/DPK1.png "dir_info")

Once clicked, the following window should pop open: 

![](https://github.com/sgoldenlab/simba/blob/master/images/DPK2.png "dir_info")

Next to the SimBA project.in file, click on browse and select your SimBA project_config.ini file. Next to project Name, give your DeepPoseKit project a name (*Note:* avoid using spaces in the name. Instead, use underscore if needed). Click on Generate project to create your DeepPoseKit project folder structure. 

You DeepPoseKit project will be generated within the chosen SimBA project. If you navigate to you SimBA project, and look within the `project_folder\logs\measures\dpk` folder, you should see a folder named after your chosen `Project Name`. The inside if this folder will contain several subfolders reminiscent of the sub-folders created by other packages:
![](https://github.com/sgoldenlab/simba/blob/master/images/DPK_2.png "dir_info")

Go ahead and close the `Create DeepPoseKit project` window. 

Once created, you may want to update the **parent** and **swap** columns in your generated *skeleton.csv* file. These columns defaults to *NaN* when DeepPoseKits are generated in SimBA - and they are most readily updated manually by opening the *skeleton.csv* file. 

![](https://github.com/sgoldenlab/simba/blob/master/images/skeleton.png "dir_info")

For more information on the  **parent** and **swap** settings, please see the [DeepPosekit](https://github.com/jgraving/DeepPoseKit) repository.

# Part 2: Loading a DeepPoseKit project in SimBA

In the main SimBA console window, click on Tracking. In the dropdown, chose DeepPoseKit, and click on Load DeepPoseKit project and the following window pops up:

![](https://github.com/sgoldenlab/simba/blob/master/images/DPK_3.png "DPK_3")

When DeepPoseKit projects are created in SimBA, they each come with their own *.ini* file that stores your project settings chosen in the SimBA GUI. This is **not** the same *project_config.ini* that stores your main SimBA project settings, but a separate *.ini* file, exclusive to your DeepPoseKit project. Go ahead and click on `browse File` and navigate to your DeepPoseKit *.ini* file located in your DeepPosekitProject (in the `project_folder\logs\measures\dpk` directory). Once done, click on `Load Project`.

# Part 3: Working with DeepPoseKit projects in SimBA

## Step 1: Import videos into your DeepPoseKit project

After loading you newly created project, first navigate to the `[Import videos]` tab. 

![](https://github.com/sgoldenlab/simba/blob/master/images/DPK_import.PNG "DPK_4")

#### To import multiple videos
1. Under the `Import multiple videos` sub-heading, click on `Browse Folder` to select a folder that contains all the videos that you wish to import into your project.
2. Enter the file type of your videos. (e.g., *mp4*, *avi*, *mov*, etc. - do *not* enter the fullstop/period in the file-ending name) in the `Video type` entry box.
3. Click on `Import multiple videos`. 
>**Note**: If you have a lot of videos, it might take a few minutes before all the videos are imported into your project.
#### To import a single video
1. Under the `Import single video` heading, click on `Browse File` to select your video.
2. Click on `Import a video`.

If you'd like to confirm that all the videos are imported into your project, then navigate to the `project_folder\logs\measures\dpk\*DeepPoseKit project name*\input\videos` folder and you should see all of your imported videos.  

## Step 2: Create a DeepPoseKit annotation set

1. To create an annotation set, click on the `[Create annotation set]` tab. For more information on the different user-definable variables, refer to the [DeepPosekit](https://github.com/jgraving/DeepPoseKit) respository.

![](https://github.com/sgoldenlab/simba/blob/master/images/dpk_anno.PNG "DPK_4")

2. Enter an Annotation Output Name in the first entry box. *Note: Do not enter a file-ending, and avoid spaces in the annotation output name. For example, if I wish the annotation set file to be called MyAnnotationSet.h5, then I enter MyAnnotationSet in the Annotation Output Name entry box. 

3. Click on `Create Annotation`. A message in the main SimBA terminal window will be printed to let you know when the annotation set creation is complete. To confirm that the annotation file has been correctly created, navigate to the `project_folder\logs\measures\dpk\*DeepPoseKit project name*\annotation_sets` folder and you should see all of your annotation file. If you named your file MyAnnotationSet in the Annotation Output Name entry box, you should see a file called *MyAnnotationSet.h5*. 

## Step 3: Using the DeepPoseKit Annotator in SimBA 

In the DeepPoseKit window, click on the `Annotator` tab and you should see the following window:

![](https://github.com/sgoldenlab/simba/blob/master/images/DPK_4.png "DPK_4")

Click on `Browse File` and select your annotation *.h5* file located in the `annotation_sets` directory. Click on `Run` and the following windows should pop open, with instructions displayed on the right: 

![](https://github.com/sgoldenlab/simba/blob/master/images/DPK_annotator.PNG "DPK_4")

After the annotations are complete, press `Esc` to save your annotations. 

## Step 4: Training DeepPoseKit models in SimBA

Click on the `Train model` tab and ou should see the following window.

![](https://github.com/sgoldenlab/simba/blob/master/images/DPK_5.png "DPK_5")

For documentation on the different parameters and network architectures available, see Jake Garvings' [DeepPosekit](https://github.com/jgraving/DeepPoseKit) documentation. In SimBA, most of the entry boxes and user-defined settings defaults to the DeepPoseKit suggested defults. To chooose the neural network archictecture, click on the `NN_architecture` drop-down menu and you should see the networks available in DeepPoseKit:

![](https://github.com/sgoldenlab/simba/blob/master/images/DPK_6.png "DPK_6")

To access architecture-specific settings, click on the `Model Settings` button. Depending on the chosen architecture, you should see one of the three possible menus:

| StackedDenseNet/Hourglass  | DeepLabCut |LEAP |
| ------------- | ------------- |------------- |
| <img src="https://github.com/sgoldenlab/simba/blob/master/images/DPK_7.png" width="200"/>  | <img src="https://github.com/sgoldenlab/simba/blob/master/images/DPK_8.png" width="200"/> |  <img src="https://github.com/sgoldenlab/simba/blob/master/images/DPK_9.png" width="200"/> |

Again, see Jake Garvings' [DeepPosekit](https://github.com/jgraving/DeepPoseKit) documentation for the model-specific setting in the [DeepLabCut](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/DeepLabCut.html#references), [StackedDenseNet](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/StackedDenseNet.html#references), [StackedHourglass](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/StackedHourglass.html#references), and [LEAP](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/LEAP.html#references). 

Once all the information is filled in, click on `Train model` and the model training will start. 

New models will be saved in the `project_folder\logs\measures\dpk\*DeepPoseKit project name*\models` sub-directory. 

## Step 5: Analyzing videos using DeepPoseKit in SimBA

1. To use a DeepPoseKit-generated model to track animals in new videos, navigate to the `[Predict new video]` tab. 

![](https://github.com/sgoldenlab/simba/blob/master/images/DPK_menu_predict.PNG "DPK_6")

Next to **Model path**, click on `Browse File` to select your DeepPoseKit-generated model. If the model was generated in SimBA, the model file should be located in the `project_folder\logs\measures\dpk\*DeepPoseKit project name*\models` sub-directory (see Step 3, above).

2. Next to **Video folder**, click on `Browse Folder` to select the folder containing your *.mp4* video files that you wish to analyze. Then click on `Predict` to start analyzing your videos. 

Once complete, a message will be printed in in main SimBA console window. The output files, containing all your body-part locatations in each frame of the video, will be located in the `project_folder\logs\measures\dpk\*DeepPoseKit project name*\predictions` sub-directory. 

You can now go ahead and import these files into SimBA to [generate behavioral predictive classifiers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#scenario-1-from-scratch) and/or [anlyze behavioral repetoirs in the videos using already created predictive classifiers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md)

## Step 6 Visualize tracking using DeepPoseKit in SimBA

To generate a visualization of analyzed videos, navigate to the `[Visualiza video]` tab.

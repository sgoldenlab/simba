# SimBA (Simple Behavioral Analysis)

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-pink.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Gitter chat](https://badges.gitter.im/USER/REPO.png)](https://gitter.im/SimBA-Resource/community)
[![Download: Weights](https://img.shields.io/badge/Download-Weights-orange.svg)](https://osf.io/5t4y9/)
[![SimBA: listserv](https://img.shields.io/static/v1?label=SimBA&message=listserv&color=blue)](https://docs.google.com/forms/d/e/1FAIpQLSfjbjae0XqNcl7GYOxmqvRsCveG-cmf4p4hBNNJ8gu5vPLHng/viewform)
[![DOI](https://zenodo.org/badge/206670333.svg)](https://zenodo.org/badge/latestdoi/206670333)

![alt-text-1](/images/SimBA_logo_4.jpg "simba logo")

## April-25-2020: SimBA pre-print manuscript release

A pre-print SimBA manuscript on bioRxiv! The manuscript details the use of SimBA for generation of social predictive classifiers in rat and mouse resident-intruder protocols - please check it out using the link below. All data, pose-estimation models, and the final classifiers generated in the manuscript, can be accessed through our [OSF repository](https://osf.io/tmu6y/) and through the [Resource](https://github.com/sgoldenlab/simba#resource-) menu further down this page.

**[Simple Behavioral Analysis (SimBA) – an open source toolkit for computer classification of complex social behaviors in experimental animals](https://www.biorxiv.org/content/10.1101/2020.04.19.049452v2)**

Please join our [Gitter chat](https://gitter.im/SimBA-Resource/community) if you have any questions, or even if you would simply like to discuss potential applications for SimBA in your work.  Please come by, stay inside, wash your hands, and check on your lab mates reguarly!

## March-05-2020: SimBA version 1.1 release
### New Features
- Region of Interest support (ROI Module) - [Documentation](/docs/ROI_tutorial.md)
- DeepPoseKit support (DPK Module) - [Documentation](/docs/DeepPoseKit_in_SimBA.md)
- SimBA accepts user-defined pose-configurations (Flexible Annotation Module) - [Documentation](/docs/Pose_config.md)
- Interactive classifer discrimination thresholding - [Documentation](/docs/validation_tutorial.md#validate-model-on-single-video)
- Individual discrimination thresholds for classifiers - [Documentation](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data)
- Heatmap visualizations -[Documentation](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-5--visualizing-machine-predictions)
- Multi-crop tool - [Documentation](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#multi-crop-videos)
- Post-classification validation method for false-positives - [Documentation](/docs/classifier_validation.md#classifier-validation)
- Many, many, many bug-fixes

Several excellent computational frameworks exist that enable high-throughput and consistent tracking of freely moving unmarked animals. Here we introduce and distribute a plug-and play pipeline that enabled users to use these pose-estimation approaches in combination with behavioral annotatation and generatation of supervised machine-learning behavioral predictive classifiers. We have developed this pipeline for the analysis of complex social behaviors, but have included the flexibility for users to generate predictive classifiers across other behavioral modalities with minimal effort and no specialized computational background.  

SimBA does not require computer science and programing experience, and SimBA is optimized for wide-ranging video acquisition parameters and quality. SimBA is written for Microsoft Windows. We may be able to provide support and advice for specific use instances, especially if it benefits multiple users and advances the scope of SimBA. Feel free to post issues and bugs here or contact us directly and we'll work on squashing them as they appear. We hope that users will contribute to the community!

- The SimBA pipeline requires no programing knowledge 
- Specialized commercial or custom-made equipment is not required
- Extensive annotations are not required
- The pipeline is flexible and can be used to create and validate classifiers for different behaviors and environments
- Currently included behavioral classifiers have been validated in mice and rats
- SimBA is written for Windows

**SimBA currently does not support analysis of video recordings of multiple similarly colored animals. SimBA provides several validated classifer libraries using videos filmed from above at 90° angle with pose-estimation data from 8 body parts per animal. SimBA now accepts any user-defined pose-estimation annotation schemes with the inclusion of the [Flexible Annotation Module in v1.1](https://github.com/sgoldenlab/simba/blob/master/docs/Pose_config.md)**. 

**Installation note:** SimBA can be installed either with TensorFlow compatability (for generating DeepLabCut and DeepPoseKit pose-estimation models), or without TensorFlow (for stand-alone use with classifiers and other functions). Please choose the appropriate branch for your needs, more details are found in the [Installation Documentation](https://github.com/sgoldenlab/simba/blob/master/README.md#installation-%EF%B8%8F).

**Listserv for release information:** If you would like to receive notification for new releases of SimBA, please **[fill out this form](https://forms.gle/R47RWN4stNSJBj9D9)** and you will be added to the listserv.

#### Mouse
![](https://github.com/sgoldenlab/simba/blob/master/images/mouse_videos.gif)

#### Rat
![](https://github.com/sgoldenlab/simba/blob/master/images/rat_videos.gif)

#### SimBA GUI workflow
![](https://github.com/sgoldenlab/simba/blob/master/images/SimBA_tkinter_3.png)


## Pipeline 👷
![](https://github.com/sgoldenlab/simba/blob/master/images/overallflow.PNG)

## Documentation: General methods

### Step 1: [Pre-process videos](docs/tutorial_process_videos.md) 

### Step 2: [Create tracking model and generate pose-estimation data](docs/Tutorial_DLC.md) 

### Step 3: [Building classfier(s)](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-6-label-behavior)

### Step 4: [Analysis/Visualization](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-9-analyze-machine-results)

### [Click here for the full *generic* tutorial on building classifiers in SimBA](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md).

## Scenario tutorials

To faciliate the initial use of SimBA, we provide several use scenarios. We have created these scenarios around a hypothetical experiment that take a user from initial use (completely new start) all the way through analyzing a complete experiment and then adding additional experimental datasets to an initial project.

### Scenario 1: [Building classifiers from scratch](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md)

### Scenario 2: [Using a classifier on new experimental data](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md)

### Scenario 3: [Updating a classifier with further annotated data](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario3.md)

### Scenario 4: [Analyzing and adding new Experimental data to a previously started project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario4_new.md)


## Installation ⚙️

- [Install SimBA](docs/installation.md)

## Tutorial 📚
- [Process video using SimBA tools](docs/Tutorial_tools.md) 🔨
- [Batch pre-process video using SimBA](docs/tutorial_process_videos.md) 🏭
- [Using DeepPoseKit in SimBA](docs/DeepPoseKit_in_SimBA.md) 📙
- [Using DeepLabCut through SimBA](docs/Tutorial_DLC.md) 📗
- [SimBA generic tutorial](docs/tutorial.md) 📘
- [SimBA behavioral annotation interface](docs/labelling_aggression_tutorial.md) 🏷️
- [Using user-defined ROIs in SimBA](/docs/ROI_tutorial.md) 🗺️
- [Recommended hardware](https://github.com/sgoldenlab/simba/blob/master/misc/system_requirements.md) 🖥️

## Resource 💾

All data (classifiers etc.) is available on our [Open Science Framework repository](https://osf.io/tmu6y/). For a schematic overview of the data respository folder structure (as of March-20-2020), click [HERE](https://github.com/sgoldenlab/simba/blob/master/images/OSF_folder_structure_031820.jpg).

### Models
Below is a link to download trained  behavior classification models to apply it on your dataset
- [Random forest models](https://osf.io/kwge8/) 🌲

### SimBA visualization examples
- [YouTube playlist](https://www.youtube.com/playlist?list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl) 📺

### Labelled images
- [Annotated images for tracking models](https://osf.io/sr3ck/) 📷

### Tracking weights
- [DeepLabCut tracking weights](https://osf.io/sr3ck/) 🏋️

### Golden Lab webpage
- [Sam Golden Lab UW](https://goldenneurolab.com/) 🧪🧫🐁



## License 📃
This project is licensed under the GNU Lesser General Public License v3.0. Note that the software is provided "as is", without warranty of any kind, express or implied. If you use the code or data, please cite us :)

## References 📜



[![Foo](https://github.com/sgoldenlab/simba/blob/master/images/cos_center_logo_small.original.png)](https://osf.io/d69jt/) [![Foo](https://github.com/sgoldenlab/simba/blob/master/images/twitter.png)](https://twitter.com/GoldenNeuron?s=20)

## Contributors 🤼
- [Simon Nilsson](https://github.com/sronilsson)
- [Jia Jie Choong](https://github.com/inoejj)
- [Sophia Hwang](https://github.com/sophihwang26)
- [Xiaoyu Tong](https://github.com/Xiaoyu-Tong)

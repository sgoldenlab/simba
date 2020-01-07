# SimBA (Simple Behavioral Analysis)

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-pink.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Gitter chat](https://badges.gitter.im/USER/REPO.png)](https://gitter.im/SimBA-Resource/community)
[![Download: Weights](https://img.shields.io/badge/Download-Weights-orange.svg)](https://osf.io/5t4y9/)

![alt-text-1](/images/SimBA_logo_4.jpg "simba logo")

Several excellent computational frameworks exist that enable high-throughput and consistent tracking of freely moving unmarked animals. Here we introduce and distribute a plug-and play pipeline that enabled users to use these pose-estimation approaches in combination with behavioral annotatation and generatation of supervised machine-learning behavioral predictive classifiers. We have developed this pipeline for the analysis of complex social behaviors, but have included the flexibility for users to generate predictive classifiers across other behavioral modalities with minimal effort and no specialized computational background.  

SimBA does not require computer science and programing experience, and SimBA is optimized for wide-ranging video acquisition parameters and quality. SimBA is written for Microsoft Windows. We may be able to provide support and advice for specific use instances, especially if it benefits multiple users and advances the scope of SimBA. Feel free to post issues and bugs here or contact us directly and we'll work on squashing them as they appear. We hope that users will contribute to the community!

- The SimBA pipeline requires no programing knowledge 
- Specialized commercial or custom-made equipment is not required
- Extensive annotations are not required
- The pipeline is flexible and can be used to create and validate classifiers for different behaviors and environments
- Currently included behavioral classifiers have been validated in mice and rats
- SimBA is written for Windows

SimBA currently does not support analysis of video recordings of multiple similarly colored animals, and is validated using videos filmed from above at 90° angle using pose-estimation data from 8 body parts per animal. However we and others are developeing multi-animal tracking of similarly colored and sized animals, and multiple recording angles supported! :muscle: We also include other body -part tracking schemes within the GUI pipeline (i.e., 1 or 2 mice, 3 to 8 body parts per mouse), but please consider these a work in progress. 

#### Mouse
![](https://github.com/sgoldenlab/simba/blob/master/images/mouse_videos.gif)

#### Rat
![](https://github.com/sgoldenlab/simba/blob/master/images/rat_videos.gif)

#### SimBA GUI workflow
![](https://github.com/sgoldenlab/simba/blob/master/images/SimBA_tkinter.png)


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

### Scenario 4: [Analyzing and adding new Experimental data to a previously started project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario3.md)


## Installation ⚙️

- [Install SimBA](docs/installation.md)

## Tutorial 📚
- [Process video using SimBA tools](docs/Tutorial_tools.md) 🔨
- [Batch pre-process video using SimBA](docs/tutorial_process_videos.md) 🏭
- [Using DeepLabCut through SimBA](docs/Tutorial_DLC.md) 📗
- [SimBA generic tutorial](docs/tutorial.md) 📘
- [SimBA behavioral annotation interface](docs/labelling_aggression_tutorial.md) 🏷️

## Resource 💾

### Models
Below is a link to download trained models to apply it on your dataset
- [Random forest models](https://osf.io/d69jt/) 🌲

### SimBA visualization examples
- [YouTube playlist](https://www.youtube.com/playlist?list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl) 📺

### Labelled images
- [DeepLabCut labelled images](https://osf.io/uhjzf/) 📷

### Tracking weights
- [DeepLabCut tracking weights](https://osf.io/5t4y9/) 🏋️

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
